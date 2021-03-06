/*
 * nvbio
 * Copyright (C) 2012-2014, NVIDIA Corporation
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * version 2 as published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 */

#include <nvBowtie/bowtie2/cuda/select.h>
#include <nvbio/basic/algorithms.h>

namespace nvbio {
namespace bowtie2 {
namespace cuda {

__global__
void select_init_kernel(
    const uint32                            count,
    SeedHitDequeArrayDeviceView             hits,
    uint32*                                 trys,
    const ParamsPOD                         params)
{
    const uint32 thread_id = threadIdx.x + BLOCKDIM*blockIdx.x;
    if (thread_id >= count) return;

    // initialize the number of trys
    if (trys)
        trys[ thread_id ] = params.max_effort_init;

    // initialize the probability trees
    if (params.randomized)
    {
        typedef SumTree<float*> ProbTree;

        // check whether we have a non-zero number of hits, otherwise we can go home
        const uint32 n_hits = hits.get_size( thread_id );
        if (n_hits == 0u)
            return;

        // build the probability tree
        float*         hit_probs = hits.get_probs( thread_id );
        const SeedHit* hit_data = hits.get_data( thread_id );

        for (uint32 i = 0; i < n_hits; ++i)
        {
            const SeedHit hit = hit_data[i];
            const uint2 range = hit.get_range();
            hit_probs[i] = 1.0f / (float(range.y - range.x) *
                                   float(range.y - range.x));
        }
        if (params.top_seed)
            hit_probs[0] = 0.0f; // assign zero probability to the top hit, as we will deterministically visit it

        // setup the tree
        ProbTree prob_tree( nvbio::max( n_hits, 1u ), hit_probs );
        prob_tree.setup();
    }
}

//
// Initialize the hit-selection pipeline
//
void select_init(
    const uint32                    count,
    SeedHitDequeArrayDeviceView     hits,
    uint32*                         trys,
    const ParamsPOD                 params)
{
    const int blocks = (count + BLOCKDIM-1) / BLOCKDIM;

    select_init_kernel<<<blocks, BLOCKDIM>>>(
        count,
        hits,
        trys,
        params );
}

///
/// select next hit extensions for all-mapping
///
__global__ 
void select_all_kernel(
    const uint64                    begin,
    const uint32                    count,
    const uint32                    n_reads,
    const uint32                    n_hit_ranges,
    const uint64                    n_hits,
    SeedHitDequeArrayDeviceView     hits,
    const uint32*                   hit_count_scan,
    const uint64*                   hit_range_scan,
          HitQueuesDeviceView       hit_queues)
{
    const uint32 thread_id = threadIdx.x + BLOCKDIM*blockIdx.x;
    if (thread_id >= count) return;

    // each thread is responsible to process a hit, starting from 'begin'
    const uint64 global_hit_id = begin + thread_id;

    // do a binary search, looking for thread_id in hit_range_scan,
    // to find the corresponding seed hit.
    const uint32 global_range_id = uint32( upper_bound( global_hit_id, hit_range_scan, n_hit_ranges ) - hit_range_scan );

    // at this point we can figure out which read this range belongs to
    const uint32 read_id = uint32( upper_bound( global_range_id, hit_count_scan, n_reads ) - hit_count_scan );

    // now we have everything we need to access the proper hit_data
    const SeedHit* hit_data = hits.get_data( read_id );

    // compute the local range index within this read
    const uint32 count_offset = read_id ? hit_count_scan[ read_id-1 ] : 0u;
    const uint32 range_id = global_range_id - count_offset;

    const SeedHit* hit = &hit_data[ range_id ];

    // compute the local hit index within this range
    const uint64 range_offset = global_range_id ? hit_range_scan[ global_range_id-1 ] : 0u;
    const uint32 hit_id = uint32( global_hit_id - range_offset );

    const uint32 r_type = hit->get_readtype() ? 1u : 0u;

    HitReference<HitQueuesDeviceView> out_hit( hit_queues, thread_id );
    out_hit.loc     = hit->front() + hit_id;
    out_hit.seed    = packed_seed( hit->get_posinread(), hit->get_indexdir(), r_type, 0u );
    out_hit.read_id = read_id;
}

///
/// select next hit extensions from the top seed ranges
///
__global__ 
void select_n_from_top_range_kernel(
    const uint32                    begin,
    const uint32                    count,
    const uint32                    n_reads,
    SeedHitDequeArrayDeviceView     hits,
    const uint32*                   hit_range_scan,
          uint32*                   loc_queue,
          packed_seed*              seed_queue,
          packed_read*              read_info)
{
    const uint32 thread_id = threadIdx.x + BLOCKDIM*blockIdx.x;
    if (thread_id >= count) return;

    // each thread is responsible to process a hit, starting from 'begin'
    const uint64 global_hit_id = begin + thread_id;

    // do a binary search, looking for thread_id in hit_range_scan,
    // to find the corresponding read id.
    const uint32 read_id = uint32( upper_bound( global_hit_id, hit_range_scan, n_reads ) - hit_range_scan );

    // now we have everything we need to access the proper hit_data
    const SeedHit* hit_data = hits.get_data( read_id );

    // fetch the top range
    const SeedHit* hit = &hit_data[0];

    // compute the local hit index within this range
    const uint64 range_offset = global_hit_id ? hit_range_scan[ read_id-1 ] : 0u;
    const uint32 hit_id = uint32( global_hit_id - range_offset );

    const uint32 r_type = hit->get_readtype() ? 1u : 0u;

    loc_queue[ thread_id ]  = hit->front() + hit_id;
    seed_queue[ thread_id ] = packed_seed( hit->get_posinread(), hit->get_indexdir(), r_type, 0u );
    read_info[ thread_id ]  = packed_read( read_id );
}

void select_all(
    const uint64                        begin,
    const uint32                        count,
    const uint32                        n_reads,
    const uint32                        n_hit_ranges,
    const uint64                        n_hits,
    const SeedHitDequeArrayDeviceView   hits,
    const uint32*                       hit_count_scan,
    const uint64*                       hit_range_scan,
          HitQueuesDeviceView           hit_queues)
{
    const int blocks = (count + BLOCKDIM-1) / BLOCKDIM;

    select_all_kernel<<<blocks, BLOCKDIM>>>(
        begin,
        count,
        n_reads,
        n_hit_ranges,
        n_hits,
        hits,
        hit_count_scan,
        hit_range_scan,
        hit_queues );
}

//
// Prune the set of active reads based on whether we found the best alignments
//
__global__ 
void prune_search_kernel(
    const uint32                                    max_dist,
    const nvbio::cuda::PingPongQueuesView<uint32>   queues,
    const io::BestAlignments*                       best_data)
{
    __shared__ volatile uint32 sm_broadcast[BLOCKDIM >> 5];
    volatile uint32& warp_broadcast = sm_broadcast[ warp_id() ];

    const uint32 thread_id = threadIdx.x + BLOCKDIM*blockIdx.x;
    if (thread_id >= queues.in_size) return;
    const uint32 read_id = queues.in_queue[ thread_id ];

    // check whether we can stop searching
    const io::BestAlignments best = best_data[ read_id ];
    if (best.has_second() && best.second_score() <= max_dist)
        return;

    // enqueue only if a valid seed
    const uint32 slot = alloc( queues.out_size, &warp_broadcast );
    NVBIO_CUDA_ASSERT( slot < queues.in_size );
    queues.out_queue[ slot ] = read_id;
}

//
// Prune the set of active reads based on whether we found the best alignments
//
void prune_search(
    const uint32                                    max_dist,
    const nvbio::cuda::PingPongQueuesView<uint32>   queues,
    const io::BestAlignments*                       best_data)
{
    const int blocks = (queues.in_size + BLOCKDIM-1) / BLOCKDIM;

    prune_search_kernel<<<blocks, BLOCKDIM>>>(
        max_dist,
        queues,
        best_data );
}

} // namespace cuda
} // namespace bowtie2
} // namespace nvbio
