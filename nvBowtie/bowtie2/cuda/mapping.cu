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

#include <nvBowtie/bowtie2/cuda/mapping.h>

namespace nvbio {
namespace bowtie2 {
namespace cuda {

//
// For all i in [0, #seed hit ranges[, output the seed hit range size in
// out_ranges[i].
//
__global__ 
void gather_ranges_kernel(
    const uint32                        count,
    const uint32                        n_reads,
    const SeedHitDequeArrayDeviceView   hits,
    const uint32*                       hit_counts_scan,
          uint64*                       out_ranges)
{
    const uint32 thread_id = threadIdx.x + BLOCKDIM*blockIdx.x;
    if (thread_id >= count) return;

    // do a binary search, looking for thread_id in hit_counts_scan,
    // to find the corresponding read id.
    const uint32 read_id = uint32( upper_bound( thread_id, hit_counts_scan, n_reads ) - hit_counts_scan );

    // at this point we can figure out which seed hit / SA range this thread is
    // responsible of
    const uint32 count_offset = read_id ? hit_counts_scan[read_id-1] : 0u;

    const uint32 range_id = thread_id - count_offset;

    const SeedHit* hits_data = hits.get_data( read_id );

    const uint2 range = hits_data[ range_id ].get_range();
 
    // and we can compute the corresponding range size
    out_ranges[ thread_id ] = range.y - range.x;
}

//
// dispatch the call to gather_ranges_kernel
//
void gather_ranges(
    const uint32                        count,
    const uint32                        n_reads,
    const SeedHitDequeArrayDeviceView   hits,
    const uint32*                       hit_counts_scan,
          uint64*                       out_ranges)
{
    const int blocks = (count + BLOCKDIM-1) / BLOCKDIM;

    gather_ranges_kernel<<<blocks, BLOCKDIM>>>( count, n_reads, hits, hit_counts_scan, out_ranges );
}


//
// perform exact read mapping
//
void map_whole_read(
    const ReadsDef::type&                           read_batch,
    const FMIndexDef::type                          fmi,
    const FMIndexDef::type                          rfmi,
    const nvbio::cuda::PingPongQueuesView<uint32>   queues,
    SeedHitDequeArrayDeviceView                     hits,
    const ParamsPOD                                 params)
{
    map_whole_read_t( read_batch, fmi, rfmi, queues, hits, params );
}

//
// perform one run of exact seed mapping for all the reads in the input queue,
// writing reads that need another run in the output queue
//
void map_exact(
    const ReadsDef::type&                           read_batch,
    const FMIndexDef::type                          fmi,
    const FMIndexDef::type                          rfmi,
    const uint32                                    retry,
    const nvbio::cuda::PingPongQueuesView<uint32>   queues,
    SeedHitDequeArrayDeviceView                     hits,
    const ParamsPOD                                 params)
{
    map_exact_t( read_batch, fmi, rfmi, retry, queues, hits, params );
}

//
// perform multiple runs of exact seed mapping in one go and keep the best
//
void map_exact(
    const ReadsDef::type&                           read_batch,
    const FMIndexDef::type                          fmi,
    const FMIndexDef::type                          rfmi,
    SeedHitDequeArrayDeviceView                     hits,
    const uint2                                     seed_range,
    const ParamsPOD                                 params)
{
    map_exact_t( read_batch, fmi, rfmi, hits, seed_range, params );
}

//
// perform one run of approximate seed mapping for all the reads in the input queue,
// writing reads that need another run in the output queue
//
void map_approx(
    const ReadsDef::type&                           read_batch,
    const FMIndexDef::type                          fmi,
    const FMIndexDef::type                          rfmi,
    const uint32                                    retry,
    const nvbio::cuda::PingPongQueuesView<uint32>   queues,
    SeedHitDequeArrayDeviceView                     hits,
    const ParamsPOD                                 params)
{
    map_approx_t( read_batch, fmi, rfmi, retry, queues, hits, params );
}

//
// perform multiple runs of approximate seed mapping in one go and keep the best
//
void map_approx(
    const ReadsDef::type&                           read_batch,
    const FMIndexDef::type                          fmi,
    const FMIndexDef::type                          rfmi,
    SeedHitDequeArrayDeviceView                     hits,
    const uint2                                     seed_range,
    const ParamsPOD                                 params)
{
    map_approx_t( read_batch, fmi, rfmi, hits, seed_range, params );
}

//
// perform one run of seed mapping
//
void map(
    const ReadsDef::type&                           read_batch,
    const FMIndexDef::type                          fmi,
    const FMIndexDef::type                          rfmi,
    const uint32                                    retry,
    const nvbio::cuda::PingPongQueuesView<uint32>   queues,
    SeedHitDequeArrayDeviceView                     hits,
    const ParamsPOD                                 params)
{
    map_t( read_batch, fmi, rfmi, retry, queues, hits, params );
}

} // namespace cuda
} // namespace bowtie2
} // namespace nvbio
