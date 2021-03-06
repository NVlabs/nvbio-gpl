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

#include <nvbio/basic/cuda/ldg.h>
#include <nvBowtie/bowtie2/cuda/pipeline_states.h>
#include <nvBowtie/bowtie2/cuda/select.h>
#include <nvBowtie/bowtie2/cuda/locate.h>
#include <nvBowtie/bowtie2/cuda/score.h>
#include <nvBowtie/bowtie2/cuda/reduce.h>
#include <nvBowtie/bowtie2/cuda/traceback.h>

namespace nvbio {
namespace bowtie2 {
namespace cuda {

template <typename scoring_tag>
void Aligner::all(
    const Params&                           params,
    const fmi_type                          fmi,
    const rfmi_type                         rfmi,
    const UberScoringScheme&                input_scoring_scheme,
    const io::SequenceDataDevice&           reference_data,
    const io::FMIndexDataDevice&            driver_data,
    const io::SequenceDataDevice&           read_data,
    Stats&                                  stats)
{
    // prepare the scoring system
    typedef typename ScoringSchemeSelector<scoring_tag>::type           scoring_scheme_type;
    typedef typename scoring_scheme_type::threshold_score_type          threshold_score_type;

    scoring_scheme_type scoring_scheme = ScoringSchemeSelector<scoring_tag>::scheme( input_scoring_scheme );

    // cast the reads to use proper iterators
    const read_batch_type reads( plain_view( read_data ) );

    Timer timer;
    Timer global_timer;

    const uint32 count = read_data.size();
    //const uint32 band_len = band_length( params.max_dist );

    seed_queues.in_size = count;

    thrust::copy(                                   // this is only done to pass a list of tasks to the
        thrust::make_counting_iterator(0u),         // score_all method
        thrust::make_counting_iterator(0u) + count,
        seed_queues.in_queue.begin() );

    //
    // Unlike Bowtie2, in the context of all-mapping we perform a single seed & extension pass.
    // However, during seed mapping, if we find too many hits for a given seeding pattern,
    // we try another one with shifted seeds.
    //

    uint64 n_alignments = 0;

    uint32 max_seeds = 0u;
    for (uint32 s = read_data.min_sequence_len(); s <= read_data.max_sequence_len(); ++s)
        max_seeds = nvbio::max( max_seeds, uint32( s / params.seed_freq(s) ) );

    for (uint32 seed = 0; seed < max_seeds; ++seed)
    {
        // initialize the seed hit counts
        hit_deques.clear_deques();

        //
        // perform mapping
        //
        if (params.allow_sub == 0)
        {
            timer.start();

            SeedHitDequeArrayDeviceView hits = hit_deques.device_view();

            map_exact( // TODO: allow inexact mapping as well?
                reads, fmi, rfmi,
                hits,
                make_uint2( seed, seed+1 ),
                params );

            optional_device_synchronize();
            nvbio::cuda::check_error("mapping kernel");

            timer.stop();
            stats.map.add( count, timer.seconds() );
        }
        else
        {
            timer.start();

            SeedHitDequeArrayDeviceView hits = hit_deques.device_view();

            map_approx( // TODO: allow inexact mapping as well?
                reads, fmi, rfmi,
                hits,
                make_uint2( seed, seed+1 ),
                params );

            optional_device_synchronize();
            nvbio::cuda::check_error("mapping kernel");

            timer.stop();
            stats.map.add( count, timer.seconds() );
        }

        // take some stats on the hits we got
        if (params.keep_stats)
            keep_stats( reads.size(), stats );

        score_all(
            params,
            fmi,
            rfmi,
            input_scoring_scheme,
            scoring_scheme,
            reference_data,
            driver_data,
            read_data,
            count,
            seed_queues.raw_input_queue(),
            stats,
            n_alignments );
    }
}

template <typename scoring_scheme_type>
void Aligner::score_all(
    const Params&                           params,
    const fmi_type                          fmi,
    const rfmi_type                         rfmi,
    const UberScoringScheme&                input_scoring_scheme,
    const scoring_scheme_type&              scoring_scheme,
    const io::SequenceDataDevice&           reference_data,
    const io::FMIndexDataDevice&            driver_data,
    const io::SequenceDataDevice&           read_data,
    const uint32                            seed_queue_size,
    const uint32*                           seed_queue,
    Stats&                                  stats,
    uint64&                                 total_alignments)
{
    // prepare the scoring system
    //typedef typename scoring_scheme_type::threshold_score_type          threshold_score_type;

    //threshold_score_type threshold_score = scoring_scheme.threshold_score( params );
    const int32 score_limit = scoring_scheme.score_limit( params );

    Timer timer;
    Timer global_timer;

    const uint32 count = read_data.size();
    const uint32 band_len = band_length( params.max_dist );

    // cast the reads to use proper iterators
    const read_batch_type reads( plain_view( read_data ) );

    // cast the genome to use proper iterators
    const io::LdgSequenceDataView   genome_view( plain_view( reference_data ) );
    const genome_access_type        genome_access( genome_view );
    const uint32                    genome_len = genome_access.bps();
    const genome_iterator           genome     = genome_access.sequence_stream();

    //
    // At this point we have a queue full of reads, each with an associated set of
    // seed hits encoded as a (sorted) list of SA ranges.
    // For each read we need to:
    //      1. select some seed hit to process (i.e. a row in one of the SA ranges)
    //      2. locate it, i.e. converting from SA to linear coordinates
    //      3. score it
    //      4. if the score is acceptable, backtrack the extension
    // The whole process is carried out as a pipeline composed by 4 different stages
    // communicating through work queues.
    // At the end of each pipeline iteration, the reads which are still alive, in the
    // sense they contain more seed hits to process, are pushed to an output queue.
    // The output queue is then reused in the next round as the input queue, and
    // viceversa.
    //
    // NOTE: in all-mapping mode, we ultimately need to explore exahustively the set of seed
    // hits for each read, and report all the alignments found.
    // If we process the seed hits from all reads simultaneously, reporting all their
    // alignments sorted by read would require an external sorting pass, as the amount of
    // storage could be fairly dramatic.
    // However, we can also simply process all the seed hits from the first read first,
    // followed by all the ones from the second read, and so on.
    // This has also the advantage of sharing the same read data across many seed extensions,
    // lowering the overall memory traffic.
    // In order to distribute work evenly we can:
    //   run a scan on the number of SA ranges from each read (A)
    //   run a scan on all the SA range sizes (B)
    //   assign a thread per hit, and find the corresponding SA range by a binary search in (B),
    //   followed by another binary search in (A) to find the corresponding read.
    //

    // run a scan on the number of SA ranges from each read
    thrust::inclusive_scan( hit_deques.counts().begin(), hit_deques.counts().begin() + count, hits_count_scan_dvec.begin() );

    const uint32 n_reads = reads.size();

    // gather all the range sizes, sorted by read, in a compacted array
    const uint32 n_hit_ranges = hits_count_scan_dvec[ count-1 ];

    log_verbose(stderr, "    ranges     : %u\n", n_hit_ranges);

    SeedHitDequeArrayDeviceView hits = hit_deques.device_view();

    gather_ranges(
        n_hit_ranges,
        n_reads,
        hits,
        hits_count_scan_dptr,
        hits_range_scan_dptr );

    optional_device_synchronize();
    nvbio::cuda::check_error("gather ranges kernel");

    // run a scan on all the SA range sizes
    thrust::inclusive_scan( hits_range_scan_dvec.begin(), hits_range_scan_dvec.begin() + n_hit_ranges, hits_range_scan_dvec.begin() );

    const uint64 n_hits = hits_range_scan_dvec[ n_hit_ranges-1 ];
    log_verbose(stderr, "    hits       : %llu\n", n_hits);

    #if defined(NVBIO_CUDA_DEBUG)
    {
        uint64 total_hits = 0;
        thrust::host_vector<uint32>  h_counts = hit_deques.counts();
        thrust::host_vector<SeedHit> h_hits   = hit_deques.hits();
        for (uint32 i = 0; i < n_reads; ++i)
        {
            uint32 cnt = h_counts[i];
            for (uint32 j = 0; j < cnt; ++j)
            {
                const SeedHit hit = h_hits[ i + n_reads * j ];
                total_hits += hit.get_range().y - hit.get_range().x;
            }
        }
        if (n_hits != total_hits)
        {
            log_error(stderr, "  expected %llu hits, computed %llu\n", total_hits, n_hits);
            throw nvbio::runtime_error("invalid number of hits");
        }
    }
    #endif

    //
    // Loop through batches of BATCH_SIZE seed hits. The seed hits are extracted
    // from the compressed variable-sized list of SA ranges, located and scored.
    // If the extension is successful, they are queued to a ring-buffer for
    // back-tracking. The ring-buffer is twice as large as the batch size, so as
    // to allow temporary overflows.
    //

    uint32 buffer_offset = 0;
    uint32 buffer_count  = 0;

    typedef AllMappingPipelineState<scoring_scheme_type>     pipeline_type;

    pipeline_type pipeline(
        0u,
        reads,
        reads,
        genome_len,
        genome,
        fmi,
        rfmi,
        scoring_scheme,
        score_limit,
        *this );

    // setup the hits queue according to whether we select multiple hits per read or not
    scoring_queues.hits_index.setup( 1u, BATCH_SIZE );

    HitQueues& hit_queues = scoring_queues.hits;

//    uint32* score_idx_dptr = nvbio::device_view( hit_queues.ssa );

    for (uint64 hit_offset = 0; hit_offset < n_hits; hit_offset += BATCH_SIZE)
    {
        const uint32 hit_count = std::min( uint32(n_hits - hit_offset), BATCH_SIZE );

        timer.start();

        SeedHitDequeArrayDeviceView hits = hit_deques.device_view();

        select_all(
            hit_offset,
            hit_count,
            n_reads,
            n_hit_ranges,
            n_hits,
            hits,
            hits_count_scan_dptr,
            hits_range_scan_dptr,
            nvbio::device_view( hit_queues ) );

        optional_device_synchronize();
        nvbio::cuda::check_error("selecting kernel");

        timer.stop();
        stats.select.add( hit_count, timer.seconds() );

        // update pipeline
        pipeline.n_hits_per_read = 1u;
        pipeline.hits_queue_size = hit_count;
        pipeline.scoring_queues  = scoring_queues.device_view();

        timer.start();

        pipeline.idx_queue = sort_hi_bits(
            hit_count,
            nvbio::device_view( hit_queues.loc ) );

        timer.stop();
        stats.sort.add( hit_count, timer.seconds() );
        
        // and locate their position in linear coordinates
        locate_init(
            reads, fmi, rfmi,
            hit_count,
            pipeline.idx_queue,
            nvbio::device_view( hit_queues ),
            params );

        optional_device_synchronize();
        nvbio::cuda::check_error("locate-init kernel");

        locate_lookup(
            reads, fmi, rfmi,
            hit_count,
            pipeline.idx_queue,
            nvbio::device_view( hit_queues ),
            params );

        optional_device_synchronize();
        nvbio::cuda::check_error("locate-lookup kernel");

        timer.stop();
        stats.locate.add( hit_count, timer.seconds() );

        timer.start();

      #if 1
        // sort the selected hits by their linear genome coordinate
        // TODO: sub-sort by read_id/RC flag so as to (1) get better coherence,
        // (2) allow removing duplicate extensions
        // sort the selected hits by their SA coordinate
        pipeline.idx_queue = sort_hi_bits(
            hit_count,
            nvbio::device_view( hit_queues.loc ) );
      #else
        // access the selected hits by read id
        thrust::copy(
            thrust::make_counting_iterator(0u),
            thrust::make_counting_iterator(0u) + hit_count,
            idx_queue_dvec.begin() );

        pipeline.idx_queue = thrust::raw_pointer_cast( &idx_queue_dvec.front() );
      #endif

        timer.stop();
        stats.sort.add( hit_count, timer.seconds() );

        //
        // assign a score to all selected hits
        //
        timer.start();

        const uint32 n_alignments = nvbio::bowtie2::cuda::score_all(
            band_len,
            pipeline,
            params,
            buffer_offset + buffer_count,
            BATCH_SIZE*2 );

        optional_device_synchronize();
        nvbio::cuda::check_error("score kernel");

        timer.stop();
        stats.score.add( hit_count, timer.seconds() );

        total_alignments += n_alignments;
        log_verbose(stderr, "\r    alignments : %u (%.1fM)    %.1f%%                ", n_alignments, float(total_alignments)*1.0e-6f, 100.0f * float(hit_offset + hit_count)/float(n_hits));

        buffer_count += n_alignments;

        //
        // perform backtracking and compute CIGARs for the found alignments
        //
        while (buffer_count)
        {
            // if it's not the very last pass, break out of the loop as
            // soon as we have less than BATCH_SIZE elements to process
            if (hit_offset + hit_count < n_hits)
            {
                if (buffer_count < BATCH_SIZE)
                    break;
            }

            timer.start();

            const uint32 n_backtracks = std::min(
                buffer_count,
                BATCH_SIZE );

            {
                // copy the read infos to the plain output buffer
                {
                    ring_buffer_to_plain_array(
                        buffer_read_info_dptr,
                        BATCH_SIZE*2,
                        buffer_offset,
                        buffer_offset + buffer_count,
                        output_read_info_dptr );

                    optional_device_synchronize();
                    nvbio::cuda::check_error("ring-buffer copy kernel");
                }

                // sort the alignments by read-id & RC flag to gain coherence
                thrust::copy(
                    thrust::make_counting_iterator(0u),
                    thrust::make_counting_iterator(0u) + n_backtracks,
                    idx_queue_dvec.begin() );

                thrust::sort_by_key(
                    output_read_info_dvec.begin(),
                    output_read_info_dvec.begin() + n_backtracks,
                    idx_queue_dvec.begin() );
            }

            // initialize cigars & MDS pools
            cigar.clear();
            mds.clear();

            banded_traceback_all(
                n_backtracks,
                idx_queue_dptr,         // input indices
                buffer_offset,          // alignment ring-buffer offset
                BATCH_SIZE*2,           // alignment ring-buffer size
                output_alignments_dptr, // output alignments
                band_len,               // band length
                pipeline,               // pipeline state
                params );               // global params

            optional_device_synchronize();
            nvbio::cuda::check_error("backtracking kernel");

            finish_alignment_all(
                n_backtracks,
                idx_queue_dptr,             // input indices
                buffer_offset,              // alignment ring-buffer offset
                BATCH_SIZE*2,               // alignment ring-buffer size
                output_alignments_dptr,     // output alignments
                band_len,                   // band length
                pipeline,                   // pipeline state
                input_scoring_scheme.sw,    // always use Smith-Waterman to compute the final scores
                params );                   // global params

            optional_device_synchronize();
            nvbio::cuda::check_error("alignment kernel");

            timer.stop();
            stats.backtrack.add( n_backtracks, timer.seconds() );

            // TODO: insert a hook to output the alignments here

            buffer_count  -= n_backtracks;
            buffer_offset  = (buffer_offset + n_backtracks) % (BATCH_SIZE*2);
        }
    }
    log_verbose_nl(stderr);
}

} // namespace cuda
} // namespace bowtie2
} // namespace nvbio
