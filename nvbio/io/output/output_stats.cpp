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

#include <nvbio/io/output/output_stats.h>
#include <nvbio/io/output/output_types.h>

namespace nvbio {
namespace io {

// keep track of alignment statistics for a given alignment
// xxxnsubtil: it's unclear to me whether the original code expected the first mate
// (i.e., the 'alignment' argument) to always be the anchor and the second to always
// be the opposite mate
void IOStats::track_alignment_statistics(const AlignmentData& alignment,
                                         const AlignmentData& mate,
                                         const uint8 mapq)
{
    n_reads++;

    if (alignment.best->is_paired())
    {
        // keep track of mapping quality histogram
        paired.mapq_bins[mapq]++;

        // count this read as mapped
        paired.n_mapped++;

        if (!alignment.second_best->is_paired())
        {
            // we only have one score; count as a unique alignment
            paired.n_unique++;
        }
        else
        {
            // we have two best scores, which implies two (or more) alignment positions
            // count as multiple alignment
            paired.n_multiple++;

            // compute final alignment score
            const int32 first  = alignment.best->score()        + mate.best->score();
            const int32 second = alignment.second_best->score() + mate.second_best->score();

            // if the two best scores are equal, count as ambiguous
            if (first == second)
                paired.n_ambiguous++;
            else {
                // else, the first score must be higher...
                NVBIO_CUDA_ASSERT(first > second);
                /// ... which counts as a nonambiguous alignment
                paired.n_unambiguous++;
            }

            // compute edit distance scores
            const uint32 first_ed  = alignment.best->ed()        + mate.best->ed();
            const uint32 second_ed = alignment.second_best->ed() + mate.second_best->ed();

            // update best edit-distance histograms
            if (first_ed < paired.mapped_ed_histogram.size())
            {
                paired.mapped_ed_histogram[first_ed]++;
                if (alignment.best->m_rc)
                {
                    paired.mapped_ed_histogram_fwd[first_ed]++;
                } else {
                    paired.mapped_ed_histogram_rev[first_ed]++;
                }
            }

            // track edit-distance correlation
            if (first_ed + 1 < 64)
            {
                if (second_ed + 1 < 64)
                {
                    paired.mapped_ed_correlation[first_ed + 1][second_ed + 1]++;
                } else {
                    paired.mapped_ed_correlation[first_ed + 1][0]++;
                }
            }
        }
    }
    else
    {
        //
        // track discordand alignments separately for each mate
        //

        const AlignmentData& alignment1 = alignment.best->mate() == 0 ? alignment : mate;
        const AlignmentData& alignment2 = alignment.best->mate() == 0 ? mate      : alignment;

        track_alignment_statistics( &mate1, alignment1, mapq );
        track_alignment_statistics( &mate2, alignment2, mapq );
    }
}

void IOStats::track_alignment_statistics(AlignmentStats*        mate,
                                         const AlignmentData&   alignment,
                                         const uint8            mapq)
{
    // check if the mate is aligned
    if (!alignment.best->is_aligned())
    {
        mate->mapped_ed_correlation[0][0]++;
        return;
    }

    // count this read as mapped
    mate->n_mapped++;

    // keep track of mapping quality histogram
    mate->mapq_bins[mapq]++;

    if (!alignment.second_best->is_aligned())
    {
        // we only have one score; count as a unique alignment
        mate->n_unique++;
    }
    else
    {
        // we have two best scores, which implies two (or more) alignment positions
        // count as multiple alignment
        mate->n_multiple++;

        // compute final alignment score
        const int32 first  = alignment.best->score();
        const int32 second = alignment.second_best->score();

        // if the two best scores are equal, count as ambiguous
        if (first == second)
            mate->n_ambiguous++;
        else {
            // else, the first score must be higher...
            NVBIO_CUDA_ASSERT(first > second);
            /// ... which counts as a nonambiguous alignment
            mate->n_unambiguous++;
        }

        // compute edit distance scores
        const uint32 first_ed  = alignment.best->ed();
        const uint32 second_ed = alignment.second_best->ed();

        // update best edit-distance histograms
        if (first_ed < mate->mapped_ed_histogram.size())
        {
            mate->mapped_ed_histogram[first_ed]++;
            if (alignment.best->m_rc)
            {
                mate->mapped_ed_histogram_fwd[first_ed]++;
            } else {
                mate->mapped_ed_histogram_rev[first_ed]++;
            }
        }

        // track edit-distance correlation
        if (first_ed + 1 < 64)
        {
            if (second_ed + 1 < 64)
                mate->mapped_ed_correlation[first_ed + 1][second_ed + 1]++;
            else
                mate->mapped_ed_correlation[first_ed + 1][0]++;
        }
    }
}

}
}
