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

    // keep track of mapping quality histogram
    mapq_bins[mapq]++;

    if (!alignment.best->is_aligned())
    {
        // sanity check
        NVBIO_CUDA_ASSERT(alignment.second_best->is_mapped() == false);

        // if first anchor was not mapped, nothing was; count as unmapped
        mapped_ed_correlation[0][0]++;
        return;
    }

    // count this read as mapped
    n_mapped++;

    if (!alignment.second_best->is_aligned())
    {
        // we only have one score; count as a unique alignment
        n_unique++;
    } else {
        // we have two best scores, which implies two (or more) alignment positions
        // count as multiple alignment
        n_multiple++;

        // compute final alignment score
        int32 first  = alignment.best->score();
        int32 second = alignment.second_best->score();

        if (mate.valid)
        {
            first += mate.best->score();
            second += mate.second_best->score();
        }

        // if the two best scores are equal, count as ambiguous
        if (first == second)
            n_ambiguous++;
        else {
            // else, the first score must be higher...
            NVBIO_CUDA_ASSERT(first > second);
            /// ... which counts as a nonambiguous alignment
            n_unambiguous++;
        }

        // compute edit distance scores
        uint32 first_ed  = alignment.best->ed();
        uint32 second_ed = alignment.second_best->score();

        // update best edit-distance histograms
        if (first_ed < mapped_ed_histogram.size())
        {
            mapped_ed_histogram[first_ed]++;
            if (alignment.best->m_rc)
            {
                mapped_ed_histogram_fwd[first_ed]++;
            } else {
                mapped_ed_histogram_rev[first_ed]++;
            }
        }

        // track edit-distance correlation
        if (first_ed + 1 < 64)
        {
            if (second_ed + 1 < 64)
            {
                mapped_ed_correlation[first_ed + 1][second_ed + 1]++;
            } else {
                mapped_ed_correlation[first_ed + 1][0]++;
            }
        }
    }
}

}
}
