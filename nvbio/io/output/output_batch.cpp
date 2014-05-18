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

#include <nvbio/io/output/output_batch.h>
#include <nvbio/io/fmi.h>
#include <nvbio/basic/timer.h>
#include <nvbio/basic/vector.h>

#include <stdio.h>
#include <stdarg.h>

namespace nvbio {
namespace io {

AlignmentResult::AlignmentResult()
{
    best[MATE_1] = Alignment::invalid();
    best[MATE_2] = Alignment::invalid();
    second_best[MATE_1] = Alignment::invalid();
    second_best[MATE_2] = Alignment::invalid();

    is_paired_end = false;
}
// copy scoring data to host, converting to io::AlignmentResult
void GPUOutputBatch::readback_scores(thrust::host_vector<io::AlignmentResult>& output,
                                     const AlignmentMate mate,
                                     const AlignmentScore score) const
{
    // copy alignment data into a staging buffer
    thrust::host_vector<io::BestAlignments> best_data_staging;
    nvbio::cuda::thrust_copy_vector(best_data_staging, best_data_dvec, count);

    // convert the contents of the staging buffer into io::AlignmentResult
    output.resize(count);
    for(uint32 c = 0; c < count; c++)
    {
        io::BestAlignments&      old_best_aln = best_data_staging[c];
        io::Alignment&           old_aln = (score == BEST_SCORE ? old_best_aln.m_a1 : old_best_aln.m_a2);
        io::AlignmentResult&    new_aln = output[c];

        if (score == BEST_SCORE)
        {
            new_aln.best[mate] = old_aln;
        } else {
            new_aln.second_best[mate] = old_aln;
        }

        if (mate == MATE_2)
        {
            new_aln.is_paired_end = true;
        }
    }
}

// copy CIGARs into host memory
void GPUOutputBatch::readback_cigars(io::HostCigarArray& host_cigar) const
{
    host_cigar.array = cigar.array;
    nvbio::cuda::thrust_copy_vector(host_cigar.coords, cigar.coords);
}

// copy MD strings back to the host
void GPUOutputBatch::readback_mds(nvbio::HostVectorArray<uint8>& host_mds) const
{
    host_mds = mds;
}

// extract alignment data for a given mate
// note that the mates can be different for the cigar, since mate 1 is always the anchor mate for cigars
AlignmentData CPUOutputBatch::get_mate(uint32 read_id, AlignmentMate mate, AlignmentMate cigar_mate)
{
    return AlignmentData(&best_alignments[read_id].best[mate],
                         &best_alignments[read_id].second_best[mate],
                         read_id,
                         read_data[mate],
                         &cigar[cigar_mate],
                         &mds[cigar_mate]);
}

// extract alignment data for the anchor mate
AlignmentData CPUOutputBatch::get_anchor(uint32 read_id)
{
    if (best_alignments[read_id].best[MATE_1].mate() == 0)
    {
        // mate 1 is the anchor
        return get_mate(read_id, MATE_1, MATE_1);
    } else {
        // mate 2 is the anchor
        return get_mate(read_id, MATE_2, MATE_1);
    }
}

// extract alignment data for the opposite mate
AlignmentData CPUOutputBatch::get_opposite_mate(uint32 read_id)
{
    if (best_alignments[read_id].best[MATE_1].mate() == 0)
    {
        // mate 2 is the opposite mate
        return get_mate(read_id, MATE_2, MATE_2);
    } else {
        // mate 1 is the opposite mate
        return get_mate(read_id, MATE_1, MATE_2);
    }
}

} // namespace io
} // namespace nvbio
