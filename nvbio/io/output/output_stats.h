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

#pragma once

#include <nvbio/io/output/output_types.h>
#include <nvbio/basic/timer.h>

#include <vector>

namespace nvbio {
namespace io {

struct AlignmentStats
{
    AlignmentStats()
        : n_mapped(0),
          n_ambiguous(0),
          n_unambiguous(0),
          n_unique(0),
          n_multiple(0),
          mapped_ed_histogram(4096, 0),
          mapped_ed_histogram_fwd(4096, 0),
          mapped_ed_histogram_rev(4096, 0)
    {
        for(uint32 c = 0; c < 64; c++)
            mapq_bins[c] = 0;

        for(uint32 c = 0; c < 64; c++)
        {
            for(uint32 d = 0; d < 64; d++)
                mapped_ed_correlation[c][d] = 0;
        }
    }

    // mapping quality stats
    uint64 mapq_bins[64];

    // mapping stats
    uint32          n_mapped;       // number of mapped reads

    uint32          n_ambiguous;    // number of reads mapped to more than one place with the same score
    uint32          n_unambiguous;  // number of reads with a single best score (even if mapped to multiple locations)

    uint32          n_unique;       // number of reads mapped to a single location
    uint32          n_multiple;     // number of reads mapped to more than one location

    // edit distance scoring histograms
    std::vector<uint32> mapped_ed_histogram;        // aggregate histogram of edit-distance scores per read
    std::vector<uint32> mapped_ed_histogram_fwd;    // histogram of edit-distance scores for reads mapped to the forward sequence
    std::vector<uint32> mapped_ed_histogram_rev;    // histogram of edit-distance scores for reads mapped to the reverse-complemented sequence

    // edit distance correlation (xxxnsubtil: what exactly does this measure?)
    uint32  mapped_ed_correlation[64][64];
};

// I/O statistics that we collect
struct IOStats
{
    int   alignments_DtoH_count; // number of reads transferred from device to host
    float alignments_DtoH_time;  // time spent moving alignment data from device to host

    // how many reads we output
    uint32 n_reads;

    // detailed mapping stats
    AlignmentStats  paired;
    AlignmentStats  mate1;
    AlignmentStats  mate2;

    // time series for tracking each OutputFile::process() call
    TimeSeries output_process_timings;

    IOStats()
        : alignments_DtoH_count(0),
          alignments_DtoH_time(0.0),
          n_reads(0)
    {}

    // paired-end alignment
    void track_alignment_statistics(const AlignmentData& alignment,
                                    const AlignmentData& mate,
                                    const uint8 mapq);

    // single-end alignment
    void track_alignment_statistics(AlignmentStats*      mate,
                                    const AlignmentData& alignment,
                                    const uint8          mapq);

    // single-end alignment
    void track_alignment_statistics(const AlignmentData& alignment,
                                    const uint8          mapq);
};

} // namespace io
} // namespace nvbio
