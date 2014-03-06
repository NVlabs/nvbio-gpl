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

#include <nvBowtie/bowtie2/cuda/defs.h>
#include <nvBowtie/bowtie2/cuda/params.h>
#include <nvbio/basic/timer.h>
#include <vector>
#include <deque>

namespace nvbio {
namespace bowtie2 {
namespace cuda {

typedef nvbio::TimeSeries KernelStats;

//
// Global statistics
//
struct Stats
{
    // constructor
    Stats(const Params& params_);

    // timing stats
    float       global_time;
    KernelStats map;
    KernelStats select;
    KernelStats sort;
    KernelStats locate;
    KernelStats score;
    KernelStats opposite_score;
    KernelStats backtrack;
    KernelStats backtrack_opposite;
    KernelStats finalize;
    KernelStats alignments_DtoH;
    KernelStats read_HtoD;
    KernelStats read_io;
    KernelStats io;
    KernelStats scoring_pipe;

    // mapping stats
    uint32              n_reads;
    uint32              n_mapped;
    uint32              n_unique;
    uint32              n_multiple;
    uint32              n_ambiguous;
    uint32              n_nonambiguous;
    std::vector<uint32> mapped;
    std::vector<uint32> f_mapped;
    std::vector<uint32> r_mapped;
    uint32              mapped2[64][64];

    // mapping quality stats
    uint64 mapq_bins[64];

    // extensive (seeding) stats
    volatile bool stats_ready;
    uint64 hits_total;
    uint64 hits_ranges;
    uint32 hits_max;
    uint32 hits_max_range;
    uint64 hits_top_total;
    uint32 hits_top_max;
    uint64 hits_bins[28];
    uint64 hits_top_bins[28];
    uint32 hits_stats;

    Params params;
};

void generate_report(Stats& stats, const char* report);

} // namespace cuda
} // namespace bowtie2
} // namespace nvbio
