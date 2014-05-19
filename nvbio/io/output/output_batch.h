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
#include <nvbio/io/output/output_stats.h>
#include <nvbio/io/output/output_file.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <nvbio/basic/vector_array.h>
#include <nvbio/io/fmi.h>
#include <nvbio/io/sequence/sequence.h>

#include <stdio.h>

namespace nvbio {
namespace io {

// a batch of alignment results on the CPU
struct CPUOutputBatch
{
    uint32 count;

    thrust::host_vector<io::AlignmentResult> best_alignments;

    // we have two cigar and MDS arrays, one for each mate
    HostCigarArray                               cigar[2];
    HostMdsArray                                 mds[2];

    // pointer to the host-side read data for each mate
    const io::SequenceDataHost<DNA_N>*           read_data[2];

    CPUOutputBatch()
        : count(0)
    { }

    // extract alignment data for a given mate
    AlignmentData get_mate(uint32 read_id, AlignmentMate mate, AlignmentMate cigar_mate);
    // extract alignment data for the anchor mate
    AlignmentData get_anchor(uint32 read_id);
    // extract alignment data for the opposite mate
    AlignmentData get_opposite_mate(uint32 read_id);
};

// base class for representing a batch of alignment results on the device
struct GPUOutputBatch
{
public:
    uint32                                     count;

    thrust::device_vector<io::BestAlignments>& best_data_dvec;
    DeviceCigarArray                           cigar;
    nvbio::DeviceVectorArray<uint8>&           mds;

    const io::SequenceDataDevice<DNA_N>&       read_data;

    GPUOutputBatch(uint32                                         _count,
                   thrust::device_vector<io::BestAlignments>&     _best_data_dvec,
                   DeviceCigarArray                               _cigar,
                   nvbio::DeviceVectorArray<uint8>&               _mds,
                   const io::SequenceDataDevice<DNA_N>&           _read_data)
            : count(_count),
              best_data_dvec(_best_data_dvec),
              cigar(_cigar),
              mds(_mds),
              read_data(_read_data)
    { }

    // copy best score data into host memory
    void readback_scores(thrust::host_vector<io::AlignmentResult>& output,
                         const AlignmentMate mate,
                         const AlignmentScore score) const;
    // copy cigars into host memory
    void readback_cigars(HostCigarArray& host_cigars) const;
    // copy md strings into host memory
    void readback_mds(nvbio::HostVectorArray<uint8>& host_mds) const;
};

} // namespace io
} // namespace nvbio
