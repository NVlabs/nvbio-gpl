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

#include <nvBowtie/bowtie2/cuda/checksums.h>
#include <nvBowtie/bowtie2/cuda/mapping.h>
#include <nvBowtie/bowtie2/cuda/seed_hit.h>
#include <crc/crc.h>
#include <thrust/scan.h>

namespace nvbio {
namespace bowtie2 {
namespace cuda {

// compute a checksum for the returned SA ranges
//
void hits_checksum(
    const uint32                n_reads,
          SeedHitDequeArray&    hit_deques,
          uint64&               crc,
          uint64&               sum)
{
    thrust::device_vector<uint32> hits_count_scan_dvec( n_reads );

    // run a scan on the number of SA ranges from each read
    thrust::inclusive_scan( hit_deques.counts().begin(), hit_deques.counts().begin() + n_reads, hits_count_scan_dvec.begin() );

    // compute how many ranges we have
    const uint32 n_hit_ranges = hits_count_scan_dvec[ n_reads-1 ];

    // gather all the range sizes, sorted by read, in a compacted array
    thrust::device_vector<uint64> hits_range_scan_dvec( n_hit_ranges );

    SeedHitDequeArrayDeviceView hits = hit_deques.device_view();

    gather_ranges(
        n_hit_ranges,
        n_reads,
        hits,
        thrust::raw_pointer_cast( &hits_count_scan_dvec.front() ),
        thrust::raw_pointer_cast( &hits_range_scan_dvec.front() ) );

    // compute the crc
    crc = device_checksum(
        hits_range_scan_dvec.begin(),
        hits_range_scan_dvec.begin() + n_hit_ranges );

    // scan the ranges
    thrust::inclusive_scan( hits_range_scan_dvec.begin(), hits_range_scan_dvec.begin() + n_hit_ranges, hits_range_scan_dvec.begin() );

    // fetch the total
    sum = hits_range_scan_dvec[ n_hit_ranges - 1 ];
}

} // namespace cuda
} // namespace bowtie2
} // namespace nvbio
