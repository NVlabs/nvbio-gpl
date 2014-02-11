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
#include <nvBowtie/bowtie2/cuda/seed_hit_deque_array.h>
#include <nvbio/basic/types.h>
#include <crc/crc.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

namespace nvbio {
namespace bowtie2 {
namespace cuda {

struct SeedHit;

// compute a checksum of a device sequence
//
template <typename Iterator>
uint64 device_checksum(
    Iterator    begin,
    Iterator    end)
{
    typedef typename thrust::iterator_traits<Iterator>::value_type value_type;

    const uint32 n = uint32( end - begin );

    // copy values to a temporary vector
    thrust::device_vector<value_type> debug_copy_dvec( n );
    thrust::copy(
        begin,
        end,
        debug_copy_dvec.begin() );

    // sort temporary vector
    thrust::sort( debug_copy_dvec.begin(), debug_copy_dvec.end() );

    // copy to the host
    thrust::host_vector<value_type> debug_copy_hvec( debug_copy_dvec );

    // compute a crc
    const char* ptr = (const char*)thrust::raw_pointer_cast( &debug_copy_hvec.front() );
    return crcCalc( ptr, sizeof(value_type)*n );
}

// compute a checksum for the returned SA ranges
//
void hits_checksum(
    const uint32                n_reads,
          SeedHitDequeArray&    hit_deques,
          uint64&               crc,
          uint64&               sum);

} // namespace cuda
} // namespace bowtie2
} // namespace nvbio
