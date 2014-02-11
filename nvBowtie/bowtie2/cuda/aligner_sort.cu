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

#include <nvBowtie/bowtie2/cuda/aligner.h>
#include <nvbio/basic/cuda/sort.h>

namespace nvbio {
namespace bowtie2 {
namespace cuda {

// return a pointer to an "index" into the given keys sorted by their hi bits
//
uint32* Aligner::sort_hi_bits(
    const uint32    count,
    const uint32*   keys)
{
    thrust::transform(
        thrust::device_ptr<const uint32>( keys ),
        thrust::device_ptr<const uint32>( keys ) + count,
        sorting_queue_dvec.begin(),
        hi_bits_functor<uint16,uint32>() );

    thrust::copy(
        thrust::make_counting_iterator(0u),
        thrust::make_counting_iterator(0u) + count,
        idx_queue_dvec.begin() );

    // Create ping-pong storage wrapper
    nvbio::cuda::SortBuffers<uint16*,uint32*> double_buffer;
    double_buffer.keys[0]   = thrust::raw_pointer_cast( &sorting_queue_dvec.front() );
    double_buffer.keys[1]   = thrust::raw_pointer_cast( &sorting_queue_dvec.front() + BATCH_SIZE );
    double_buffer.values[0] = idx_queue_dptr;
    double_buffer.values[1] = idx_queue_dptr + BATCH_SIZE;

    sort_enactor.sort( count, double_buffer );

    return double_buffer.values[double_buffer.selector];
}

// sort a set of keys in place
//
void Aligner::sort_inplace(
    const uint32    count,
    uint32*         keys)
{
    // create the ping-pong storage wrapper
    nvbio::cuda::SortBuffers<uint32*> double_buffer;
    double_buffer.keys[0]   = keys;
    double_buffer.keys[1]   = (uint32*)thrust::raw_pointer_cast( &sorting_queue_dvec.front() );

    // enact the sort
    sort_enactor.sort( count, double_buffer );

    // copy the sorted data back in place if it ended up in the temporary buffer
    if (double_buffer.selector)
        cudaMemcpy( keys, double_buffer.keys[1], count * sizeof(uint32), cudaMemcpyDeviceToDevice );
}

} // namespace cuda
} // namespace bowtie2
} // namespace nvbio