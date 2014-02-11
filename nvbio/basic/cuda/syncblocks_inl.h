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

#include <nvbio/basic/types.h>
#include <nvbio/basic/numbers.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>

namespace nvbio {
namespace cuda {

// constructor
//
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
syncblocks::syncblocks(int32* counter) : m_counter( counter ) {}

// implements an inter-CTA synchronization primitive
//
NVBIO_FORCEINLINE NVBIO_DEVICE
bool syncblocks::enact(const uint32 max_iter)
{
    __threadfence();
    __syncthreads();

    // each block does an atomicAdd on an integer, waiting for all CTAs to be
    // counted. When this happens, a global semaphore is released.
    // The CTA counter is always increased across multiple calls to syncblocks,
    // so that its value will say which syncblocks each CTA is participating
    // too.
    // Similarly, the semaphore is always increasing. As soon as the semaphore
    // is higher than the syncblocks a CTA has just entered, the semaphore is
    // considered 'released' for that syncblocks.
    __shared__ volatile bool ret;
    if (threadIdx.x == 0)
    {
        const uint32 grid_size = gridDim.x * gridDim.y * gridDim.z;

        int32* semaphore = (m_counter + 1);

        // add 1 atomically to the shared counter
        const uint32 slot = atomicAdd( m_counter, 1 );

        // compute which syncblocks we are particpating too based on the result we got from the atomicAdd
        const uint32 iteration = slot / grid_size;

        const bool is_last_block = (slot - iteration*grid_size) == (grid_size-1);
        if (is_last_block)
        {
            // release the semaphore
            atomicAdd( semaphore, 1 );
        }

        // wait for the semaphore write to become public
        __threadfence();

        // spin until the semaphore is released
        for (uint32 iter = 0; iter < max_iter && *(volatile int32*)semaphore <= iteration; ++iter) {}

        ret = (*(volatile int32*)semaphore > iteration);
    }

    // synchronize all threads in this CTA
    __syncthreads();
    return ret;
}

// constructor
//
inline syncblocks_storage::syncblocks_storage()
{
    // alloc a counter and a semaphore
    m_counter.resize( 2, 0 );
}

// return a syncblocks object
//
inline syncblocks syncblocks_storage::get()
{
    return syncblocks( thrust::raw_pointer_cast( &m_counter.front() ) );
}

// clear the syncblocks, useful if one wants to reuse it
// across differently sized kernel launches.
//
inline void syncblocks_storage::clear()
{
    thrust::fill( m_counter.begin(), m_counter.end(), 0 );
}

} // namespace cuda
} // namespace nvbio
