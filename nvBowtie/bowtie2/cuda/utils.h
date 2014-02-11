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

#include <nvbio/basic/timer.h>
#include <nvbio/basic/cuda/timer.h>
#include <cuda_runtime.h>

namespace nvbio {
namespace bowtie2 {
namespace cuda {

inline void optional_device_synchronize()
{
#if DO_OPTIONAL_SYNCHRONIZE
    cudaDeviceSynchronize();
#endif
}

#if DO_DEVICE_TIMING
typedef nvbio::cuda::Timer DeviceTimer;
#else
typedef nvbio::FakeTimer   DeviceTimer;
#endif

// convert a range from a [begin,last] representation to a [begin,end) one
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
uint2 inclusive_to_exclusive(const uint2 range) { return make_uint2( range.x, range.y + 1u ); }

#ifdef __CUDACC__

// alloc some slots from a counter using a warp-synchronous reduction
NVBIO_FORCEINLINE NVBIO_DEVICE uint32 alloc(uint32* counter, volatile uint32 *warp_broadcast)
{
#if USE_WARP_SYNCHRONOUS_QUEUES
    const uint32 mask = __ballot( true );
    const uint32 pop_scan  = __popc( mask << (32u - warp_tid()) );
    const uint32 pop_count = __popc( mask );
    if (pop_scan == 0)
        *warp_broadcast = atomicAdd( counter, pop_count );

    return *warp_broadcast + pop_scan;
#else
    return atomicAdd( counter, 1u );
#endif
}

#endif // __CUDACC__

} // namespace cuda
} // namespace bowtie2
} // namespace nvbio
