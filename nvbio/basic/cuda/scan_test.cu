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

// scan_test.cu
//

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <nvbio/basic/timer.h>
#include <nvbio/basic/console.h>
#include <nvbio/basic/cuda/arch.h>
#include <nvbio/basic/cuda/scan.h>
#include <thrust/device_vector.h>

using namespace nvbio;

namespace nvbio {
namespace cuda {

template <uint32 BLOCKDIM, uint32 VECDIM, uint32 N_TESTS>
__global__ void all_test_kernel(uint32* dst)
{
    __shared__ volatile uint8 sm[ BLOCKDIM >> cuda::Arch::LOG_WARP_SIZE ];

    uint32 r = 0;
    for (uint32 i = 0; i < N_TESTS; ++i)
    {
        const bool p = threadIdx.x > 2*i;
        r += all<VECDIM>( p, sm );
    }
    dst[ threadIdx.x + blockDim.x*blockIdx.x ] = r;
}
template <uint32 BLOCKDIM, uint32 VECDIM, uint32 N_TESTS>
__global__ void any_test_kernel(uint32* dst)
{
    __shared__ volatile uint8 sm[ BLOCKDIM >> cuda::Arch::LOG_WARP_SIZE ];

    uint32 r = 0;
    for (uint32 i = 0; i < N_TESTS; ++i)
    {
        const bool p = threadIdx.x > 2*i;
        r += any<VECDIM>( p, sm );
    }
    dst[ threadIdx.x + blockDim.x*blockIdx.x ] = r;
}

template <uint32 N_THREADS, uint32 BLOCKDIM, uint32 VECDIM, uint32 N_TESTS>
void all_test(uint32* dst)
{
    const uint32 N_RUNS = 16;
    Timer timer;
    timer.start();

    for (uint32 i = 0; i < N_RUNS; ++i)
        all_test_kernel<BLOCKDIM,VECDIM,N_TESTS><<<BLOCKDIM, N_THREADS / BLOCKDIM>>>( dst );

    cudaThreadSynchronize();

    timer.stop();

    fprintf(stderr, "  all<%u> throughput: %.3f G vectors/s, %.3f G threads/s\n", VECDIM,
        (1.0e-9f * float(N_THREADS/VECDIM)*float(N_TESTS))*(float(N_RUNS)/timer.seconds()),
        (1.0e-9f * float(N_THREADS)*float(N_TESTS))*(float(N_RUNS)/timer.seconds()));
}
template <uint32 N_THREADS, uint32 BLOCKDIM, uint32 VECDIM, uint32 N_TESTS>
void any_test(uint32* dst)
{
    const uint32 N_RUNS = 16;
    Timer timer;
    timer.start();

    for (uint32 i = 0; i < N_RUNS; ++i)
        any_test_kernel<BLOCKDIM,VECDIM,N_TESTS><<<BLOCKDIM, N_THREADS / BLOCKDIM>>>( dst );

    cudaThreadSynchronize();

    timer.stop();

    fprintf(stderr, "  any<%u> throughput: %.3f G vectors/s, %.3f G threads/s\n", VECDIM,
        (1.0e-9f * float(N_THREADS/VECDIM)*float(N_TESTS))*(float(N_RUNS)/timer.seconds()),
        (1.0e-9f * float(N_THREADS)*float(N_TESTS))*(float(N_RUNS)/timer.seconds()));
}

void scan_test()
{
    const uint32 BLOCKDIM  = 128;
    const uint32 N_THREADS = BLOCKDIM * 1024;
    NVBIO_VAR_UNUSED const uint32 N_TESTS   = 128;

    thrust::device_vector<uint32> dst( N_THREADS );
    uint32* dst_dptr = thrust::raw_pointer_cast(&dst.front());

    fprintf(stderr, "all test... started\n");
    all_test<N_THREADS,BLOCKDIM,2,N_TESTS>( dst_dptr );
    all_test<N_THREADS,BLOCKDIM,4,N_TESTS>( dst_dptr );
    all_test<N_THREADS,BLOCKDIM,8,N_TESTS>( dst_dptr );
    all_test<N_THREADS,BLOCKDIM,16,N_TESTS>( dst_dptr );
    all_test<N_THREADS,BLOCKDIM,32,N_TESTS>( dst_dptr );
    all_test<N_THREADS,BLOCKDIM,64,N_TESTS>( dst_dptr );
    all_test<N_THREADS,BLOCKDIM,128,N_TESTS>( dst_dptr );
    fprintf(stderr, "all test... done\n");

    fprintf(stderr, "any test... started\n");
    any_test<N_THREADS,BLOCKDIM,2,N_TESTS>( dst_dptr );
    any_test<N_THREADS,BLOCKDIM,4,N_TESTS>( dst_dptr );
    any_test<N_THREADS,BLOCKDIM,8,N_TESTS>( dst_dptr );
    any_test<N_THREADS,BLOCKDIM,16,N_TESTS>( dst_dptr );
    any_test<N_THREADS,BLOCKDIM,32,N_TESTS>( dst_dptr );
    any_test<N_THREADS,BLOCKDIM,64,N_TESTS>( dst_dptr );
    any_test<N_THREADS,BLOCKDIM,128,N_TESTS>( dst_dptr );
    fprintf(stderr, "any test... done\n");
}

} // namespace cuda
} // namespace nvbio
