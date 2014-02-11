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

// alloc_test.cu
//

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <nvbio/basic/timer.h>
#include <nvbio/basic/console.h>
#include <nvbio/basic/cuda/arch.h>

namespace nvbio {

int alloc_test()
{
    log_info( stderr, "alloc test... started\n" );
    const uint32 N_TESTS = 32;

    for (size_t size = 1024*1024; size <= size_t(1u << 30); size *= 4)
    {
        Timer timer;

        float cuda_malloc_time = 0.0f;
        float cuda_free_time   = 0.0f;

        float malloc_time = 0.0f;
        float free_time   = 0.0f;

        for (uint32 i = 0; i < N_TESTS; ++i)
        {
            void* ptr;

            // cuda
            timer.start();
            cudaMalloc( &ptr, size );
            timer.stop();

            cuda_malloc_time += timer.seconds();

            timer.start();
            cudaFree( ptr );
            timer.stop();

            cuda_free_time += timer.seconds();

            // cpu
            timer.start();
            ptr = malloc( size );
            timer.stop();

            malloc_time += timer.seconds();

            timer.start();
            free( ptr );
            timer.stop();

            free_time += timer.seconds();
        }

        const float GB = float(1024*1024*1024);

        cuda_malloc_time /= N_TESTS;
        cuda_free_time /= N_TESTS;
        malloc_time /= N_TESTS;
        free_time /= N_TESTS;

        log_info( stderr, "  %u MB:\n", size/(1024*1024) );
        log_info( stderr, "    cuda malloc : %.2f ms, %.3f GB/s\n", cuda_malloc_time*1000.0f, (float(size)/(cuda_malloc_time)) / GB );
        log_info( stderr, "    cuda free   : %.2f ms, %.3f GB/s\n", cuda_free_time*1000.0f,   (float(size)/(cuda_free_time)) / GB );
        log_info( stderr, "    malloc      : %.2f ms, %.3f GB/s\n", malloc_time*1000.0f,      (float(size)/(malloc_time)) / GB );
        log_info( stderr, "    free        : %.2f ms, %.3f GB/s\n", free_time*1000.0f,        (float(size)/(free_time)) / GB );
    }
    log_info( stderr, "alloc test... done\n" );
    return 0;
}

} // namespace nvbio
