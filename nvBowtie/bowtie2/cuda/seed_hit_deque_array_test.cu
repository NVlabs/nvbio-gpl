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

//#define NVBIO_CUDA_DEBUG
//#define NVBIO_CUDA_ASSERTS

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <nvbio/basic/timer.h>
#include <nvbio/basic/console.h>
#include <nvBowtie/bowtie2/cuda/seed_hit_deque_array.h>


namespace nvbio {
namespace bowtie2 {
namespace cuda {

namespace { // anonymous namespace

__global__
void setup_deques_kernel(SeedHitDequeArrayDeviceView seed_hit_deques, const uint32 n_reads, uint32* error)
{
    if (threadIdx.x >= n_reads)
        return;

    typedef SeedHitDequeArrayDeviceView::reference      hit_deque_reference;
    typedef SeedHitDequeArrayDeviceView::hit_deque_type hit_deque_type;

    // fetch the deque bound to this read
    hit_deque_reference hit_deque = seed_hit_deques[ threadIdx.x ];

    // alloc storage for 2 entries in this read's deque
    hit_deque.alloc( 2u );

    // push first hit
    hit_deque.push( SeedHit( STANDARD, FORWARD, threadIdx.x * 2u + 1u, make_uint2( 0, 100 ) ) );

    // push second hit
    hit_deque.push( SeedHit( STANDARD, FORWARD, threadIdx.x * 2u + 0u, make_uint2( 0, 10 ) ) );
}

__global__
void check_deques_kernel(SeedHitDequeArrayDeviceView seed_hit_deques, const uint32 n_reads, uint32* error)
{
    if (threadIdx.x >= n_reads)
        return;

    typedef SeedHitDequeArrayDeviceView::reference      hit_deque_reference;
    typedef SeedHitDequeArrayDeviceView::hit_deque_type hit_deque_type;

    // fetch the deque bound to this read
    hit_deque_reference hit_deque = seed_hit_deques[ threadIdx.x ];

    SeedHit hit;

    // pop first hit
    hit = hit_deque.top(); hit_deque.pop();
    if (hit.get_posinread() != threadIdx.x * 2u + 0u)
        *error = 1;

    // pop second hit
    hit = hit_deque.top(); hit_deque.pop();
    if (hit.get_posinread() != threadIdx.x * 2u + 1u)
        *error = 2;
}

} // anonymous namespace

void test_seed_hit_deques()
{
    log_info(stderr, "test seed_hit_deques... started\n");
    SeedHitDequeArray seed_hit_deques;

    const uint32 n_hits          = 100;
    const uint32 n_reads         = 50;

    const uint64 bytes = seed_hit_deques.resize( n_reads, n_hits );
    log_info(stderr, "  allocated %llu bytes\n", bytes);

    thrust::device_vector<uint32> error(1,0);

    setup_deques_kernel<<<1,128>>>( seed_hit_deques.device_view(), n_reads, device_view( error ) );
    check_deques_kernel<<<1,128>>>( seed_hit_deques.device_view(), n_reads, device_view( error ) );
    cudaThreadSynchronize();

    const uint32 error_code = error[0];
    if (error_code)
        log_error( stderr, "test_read_hits_index failed! (error code: %u)\n", error_code );

    log_info(stderr, "test seed_hit_deques... done\n");
}

} // namespace cuda
} // namespace bowtie2
} // namespace nvbio
