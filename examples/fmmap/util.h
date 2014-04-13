/*
 * nvbio
 * Copyright (C) 2011-2014, NVIDIA Corporation
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

#include <nvbio/basic/numbers.h>

// return 1 or 0 depending on whether a number is >= than a given threshold
struct above_threshold
{
    typedef int16  argument_type;
    typedef uint32 result_type;

    // constructor
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    above_threshold(const int16 _t) : t(_t) {}

    // functor operator
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 operator() (const int16 s) { return s >= t ? 1u : 0u; }

    const int16 t;
};

// update the best scores vector
//
__global__
void update_scores_kernel(
    const uint32  n,
    const uint32* reads,
    const int16*  scores,
        int16*    best)
{
    const uint32 i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= n) return;

    const uint32 read_id = reads[i];
    const int16  score   = scores[i];

    best[ read_id ] = nvbio::max( best[ read_id ], score );
}

// update the best scores vector
//
void update_scores(
    const uint32  n,
    const uint32* reads,
    const int16*  scores,
        int16*    best)
{
    const uint32 block_dim = 128;
    const uint32 n_blocks = util::divide_ri( n, block_dim );

    update_scores_kernel<<<n_blocks,block_dim>>>( n, reads, scores, best );
}
