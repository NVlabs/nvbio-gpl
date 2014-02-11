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

namespace nvbio {

#if defined(NVBIO_ENABLE_PROFILING)
#if defined(__CUDA_ARCH__)
#define NVBIO_INC_UTILIZATION(v1,v2) \
do { \
    atomicAdd( (uint32*)&v1, 1 ); \
    const uint32 mask = __ballot(true); \
    if (__popc( mask >> warp_tid() ) == 1) \
        v2 += 32; \
} while (0)
#define NVBIO_STATS_SET(x,v)          x = v
#define NVBIO_STATS_ADD(x,v)          x += v
#else
#define NVBIO_INC_UTILIZATION(v1,v2)
#define NVBIO_STATS_SET(x,v)          x = v
#define NVBIO_STATS_ADD(x,v)          x += v
#define NVBIO_STATS(stmnt)            stmnt
#endif
#else
#define NVBIO_INC_UTILIZATION(v1,v2)
#define NVBIO_STATS_SET(x,v)
#define NVBIO_STATS_ADD(x,v)
#define NVBIO_STATS(stmnt)
#endif

} // namespace nvbio
