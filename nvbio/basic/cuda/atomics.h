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

//#define USE_WARP_VOTE


namespace nvbio {

NVBIO_FORCEINLINE NVBIO_DEVICE void warp_incff(uint32* count)
{
#ifdef USE_WARP_VOTE
    *(volatile uint32*)count += __popc( __ballot( true ) );
#else
    atomicInc( count, uint32(-1) );
#endif
}
NVBIO_FORCEINLINE NVBIO_DEVICE uint32 warp_inc(uint32* count)
{
#ifdef USE_WARP_VOTE
    const volatile uint32 val = *(volatile uint32*)count;
    const volatile uint32 mask = __ballot( true );
    const uint32   warp_count  = __popc( mask );
    const uint32   warp_scan   = __popc( mask >> warp_tid() ) - 1u;
    if (warp_scan == 0)
        *(volatile uint32*)count = val + warp_count;
    return val + warp_scan;
#else
    return atomicInc( count, uint32(-1) );
#endif
}
NVBIO_FORCEINLINE NVBIO_DEVICE void warp_decff(uint32* count)
{
#ifdef USE_WARP_VOTE
    *(volatile uint32*)count -= __popc( __ballot( true ) );
#else
    atomicDec( count, uint32(-1) );
#endif
}
NVBIO_FORCEINLINE NVBIO_DEVICE uint32 warp_dec(uint32* count)
{
#ifdef USE_WARP_VOTE
    const volatile uint32 val  = *(volatile uint32*)count;
    const volatile uint32 mask = __ballot( true );
    const uint32   warp_count  = __popc( mask );
    const uint32   warp_scan   = __popc( mask >> warp_tid() ) - 1u;
    if (warp_scan == 0)
        *(volatile uint32*)count = val - warp_count;
    return val - warp_scan;
#else
    return atomicDec( count, uint32(-1) );
#endif
}

} // namespace nvbio
