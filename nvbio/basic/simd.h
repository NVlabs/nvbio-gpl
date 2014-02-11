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
#ifdef __CUDACC__
#include <nvbio/basic/cuda/simd_functions.h>
#endif
#include <cmath>
#include <limits>

namespace nvbio {

///
/// A 4-way uint8 SIMD type
///
struct simd4u8
{
    struct base_rep_tag {};

    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    simd4u8() {}

    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    explicit simd4u8(const uint32 op, const base_rep_tag) { m = op; }

    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    explicit simd4u8(const uint4 v);

    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    explicit simd4u8(const uint8 v);

    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    simd4u8(const uint8 v1, const uint8 v2, const uint8 v3, const uint8 v4);

    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    simd4u8& operator= (const uint4 v);

    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    simd4u8& operator= (const uchar4 v);

    uint32 m;
};

NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
bool any(const simd4u8 op) { return op.m != 0; }

NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8 operator~(const simd4u8 op) { return simd4u8( ~op.m ); }

NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8 operator== (const simd4u8 op1, const simd4u8 op2);

NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8 operator!= (const simd4u8 op1, const simd4u8 op2);

NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8 operator>= (const simd4u8 op1, const simd4u8 op2);

NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8 operator> (const simd4u8 op1, const simd4u8 op2);

NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8 operator<= (const simd4u8 op1, const simd4u8 op2);

NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8 operator< (const simd4u8 op1, const simd4u8 op2);

NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8 operator+ (const simd4u8 op1, const simd4u8 op2);

NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8& operator+= (simd4u8& op1, const simd4u8 op2);

NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8 operator- (const simd4u8 op1, const simd4u8 op2);

NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8& operator-= (simd4u8& op1, const simd4u8 op2);

NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8 max(const simd4u8 op1, const simd4u8 op2);

NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8 min(const simd4u8 op1, const simd4u8 op2);

NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8 and_op(const simd4u8 op1, const simd4u8 op2);

NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8 or_op(const simd4u8 op1, const simd4u8 op2);

NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8 ternary_op(const simd4u8 mask, const simd4u8 op1, const simd4u8 op2);

template <uint32 I>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
uint8 get(const simd4u8 op);

} // namespace nvbio

#include <nvbio/basic/simd_inl.h>
