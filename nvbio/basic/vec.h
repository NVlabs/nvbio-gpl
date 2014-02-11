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
#include <cmath>
#include <limits>

namespace nvbio {

///
/// A generic small vector class with the dimension set at compile-time
///
template <uint32 DIM, typename T>
struct Vec
{
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    Vec() {}

    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    Vec(const T* v);

    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    explicit Vec(const T v);

    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    Vec<DIM,T>& operator= (const Vec<DIM,T>& op2);

    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    const T& operator[] (const uint32 i) const { return data[i]; }

    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
          T& operator[] (const uint32 i)       { return data[i]; }

    T data[DIM];
};

template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T> operator+ (const Vec<DIM,T>& op1, const Vec<DIM,T>& op2);

template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T>& operator+= (Vec<DIM,T>& op1, const Vec<DIM,T>& op2);

template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T> operator- (const Vec<DIM,T>& op1, const Vec<DIM,T>& op2);

template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T>& operator-= (Vec<DIM,T>& op1, const Vec<DIM,T>& op2);

template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T> operator* (const Vec<DIM,T>& op1, const Vec<DIM,T>& op2);

template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T>& operator*= (Vec<DIM,T>& op1, const Vec<DIM,T>& op2);

template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T> operator/ (const Vec<DIM,T>& op1, const Vec<DIM,T>& op2);

template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T>& operator/= (Vec<DIM,T>& op1, const Vec<DIM,T>& op2);

template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T> min(const Vec<DIM,T>& op1, const Vec<DIM,T>& op2);

template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T> max(const Vec<DIM,T>& op1, const Vec<DIM,T>& op2);

template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
bool any(const Vec<DIM,T>& op);

} // namespace nvbio

#include <nvbio/basic/vec_inl.h>
