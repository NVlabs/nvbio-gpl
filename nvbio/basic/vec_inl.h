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

namespace nvbio {

template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T>::Vec(const T* v)
{
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        data[d] = v[d];
}

template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T>::Vec(const T v)
{
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        data[d] = v;
}

template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T>& Vec<DIM,T>::operator= (const Vec<DIM,T>& op2)
{
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        data[d] = op2.data[d];

    return *this;
}

template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T> operator+ (const Vec<DIM,T>& op1, const Vec<DIM,T>& op2)
{
    Vec<DIM,T> r;
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        r.data[d] = op1.data[d] + op2.data[d];
    return r;
}
template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T>& operator+= (Vec<DIM,T>& op1, const Vec<DIM,T>& op2)
{
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        op1.data[d] = op1.data[d] + op2.data[d];
    return op1;
}
template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T> operator- (const Vec<DIM,T>& op1, const Vec<DIM,T>& op2)
{
    Vec<DIM,T> r;
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        r.data[d] = op1.data[d] - op2.data[d];
    return r;
}
template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T>& operator-= (Vec<DIM,T>& op1, const Vec<DIM,T>& op2)
{
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        op1.data[d] = op1.data[d] - op2.data[d];
    return op1;
}
template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T> operator* (const Vec<DIM,T>& op1, const Vec<DIM,T>& op2)
{
    Vec<DIM,T> r;
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        r.data[d] = op1.data[d] - op2.data[d];
    return r;
}
template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T>& operator*= (Vec<DIM,T>& op1, const Vec<DIM,T>& op2)
{
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        op1.data[d] = op1.data[d] * op2.data[d];
    return op1;
}
template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T> operator/ (const Vec<DIM,T>& op1, const Vec<DIM,T>& op2)
{
    Vec<DIM,T> r;
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        r.data[d] = op1.data[d] / op2.data[d];
    return r;
}
template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T>& operator/= (Vec<DIM,T>& op1, const Vec<DIM,T>& op2)
{
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        op1.data[d] = op1.data[d] / op2.data[d];
    return op1;
}
template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T> min(const Vec<DIM,T>& op1, const Vec<DIM,T>& op2)
{
    Vec<DIM,T> r;
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        r.data[d] = nvbio::min( op1.data[d], op2.data[d] );
    return r;
}
template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
Vec<DIM,T> max(const Vec<DIM,T>& op1, const Vec<DIM,T>& op2)
{
    Vec<DIM,T> r;
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        r.data[d] = nvbio::max( op1.data[d], op2.data[d] );
    return r;
}
template <uint32 DIM, typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
bool any(const Vec<DIM,T>& op)
{
    bool r = false;
    #pragma unroll
    for (uint32 d = 0; d < DIM; ++d)
        r = r && (op.data[d] != 0);
    return r;
}

} // namespace nvbio
