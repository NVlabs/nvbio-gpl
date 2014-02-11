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

namespace nvbio {

NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8::simd4u8(const uint4 v)
{
    m  =  v.x;
    m |= (v.y << 8);
    m |= (v.z << 16);
    m |= (v.w << 24);
}

NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8::simd4u8(const uint8 v)
{
    m  =  v;
    m |= (v << 8);
    m |= (v << 16);
    m |= (v << 24);
}

NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8::simd4u8(const uint8 v1, const uint8 v2, const uint8 v3, const uint8 v4)
{
    m  =  v1;
    m |= (v2 << 8);
    m |= (v3 << 16);
    m |= (v4 << 24);
}

NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8& simd4u8::operator= (const uint4 v)
{
    m  = v.x;
    m |= (v.y << 8);
    m |= (v.z << 16);
    m |= (v.w << 24);
    return *this;
}
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8& simd4u8::operator= (const uchar4 v)
{
    m  = v.x;
    m |= (v.y << 8);
    m |= (v.z << 16);
    m |= (v.w << 24);
    return *this;
}

NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8 operator== (const simd4u8 op1, const simd4u8 op2)
{
#if defined(NVBIO_DEVICE_COMPILATION)
    return simd4u8( vcmpeq4( op1.m ,op2.m ), simd4u8::base_rep_tag() );
#else
    return simd4u8(
        get<0>(op1) == get<0>(op2) ? 0xFFu : 0u,
        get<1>(op1) == get<1>(op2) ? 0xFFu : 0u,
        get<2>(op1) == get<2>(op2) ? 0xFFu : 0u,
        get<3>(op1) == get<3>(op2) ? 0xFFu : 0u );
#endif
}
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8 operator!= (const simd4u8 op1, const simd4u8 op2)
{
#if defined(NVBIO_DEVICE_COMPILATION)
    return simd4u8( vcmpne4( op1.m ,op2.m ), simd4u8::base_rep_tag() );
#else
    return simd4u8(
        get<0>(op1) != get<0>(op2) ? 0xFFu : 0u,
        get<1>(op1) != get<1>(op2) ? 0xFFu : 0u,
        get<2>(op1) != get<2>(op2) ? 0xFFu : 0u,
        get<3>(op1) != get<3>(op2) ? 0xFFu : 0u );
#endif
}
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8 operator>= (const simd4u8 op1, const simd4u8 op2)
{
#if defined(NVBIO_DEVICE_COMPILATION)
    return simd4u8( vcmpgeu4( op1.m ,op2.m ), simd4u8::base_rep_tag() );
#else
    return simd4u8(
        get<0>(op1) <= get<0>(op2) ? 0xFFu : 0u,
        get<1>(op1) <= get<1>(op2) ? 0xFFu : 0u,
        get<2>(op1) <= get<2>(op2) ? 0xFFu : 0u,
        get<3>(op1) <= get<3>(op2) ? 0xFFu : 0u );
#endif
}
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8 operator> (const simd4u8 op1, const simd4u8 op2)
{
#if defined(NVBIO_DEVICE_COMPILATION)
    return simd4u8( vcmpgtu4( op1.m ,op2.m ), simd4u8::base_rep_tag() );
#else
    return simd4u8(
        get<0>(op1) > get<0>(op2) ? 0xFFu : 0u,
        get<1>(op1) > get<1>(op2) ? 0xFFu : 0u,
        get<2>(op1) > get<2>(op2) ? 0xFFu : 0u,
        get<3>(op1) > get<3>(op2) ? 0xFFu : 0u );
#endif
}
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8 operator<= (const simd4u8 op1, const simd4u8 op2)
{
#if defined(NVBIO_DEVICE_COMPILATION)
    return simd4u8( vcmpleu4( op1.m ,op2.m ), simd4u8::base_rep_tag() );
#else
    return simd4u8(
        get<0>(op1) <= get<0>(op2) ? 0xFFu : 0u,
        get<1>(op1) <= get<1>(op2) ? 0xFFu : 0u,
        get<2>(op1) <= get<2>(op2) ? 0xFFu : 0u,
        get<3>(op1) <= get<3>(op2) ? 0xFFu : 0u );
#endif
}
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8 operator< (const simd4u8 op1, const simd4u8 op2)
{
#if defined(NVBIO_DEVICE_COMPILATION)
    return simd4u8( vcmpltu4( op1.m ,op2.m ), simd4u8::base_rep_tag() );
#else
    return simd4u8(
        get<0>(op1) < get<0>(op2) ? 0xFFu : 0u,
        get<1>(op1) < get<1>(op2) ? 0xFFu : 0u,
        get<2>(op1) < get<2>(op2) ? 0xFFu : 0u,
        get<3>(op1) < get<3>(op2) ? 0xFFu : 0u );
#endif
}
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8 operator+ (const simd4u8 op1, const simd4u8 op2)
{
#if defined(NVBIO_DEVICE_COMPILATION)
    return simd4u8( vadd4( op1.m, op2.m ), simd4u8::base_rep_tag() ); // per-byte (un)signed addition, with wrap-around: a + b
#else
    return simd4u8(
        get<0>(op1) + get<0>(op2),
        get<1>(op1) + get<1>(op2),
        get<2>(op1) + get<2>(op2),
        get<3>(op1) + get<3>(op2) );
#endif
}
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8& operator+= (simd4u8& op1, const simd4u8 op2)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
    op1.m = vadd4( op1.m, op2.m ); // per-byte (un)signed addition, with wrap-around: a + b
#else
    op1 = simd4u8(
        get<0>(op1) + get<0>(op2),
        get<1>(op1) + get<1>(op2),
        get<2>(op1) + get<2>(op2),
        get<3>(op1) + get<3>(op2) );
#endif
    return op1;
}
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8 operator- (const simd4u8 op1, const simd4u8 op2)
{
#if defined(NVBIO_DEVICE_COMPILATION)
    return simd4u8( vsub4( op1.m, op2.m ), simd4u8::base_rep_tag() ); // per-byte (un)signed subtraction, with wrap-around: a - b
#else
    return simd4u8(
        get<0>(op1) - get<0>(op2),
        get<1>(op1) - get<1>(op2),
        get<2>(op1) - get<2>(op2),
        get<3>(op1) - get<3>(op2) );
#endif
}
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8& operator-= (simd4u8& op1, const simd4u8 op2)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ > 0
    op1.m = vsub4( op1.m, op2.m ); // per-byte (un)signed subtraction, with wrap-around: a - b
#else
    op1 = simd4u8(
        get<0>(op1) - get<0>(op2),
        get<1>(op1) - get<1>(op2),
        get<2>(op1) - get<2>(op2),
        get<3>(op1) - get<3>(op2) );
#endif
    return op1;
}

NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8 max(const simd4u8 op1, const simd4u8 op2)
{
#if defined(NVBIO_DEVICE_COMPILATION)
    return simd4u8( vmaxu4( op1.m, op2.m ), simd4u8::base_rep_tag() );
#else
    return simd4u8(
        nvbio::max( get<0>(op1), get<0>(op2) ),
        nvbio::max( get<1>(op1), get<1>(op2) ),
        nvbio::max( get<2>(op1), get<2>(op2) ),
        nvbio::max( get<3>(op1), get<3>(op2) ) );
#endif
}
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8 min(const simd4u8 op1, const simd4u8 op2)
{
#if defined(NVBIO_DEVICE_COMPILATION)
    return simd4u8( vminu4( op1.m, op2.m ), simd4u8::base_rep_tag() );
#else
    return simd4u8(
        nvbio::min( get<0>(op1), get<0>(op2) ),
        nvbio::min( get<1>(op1), get<1>(op2) ),
        nvbio::min( get<2>(op1), get<2>(op2) ),
        nvbio::min( get<3>(op1), get<3>(op2) ) );
#endif
}

NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8 and_op(const simd4u8 op1, const simd4u8 op2)
{
    return simd4u8( op1.m & op2.m, simd4u8::base_rep_tag() );
}
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8 or_op(const simd4u8 op1, const simd4u8 op2)
{
    return simd4u8( op1.m | op2.m, simd4u8::base_rep_tag() );
}

template <uint32 I>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
uint8 get(const simd4u8 op)
{
    return (op.m >> (I*8)) & 255u;
}
template <uint32 I>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
void set(simd4u8 op, const uint8 v)
{
    op.m &= ~(255u << (I*8));
    op.m |= v << (I*8);
}

NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
simd4u8 ternary_op(const simd4u8 mask, const simd4u8 op1, const simd4u8 op2)
{
    return or_op( and_op( mask, op1 ), and_op( ~mask, op2 ) );
}

} // namespace nvbio
