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

/*! \file cached_iterator.h
 *   \brief CUDA-compatible iterator wrappers allowing to cache the dereferenced
 *   value of generic iterators
 */

#include <nvbio/basic/types.h>

namespace nvbio {

// less than
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator< (
    const const_cached_iterator<Stream>& it1,
    const const_cached_iterator<Stream>& it2)
{
    return it1.stream() < it2.stream();
}

// less than
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator<= (
    const const_cached_iterator<Stream>& it1,
    const const_cached_iterator<Stream>& it2)
{
    return it1.stream() <= it2.stream();
}

// greater than
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator> (
    const const_cached_iterator<Stream>& it1,
    const const_cached_iterator<Stream>& it2)
{
    return it1.stream() > it2.stream();
}

// greater than
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator>= (
    const const_cached_iterator<Stream>& it1,
    const const_cached_iterator<Stream>& it2)
{
    return it1.stream() >= it2.stream();
}

// equality test
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator== (
    const const_cached_iterator<Stream>& it1,
    const const_cached_iterator<Stream>& it2)
{
    return it1.stream() == it2.stream();
}

// inequality test
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator!= (
    const const_cached_iterator<Stream>& it1,
    const const_cached_iterator<Stream>& it2)
{
    return it1.stream() != it2.stream();
}

// pre-increment operator
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
const_cached_iterator<Stream>& operator++ (const_cached_iterator<Stream>& it)
{
    ++it.m_stream;
    it.m_cached_idx = uint32(-1);
    return it;
}

// post-increment operator
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
const_cached_iterator<Stream> operator++ (const_cached_iterator<Stream>& it, int dummy)
{
    const_cached_iterator<Stream> r( it.m_stream );
    ++it.m_stream;
    it.m_cached_idx = uint32(-1);
    return r;
}

// pre-decrement operator
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
const_cached_iterator<Stream>& operator-- (const_cached_iterator<Stream>& it)
{
    --it.m_stream;
    it.m_cached_idx = uint32(-1);
    return it;
}

// post-decrement operator
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
const_cached_iterator<Stream> operator-- (const_cached_iterator<Stream>& it, int dummy)
{
    const_cached_iterator<Stream> r( it.m_stream );
    --it.m_stream;
    it.m_cached_idx = uint32(-1);
    return r;
}

// add offset
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
const_cached_iterator<Stream>& operator+= (const_cached_iterator<Stream>& it, const typename const_cached_iterator<Stream>::difference_type distance)
{
    it.m_stream += distance;
    it.m_cached_idx = uint32(-1);
    return it;
}

// subtract offset
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
const_cached_iterator<Stream>& operator-= (const_cached_iterator<Stream>& it, const typename const_cached_iterator<Stream>::difference_type distance)
{
    it.m_stream -= distance;
    it.m_cached_idx = uint32(-1);
    return it;
}

// add offset
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
const_cached_iterator<Stream> operator+ (const const_cached_iterator<Stream> it, const typename const_cached_iterator<Stream>::difference_type distance)
{
    return const_cached_iterator<Stream>( it.stream() + distance );
}

// subtract offset
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
const_cached_iterator<Stream> operator- (const const_cached_iterator<Stream> it, const typename const_cached_iterator<Stream>::difference_type distance)
{
    return const_cached_iterator<Stream>( it.stream() - distance );
}

// difference
//
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
typename const_cached_iterator<Stream>::difference_type operator- (const const_cached_iterator<Stream> it1, const const_cached_iterator<Stream> it2)
{
    return it1.stream() - it2.stream();
}

} // namespace nvbio
