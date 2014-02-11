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
#include <nvbio/basic/iterator.h>

namespace nvbio {

/// \page iterators_page Iterators
///
/// NVBIO provides a few adaptable iterator classes which can be used to construct
/// different views on top of some underlying iterator:
///
/// - strided_iterator
/// - block_strided_iterator
/// - transform_iterator
/// - cached_iterator
/// - const_cached_iterator
///

///@addtogroup Basic
///@{

///@addtogroup Iterators
///@{

///
/// Wrapper class to create a strided iterator out of another base iterator, i.e:
///
///   it[  <b>j</b>  ] = base[ <b>j</b>  *  <i>stride</i> ]
///
template <typename T>
struct strided_iterator
{
    typedef typename std::iterator_traits<T>::value_type        value_type;
    typedef typename std::iterator_traits<T>::reference         reference;
    typedef typename to_const<reference>::type                  const_reference;
    typedef typename std::iterator_traits<T>::pointer           pointer;
    typedef typename std::iterator_traits<T>::difference_type   difference_type;
    //typedef typename std::iterator_traits<T>::distance_type     distance_type;
    typedef std::random_access_iterator_tag                     iterator_category;

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    strided_iterator() {}

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    strided_iterator(T vec, const uint32 stride) : m_vec( vec ), m_stride( stride ) {}

    /// const dereferencing operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    const_reference operator*() const { return *m_vec; }

    /// const indexing operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    const_reference operator[](const uint32 i) const { return m_vec[i*m_stride]; }

    /// indexing operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    reference operator[](const uint32 i) { return m_vec[i*m_stride]; }

    /// addition
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    strided_iterator<T> operator+(const uint32 i) const
    {
        return strided_iterator( m_vec + i * m_stride, m_stride );
    }

    /// iterator subtraction
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    difference_type operator-(const strided_iterator<T> it) const
    {
        return m_vec - it.m_vec;
    }

    T      m_vec;
    uint32 m_stride;
};

/// operator ==
///
template <typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
bool operator==(const strided_iterator<T> it1, const strided_iterator<T> it2)
{
    return (it1.m_vec == it2.m_vec) && (it1.m_stride == it2.m_stride);
}
/// operator !=
///
template <typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
bool operator!=(const strided_iterator<T> it1, const strided_iterator<T> it2)
{
    return (it1.m_vec != it2.m_vec) || (it1.m_stride != it2.m_stride);
}
/// operator <
///
template <typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
bool operator<(const strided_iterator<T> it1, const strided_iterator<T> it2) { return (it1.m_vec < it2.m_vec); }
/// operator <=
///
template <typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
bool operator<=(const strided_iterator<T> it1, const strided_iterator<T> it2) { return (it1.m_vec <= it2.m_vec); }
/// operator >
///
template <typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
bool operator>(const strided_iterator<T> it1, const strided_iterator<T> it2) { return (it1.m_vec > it2.m_vec); }
/// operator >=
///
template <typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
bool operator>=(const strided_iterator<T> it1, const strided_iterator<T> it2) { return (it1.m_vec >= it2.m_vec); }

///
/// Wrapper class to create a block-strided iterator out of another base iterator, i.e:
///
///   it[  <b>j</b>  ] = base[ (<b>j</b>  /  BLOCKSIZE) * <i>stride</i> + (<b>j</b>  %%  BLOCKSIZE) ]
///
template <uint32 BLOCKSIZE, typename T>
struct block_strided_iterator
{
    typedef typename std::iterator_traits<T>::value_type        value_type;
    typedef typename std::iterator_traits<T>::reference         reference;
    typedef typename to_const<reference>::type                  const_reference;
    typedef typename std::iterator_traits<T>::pointer           pointer;
    typedef typename std::iterator_traits<T>::difference_type   difference_type;
    //typedef typename std::iterator_traits<T>::distance_type     distance_type;
    typedef std::random_access_iterator_tag                     iterator_category;

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    block_strided_iterator() {}

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    block_strided_iterator(T vec, const uint32 stride, const uint32 offset = 0) : m_vec( vec ), m_offset(offset), m_stride( stride ) {}

    /// const dereferencing operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    const_reference operator*() const { return m_vec[m_offset]; }

    /// const indexing operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    const_reference operator[](const uint32 i) const { return m_vec[((i+m_offset)/BLOCKSIZE)*m_stride + ((i+m_offset)%BLOCKSIZE)]; }

    /// indexing operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    reference operator[](const uint32 i) { return m_vec[((i+m_offset)/BLOCKSIZE)*m_stride + ((i+m_offset)%BLOCKSIZE)]; }

    /// addition
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    block_strided_iterator<BLOCKSIZE,T> operator+(const uint32 i) const
    {
        return block_strided_iterator<BLOCKSIZE,T>( m_vec, m_stride, m_offset + i );
    }

    /// iterator subtraction
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    difference_type operator-(const block_strided_iterator<BLOCKSIZE,T> it) const
    {
        return (m_vec + m_offset) - (it.m_vec + it.m_offset);
    }

    T      m_vec;
    uint32 m_offset;
    uint32 m_stride;
};

/// operator ==
///
template <uint32 BLOCKSIZE,typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
bool operator==(const block_strided_iterator<BLOCKSIZE,T> it1, const block_strided_iterator<BLOCKSIZE,T> it2)
{
    return (it1.m_vec == it2.m_vec) && (it1.m_offset == it2.m_offset) && (it1.m_stride == it2.m_stride);
}
/// operator !=
///
template <uint32 BLOCKSIZE,typename T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
bool operator!=(const block_strided_iterator<BLOCKSIZE,T> it1, const block_strided_iterator<BLOCKSIZE,T> it2)
{
    return (it1.m_vec != it2.m_vec) || (it1.m_offset != it2.m_offset) || (it1.m_stride != it2.m_stride);
}

///@} Iterators
///@} Basic

} // namespace nvbio
