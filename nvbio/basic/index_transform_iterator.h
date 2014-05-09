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

#include <nvbio/basic/types.h>
#include <nvbio/basic/iterator.h>
#include <nvbio/basic/iterator_reference.h>

namespace nvbio {

///@addtogroup Basic
///@{

///@addtogroup Iterators
///@{

///
/// Wrapper class to create a transform iterator out of another base iterator
/// and an index transformation functor
///
template <typename T, typename Transform>
struct index_transform_iterator
{
    typedef index_transform_iterator<T,Transform>               this_type;
    typedef typename Transform::result_type                     value_type;
    typedef iterator_reference<this_type>                       reference;
    typedef value_type                                          const_reference;
    typedef value_type*                                         pointer;
    typedef typename std::iterator_traits<T>::difference_type   difference_type;
    //typedef typename std::iterator_traits<T>::distance_type     distance_type;
    typedef std::random_access_iterator_tag                     iterator_category;

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    index_transform_iterator() {}

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    index_transform_iterator(const T base, const Transform f, const difference_type i = 0) : m_base( base ), m_f( f ), m_index( i ) {}

    /// copy constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    index_transform_iterator(const index_transform_iterator& it) : m_base( it.m_base ), m_f( it.m_f ), m_index( it.m_index ) {}

    /// set method
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    void set(const value_type v) { m_base[ m_f( m_index ) ] = v; }

    /// const indexing operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    const_reference operator[](const uint32 i) const { return m_base[ m_f( m_index + i ) ]; }

    /// const indexing operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    reference operator[](const uint32 i) { return reference( *this + i ); }

    /// dereference operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    value_type operator*() const { return m_base[ m_f(m_index) ]; }

    /// dereference operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    reference operator*() { return reference( *this ); }

    /// pre-increment
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    index_transform_iterator<T,Transform>& operator++()
    {
        ++m_index;
        return *this;
    }

    /// post-increment
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    index_transform_iterator<T,Transform> operator++(int i)
    {
        index_transform_iterator<T,Transform> r( m_base, m_f, m_index );
        ++m_index;
        return r;
    }

    /// pre-decrement
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    index_transform_iterator<T,Transform>& operator--()
    {
        --m_index;
        return *this;
    }

    /// post-decrement
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    index_transform_iterator<T,Transform> operator--(int i)
    {
        index_transform_iterator<T,Transform> r( m_base, m_f, m_index );
        --m_index;
        return r;
    }

    /// addition
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    index_transform_iterator<T,Transform> operator+(const difference_type i) const
    {
        return index_transform_iterator( m_base, m_f, m_index + i );
    }

    /// subtraction
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    index_transform_iterator<T,Transform> operator-(const difference_type i) const
    {
        return index_transform_iterator( m_base, m_f, m_index - i );
    }

    /// addition
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    index_transform_iterator<T,Transform>& operator+=(const difference_type i)
    {
        m_index += i;
        return *this;
    }

    /// subtraction
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    index_transform_iterator<T,Transform>& operator-=(const difference_type i)
    {
        m_index -= i;
        return *this;
    }

    /// iterator subtraction
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    difference_type operator-(const index_transform_iterator<T,Transform> it) const
    {
        return m_index - it.m_index;
    }

    /// assignment
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    index_transform_iterator& operator=(const index_transform_iterator<T,Transform>& it)
    {
        m_base  = it.m_base;
        m_f     = it.m_f;
        m_index = it.m_index;
        return *this;
    }

    T               m_base;
    Transform       m_f;
    difference_type m_index;
};

/// make a transform_iterator
///
template <typename T, typename Transform>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
index_transform_iterator<T,Transform> make_index_transform_iterator(const T it, const Transform f)
{
    return index_transform_iterator<T,Transform>( it, f );
}


/// operator ==
///
template <typename T, typename Transform>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
bool operator==(const index_transform_iterator<T,Transform> it1, const index_transform_iterator<T,Transform> it2)
{
    return (it1.m_base == it2.m_base) && (it1.m_f == it2.m_f) && (it1.m_index == it2.m_index);
}
/// operator !=
///
template <typename T, typename Transform>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
bool operator!=(const index_transform_iterator<T,Transform> it1, const index_transform_iterator<T,Transform> it2)
{
    return (it1.m_base != it2.m_base) || (it1.m_f != it2.m_f) || (it1.m_index != it2.m_index);
}
/// operator <
///
template <typename T, typename Transform>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
bool operator<(const index_transform_iterator<T,Transform> it1, const index_transform_iterator<T,Transform> it2) { return (it1.m_index < it2.m_index); }
/// operator <=
///
template <typename T, typename Transform>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
bool operator<=(const index_transform_iterator<T,Transform> it1, const index_transform_iterator<T,Transform> it2) { return (it1.m_index <= it2.m_index); }
/// operator >
///
template <typename T, typename Transform>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
bool operator>(const index_transform_iterator<T,Transform> it1, const index_transform_iterator<T,Transform> it2) { return (it1.m_index > it2.m_index); }
/// operator >=
///
template <typename T, typename Transform>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
bool operator>=(const index_transform_iterator<T,Transform> it1, const index_transform_iterator<T,Transform> it2) { return (it1.m_index >= it2.m_index); }

///@} Iterators
///@} Basic

} // namespace nvbio
