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

///@addtogroup Basic
///@{

///@addtogroup Iterators
///@{

///
/// Wrapper class to create a transform iterator out of another base iterator
/// and a function
///
template <typename T, typename Transform>
struct transform_iterator
{
    typedef typename Transform::result_type                     value_type;
    typedef value_type&                                         reference;
    typedef value_type                                          const_reference;
    typedef value_type*                                         pointer;
    typedef typename std::iterator_traits<T>::difference_type   difference_type;
    //typedef typename std::iterator_traits<T>::distance_type     distance_type;
    typedef typename std::iterator_traits<T>::iterator_category iterator_category;

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    transform_iterator() {}

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    transform_iterator(const T base, const Transform f) : m_base( base ), m_f( f ) {}

    /// copy constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    transform_iterator(const transform_iterator& it) : m_base( it.m_base ), m_f( it.m_f ) {}

    /// const indexing operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    const_reference operator[](const uint32 i) const { return m_f( m_base[i] ); }

    /// dereference operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    value_type operator*() const { return m_f( m_base[0] ); }

    /// pre-increment
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    transform_iterator<T,Transform>& operator++()
    {
        ++m_base;
        return *this;
    }

    /// post-increment
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    transform_iterator<T,Transform> operator++(int i)
    {
        transform_iterator<T,Transform> r( m_base, m_f );
        ++m_base;
        return r;
    }

    /// pre-decrement
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    transform_iterator<T,Transform>& operator--()
    {
        --m_base;
        return *this;
    }

    /// post-decrement
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    transform_iterator<T,Transform> operator--(int i)
    {
        transform_iterator<T,Transform> r( m_base, m_f );
        --m_base;
        return r;
    }

    /// addition
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    transform_iterator<T,Transform> operator+(const difference_type i) const
    {
        return transform_iterator( m_base + i, m_f );
    }

    /// subtraction
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    transform_iterator<T,Transform> operator-(const difference_type i) const
    {
        return transform_iterator( m_base - i, m_f );
    }

    /// addition
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    transform_iterator<T,Transform>& operator+=(const difference_type i)
    {
        m_base += i;
        return *this;
    }

    /// subtraction
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    transform_iterator<T,Transform>& operator-=(const difference_type i)
    {
        m_base -= i;
        return *this;
    }

    /// iterator subtraction
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    difference_type operator-(const transform_iterator<T,Transform> it) const
    {
        return m_base - it.m_base;
    }

    /// assignment
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    transform_iterator& operator=(const transform_iterator<T,Transform>& it)
    {
        m_base = it.m_base;
        m_f    = it.m_f;
        return *this;
    }

    T         m_base;
    Transform m_f;
};

/// make a transform_iterator
///
template <typename T, typename Transform>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
transform_iterator<T,Transform> make_transform_iterator(const T it, const Transform f)
{
    return transform_iterator<T,Transform>( it, f );
}


/// operator ==
///
template <typename T, typename Transform>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
bool operator==(const transform_iterator<T,Transform> it1, const transform_iterator<T,Transform> it2)
{
    return (it1.m_base == it2.m_base) && (it1.m_f == it2.m_f);
}
/// operator !=
///
template <typename T, typename Transform>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
bool operator!=(const transform_iterator<T,Transform> it1, const transform_iterator<T,Transform> it2)
{
    return (it1.m_base != it2.m_base) || (it1.m_f != it2.m_f);
}
/// operator <
///
template <typename T, typename Transform>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
bool operator<(const transform_iterator<T,Transform> it1, const transform_iterator<T,Transform> it2) { return (it1.m_base < it2.m_base); }
/// operator <=
///
template <typename T, typename Transform>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
bool operator<=(const transform_iterator<T,Transform> it1, const transform_iterator<T,Transform> it2) { return (it1.m_base <= it2.m_base); }
/// operator >
///
template <typename T, typename Transform>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
bool operator>(const transform_iterator<T,Transform> it1, const transform_iterator<T,Transform> it2) { return (it1.m_base > it2.m_base); }
/// operator >=
///
template <typename T, typename Transform>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
bool operator>=(const transform_iterator<T,Transform> it1, const transform_iterator<T,Transform> it2) { return (it1.m_base >= it2.m_base); }

///@} Iterators
///@} Basic

} // namespace nvbio
