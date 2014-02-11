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

///
/// A helper class to select a component out of a vector of interleaved data.
///
template<uint32 STRIDE, uint32 WHICH, typename BaseIterator>
struct deinterleaved_iterator
{
    typedef typename BaseIterator::value_type   value_type;
    typedef typename BaseIterator::reference    reference;
    typedef const value_type*                   pointer;
    typedef int32                               difference_type;
    typedef std::random_access_iterator_tag     iterator_category;

    typedef deinterleaved_iterator<STRIDE,WHICH,BaseIterator> this_type;

    /// constructor
    ///
    NVBIO_HOST_DEVICE_TEMPLATE
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    deinterleaved_iterator(const BaseIterator it) : m_it( it ) {}

    /// copy constructor
    ///
    NVBIO_HOST_DEVICE_TEMPLATE
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    deinterleaved_iterator(const deinterleaved_iterator& it) : m_it( it.m_it ) {}

    /// indexing operator
    ///
    NVBIO_HOST_DEVICE_TEMPLATE
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    value_type operator[] (const uint32 i) const { return m_it[ i*STRIDE + WHICH ]; }

    /// indexing operator
    ///
    NVBIO_HOST_DEVICE_TEMPLATE
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    reference operator[] (const uint32 i) { return m_it[ i*STRIDE + WHICH ]; }

    /// dereference operator
    ///
    NVBIO_HOST_DEVICE_TEMPLATE
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    reference operator*() const { return m_it[ WHICH ]; }

    /// pre-increment
    ///
    NVBIO_HOST_DEVICE_TEMPLATE
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    deinterleaved_iterator& operator++()
    {
        ++m_it;
        return *this;
    }

    /// post-increment
    ///
    NVBIO_HOST_DEVICE_TEMPLATE
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    deinterleaved_iterator operator++(int i)
    {
        this_type r( m_it );
        ++m_it;
        return r;
    }

    /// pre-decrement
    ///
    NVBIO_HOST_DEVICE_TEMPLATE
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    deinterleaved_iterator& operator--()
    {
        --m_it;
        return *this;
    }

    /// post-decrement
    ///
    NVBIO_HOST_DEVICE_TEMPLATE
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    deinterleaved_iterator operator--(int i)
    {
        this_type r( m_it );
        --m_it;
        return r;
    }

    /// addition
    ///
    NVBIO_HOST_DEVICE_TEMPLATE
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    deinterleaved_iterator operator+(const difference_type i) const
    {
        return transform_iterator( m_it + i );
    }

    /// subtraction
    ///
    NVBIO_HOST_DEVICE_TEMPLATE
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    deinterleaved_iterator operator-(const difference_type i) const
    {
        return this_type( m_it - i );
    }

    /// addition
    ///
    NVBIO_HOST_DEVICE_TEMPLATE
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    deinterleaved_iterator& operator+=(const difference_type i)
    {
        m_it += i;
        return *this;
    }

    /// subtraction
    ///
    NVBIO_HOST_DEVICE_TEMPLATE
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    deinterleaved_iterator& operator-=(const difference_type i)
    {
        m_it -= i;
        return *this;
    }

    /// iterator subtraction
    ///
    NVBIO_HOST_DEVICE_TEMPLATE
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    difference_type operator-(const deinterleaved_iterator& it) const
    {
        return m_it - it.m_it;
    }

    /// assignment
    ///
    NVBIO_HOST_DEVICE_TEMPLATE
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    deinterleaved_iterator& operator=(const deinterleaved_iterator& it)
    {
        m_it = it.m_it;
        return *this;
    }

    BaseIterator m_it;
};


} // namespace nvbio
