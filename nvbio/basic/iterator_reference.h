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

namespace nvbio {

///@addtogroup Basic
///@{

///@addtogroup Iterators
///@{

///
/// An iterator reference wrapper, allowing an iterator to return assignable references.
/// Besides the standard iterator interface, the Iterator class must implement a set() method:
///
///\code
/// // set the value of the iterator
/// void set(const value_type v);
///\endcode
///
template <typename Iterator>
struct iterator_reference
{
    typedef typename std::iterator_traits<Iterator>::value_type value_type;

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE iterator_reference(Iterator it) : m_it( it ) {}

    /// copy constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE iterator_reference(const iterator_reference& ref) : m_it( ref.m_it ) {}

    /// assignment operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE iterator_reference& operator= (const iterator_reference& ref) { m_it.set( *ref.m_it ); return *this; }

    /// assignment operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE iterator_reference& operator= (const value_type s) { m_it.set( s ); return *this; }

    /// conversion operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE operator value_type() const { return *m_it; }

    Iterator m_it;
};

/// redefine the to_const meta-function for iterator_reference to just return a symbol
///
template <typename Iterator> struct to_const< iterator_reference<Iterator> >
{
    typedef typename iterator_reference<Iterator>::value_type type;
};

///@} Iterators
///@} Basic

} // namespace nvbio

namespace std {

/// overload swap for iterator_reference to make sure it does the right thing
///
template <typename Iterator>
void swap(
    nvbio::iterator_reference<Iterator> ref1,
    nvbio::iterator_reference<Iterator> ref2)
{
    typename nvbio::iterator_reference<Iterator>::value_type tmp = ref1;

    ref1 = ref2;
    ref2 = tmp;
}

} // namespace std
