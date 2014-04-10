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
#include <iterator>

namespace nvbio {

/// \page vector_wrappers_page Vector Wrappers
///
/// This module implements a vector adaptor, which allows to create an "std::vector"-like
/// container on top of a base iterator.
///
/// - vector_wrapper
///
/// \section VectorWrapperExampleSection Example
///
///\code
/// // build a vector_wrapper out of a static array
/// typedef vector_wrapper<uint32*> vector_type;
///
/// uint32 storage[16];
///
/// vector_type vector( 0, storage );
///
/// // use push_back()
/// vector.push_back( 3 );
/// vector.push_back( 7 );
/// vector.push_back( 11 );
///
/// // use resize()
/// vector.resize( 4 );
///
/// // use the indexing operator[]
/// vector[3] = 8;
///
/// // use the begin() / end() iterators
/// std::sort( vector.begin(), vector.end() );
///
/// // use front() and back()
/// printf("(%u, %u)\n");                       // -> (3,11)
///\endcode
///

///@addtogroup Basic
///@{

///
/// Wrapper class to create a "vector"-like container on top of a generic base iterator.
/// See \ref VectorWrapperExampleSection.
///
/// \tparam Iterator        base iterator type
///
template <typename Iterator>
struct vector_wrapper
{
    typedef Iterator                                                    iterator;
    typedef Iterator                                                    const_iterator;

    typedef typename std::iterator_traits<Iterator>::value_type         value_type;
    typedef typename std::iterator_traits<Iterator>::reference          reference;
    typedef typename to_const<reference>::type                          const_reference;
    typedef typename std::iterator_traits<Iterator>::pointer            pointer;
    typedef uint32                                                      size_type;
    typedef typename std::iterator_traits<Iterator>::difference_type    difference_type;
    //typedef typename std::iterator_traits<Iterator>::distance_type     distance_type;
    typedef std::random_access_iterator_tag                             iterator_category;

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    vector_wrapper() {}

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    vector_wrapper(const uint32 size, Iterator vec) : m_size( size ), m_vec( vec ) {}

    /// return vector size
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 size() const { return m_size; }

    /// return vector size
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 length() const { return m_size; }

    /// return true iff size is null
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    bool empty() const { return m_size == 0; }

    /// const indexing operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    const_reference operator[](const uint32 i) const { NVBIO_CUDA_DEBUG_ASSERT( i < m_size, "vector_wrapper: access out of bounds, %u >= %u\n", i, m_size ); return m_vec[i]; }

    /// indexing operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    reference operator[](const uint32 i)             { NVBIO_CUDA_DEBUG_ASSERT( i < m_size, "vector_wrapper: access out of bounds, %u >= %u\n", i, m_size ); return m_vec[i]; }

    /// push back
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    void push_back(const_reference val) { m_vec[ m_size ] = val; m_size++; }

    /// pop back
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    void pop_back() { --m_size; }

    /// return reference to front
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    const_reference front(void) const { return m_vec[0]; }

    /// return reference to front
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    reference front(void) { return m_vec[0]; }

    /// return reference to back
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    const_reference back(void) const { return m_vec[m_size-1]; }

    /// return reference to back
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    reference back(void) { return m_vec[m_size-1]; }

    /// return the base iterator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    Iterator base() const { return m_vec; }

    /// return begin iterator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    const_iterator begin() const { return m_vec; }

    /// return end iterator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    const_iterator end() const { return m_vec + m_size; }

    /// return begin iterator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    iterator begin() { return m_vec; }

    /// return end iterator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    iterator end() { return m_vec + m_size; }

    uint32      m_size;
    Iterator    m_vec;
};

/// return length of a vector
///
template <typename Iterator>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
uint32 length(const vector_wrapper<Iterator>& vec) const { return vec.length(); }

///@} Basic

} // namespace nvbio
