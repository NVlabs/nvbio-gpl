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

/*! \vector_view.h
 *   \brief Define a vector_view POD type and plain_view() for std::vector.
 */

#pragma once

#include <nvbio/basic/types.h>
#include <vector>

namespace nvbio {

///@addtogroup Basic
///@{

///
/// Wrapper class to create a "vector"-like container on top of a generic base iterator.
/// See \ref VectorWrapperExampleSection.
///
/// \tparam Iterator        base iterator type
///
template <typename Iterator, typename IndexType = uint64>
struct vector_view
{
    typedef Iterator                                                    iterator;
    typedef Iterator                                                    const_iterator;

    typedef typename std::iterator_traits<Iterator>::value_type         value_type;
    typedef typename std::iterator_traits<Iterator>::reference          reference;
    typedef typename to_const<reference>::type                          const_reference;
    typedef typename std::iterator_traits<Iterator>::pointer            pointer;
    typedef IndexType                                                   size_type;
    typedef typename std::iterator_traits<Iterator>::difference_type    difference_type;
    //typedef typename std::iterator_traits<Iterator>::distance_type     distance_type;
    typedef std::random_access_iterator_tag                             iterator_category;

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    vector_view() {}

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    vector_view(const IndexType size, Iterator vec) : m_size( size ), m_vec( vec ) {}

    /// return vector size
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    IndexType size() const { return m_size; }

    /// return vector size
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    IndexType length() const { return m_size; }

    /// return true iff size is null
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    bool empty() const { return m_size == 0; }

    /// dereference operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    const_reference operator*() const                   { NVBIO_CUDA_DEBUG_ASSERT( 0 < m_size, "vector_view: access out of bounds, %u >= %u\n", 0, m_size ); return *m_vec; }

    /// dereference operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    reference operator*()                               { NVBIO_CUDA_DEBUG_ASSERT( 0 < m_size, "vector_view: access out of bounds, %u >= %u\n", 0, m_size ); return *m_vec; }

    /// const indexing operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    const_reference operator[](const IndexType i) const { NVBIO_CUDA_DEBUG_ASSERT( i < m_size, "vector_view: access out of bounds, %u >= %u\n", i, m_size ); return m_vec[i]; }

    /// indexing operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    reference operator[](const IndexType i)             { NVBIO_CUDA_DEBUG_ASSERT( i < m_size, "vector_view: access out of bounds, %u >= %u\n", i, m_size ); return m_vec[i]; }

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

    /// automatic conversion to the basic iterator type
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    operator Iterator() const { return m_vec; }

    IndexType   m_size;
    Iterator    m_vec;
};

/// return length of a vector_view
///
template <typename Iterator>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
uint32 length(const vector_view<Iterator>& vec) { return vec.length(); }

/// return the raw pointer of a vector_view
///
template <typename T>
T raw_pointer(const vector_view<T>& vec) { return vec.base(); }

//
// --- std::vector views ----------------------------------------------------------------------------
//

/// define the plain view of a std::vector
///
template <typename T> struct plain_view_subtype< std::vector<T> > { typedef vector_view<T*> type; };

/// return the plain view of a std::vector
///
template <typename T>
vector_view<T*> plain_view(std::vector<T>& vec) { return vector_view<T*>( vec.size(), vec.size() ? &vec[0] : NULL ); }

/// return the plain view of a std::vector
///
template <typename T>
vector_view<const T*> plain_view(const std::vector<T>& vec) { return vector_view<const T*>( vec.size(), vec.size() ? &vec[0] : NULL ); }

/// return the raw pointer of a std::vector
///
template <typename T>
T* raw_pointer(std::vector<T>& vec) { return vec.size() ? &vec[0] : NULL; }

/// return the raw pointer of a std::vector
///
template <typename T>
const T* raw_pointer(const std::vector<T>& vec) { return vec.size() ? &vec[0] : NULL; }

///@} Basic

} // namespace nvbio
