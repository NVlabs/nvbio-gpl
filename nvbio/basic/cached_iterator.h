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

#pragma once

#include <nvbio/basic/types.h>
#include <nvbio/basic/iterator.h>

namespace nvbio {

///@addtogroup Basic
///@{

///@addtogroup Iterators
///@{

///
/// A simple class to cache accesses to a stream into a register. For example,
/// this iterator can be useful to wrap the underlying (uint32 or uint4) storage
/// of PackedStream's, so as to avoid repeated accesses to long latency memories.
///
/// \section CachedIteratorExample Example
///
/// \code
/// void write_stream(uint32* gmem_storage)
/// {
///     typedef nvbio::cached_iterator<const uint32*> cached_storage_type;
///     nvbio::PackedStream<cached_storage_type, uint8, 2, false> packed_stream( gmem_storage );
///     packed_stream[0]  = 1;
///     packed_stream[5]  = 0;
///     packed_stream[10] = 1;
///     packed_stream[15] = 0;      // all the writes above, which happen to hit the same word
///                                 // gmem_storage[0], will be cached.
///     packed_stream[20] = 1;      // this write will trigger a new memory access to gmem_storage[1].
///     return packed_stream[25];   // this read will be cached.
/// }
/// \endcode
///
template <typename InputStream>
struct cached_iterator
{
    typedef typename std::iterator_traits<InputStream>::value_type      value_type;
    typedef typename std::iterator_traits<InputStream>::reference       reference;
    typedef typename std::iterator_traits<InputStream>::pointer         pointer;
    typedef typename std::iterator_traits<InputStream>::difference_type difference_type;
    //typedef typename std::iterator_traits<InputStream>::distance_type   distance_type;
    typedef typename std::random_access_iterator_tag                    iterator_category;

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE cached_iterator()
        : m_cache_idx(uint32(-1)) {}

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE cached_iterator(InputStream stream)
        : m_stream( stream ), m_cache_idx(uint32(-1)), m_cache_val(value_type()) {}

    /// destructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE ~cached_iterator()
    {
        if (m_cache_idx != uint32(-1))
            m_stream[ m_cache_idx ] = m_cache_val;
    }

    /// indexing operator
    ///
    /// \param i        requested value
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE value_type& operator[] (const uint32 i)
    {
        if (m_cache_idx == i)
            return m_cache_val;

        if (m_cache_idx != uint32(-1))
            m_stream[ m_cache_idx ] = m_cache_val;

        m_cache_idx = i;
        m_cache_val = m_stream[i];
        return m_cache_val;
    }

private:
    InputStream m_stream;
    uint32      m_cache_idx;
    value_type  m_cache_val;
};

///
/// A simple class to cache accesses to a stream into a register. For example,
/// this iterator can be useful to wrap the underlying (uint32 or uint4) storage
/// of PackedStream's, so as to avoid repeated accesses to long latency memories.
///
/// \section ConstCachedIteratorExample Example
///
/// \code
/// uint8 read_stream(const uint32* gmem_storage)
/// {
///     typedef nvbio::const_cached_iterator<const uint32*> cached_storage_type;
///     nvbio::PackedStream<cached_storage_type, uint8, 2, false> packed_stream( gmem_storage );
///     return packed_stream[0]  |
///            packed_stream[5]  |
///            packed_stream[10] |
///            packed_stream[15] | // all the reads above, which happen to hit the same word
///                                // gmem_storage[0], will be cached.
///            packed_stream[20];  // this read will trigger a new memory access to gmem_storage[1].
/// }
/// \endcode
///
template <typename InputStream>
struct const_cached_iterator
{
    typedef typename std::iterator_traits<InputStream>::value_type      value_type;
    typedef typename std::iterator_traits<InputStream>::reference       reference;
    typedef typename std::iterator_traits<InputStream>::pointer         pointer;
    typedef typename std::iterator_traits<InputStream>::difference_type difference_type;
    //typedef typename std::iterator_traits<InputStream>::distance_type   distance_type;
    typedef typename std::random_access_iterator_tag                    iterator_category;

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE const_cached_iterator() {}

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE const_cached_iterator(InputStream stream)
        : m_stream( stream ), m_cache_idx(uint32(-1)), m_cache_val(value_type()) {}

    /// indexing operator
    ///
    /// \param i        requested value
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE value_type operator[] (const uint32 i) const
    {
        if (m_cache_idx == i)
            return m_cache_val;

        m_cache_idx = i;
        m_cache_val = m_stream[i];
        return m_cache_val;
    }

    /// base stream
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    InputStream stream() const { return m_stream; }

            InputStream m_stream;
    mutable uint32      m_cache_idx;
    mutable value_type  m_cache_val;
};

/// make a cached iterator
///
template <typename InputStream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE 
cached_iterator<InputStream> make_cached_iterator(InputStream it)
{
    return cached_iterator<InputStream>( it );
}

/// make a const cached iterator
///
template <typename InputStream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE 
const_cached_iterator<InputStream> make_const_cached_iterator(InputStream it)
{
    return cached_iterator<InputStream>( it );
}

/// less than
///
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator< (
    const const_cached_iterator<Stream>& it1,
    const const_cached_iterator<Stream>& it2);

/// greater than
///
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator> (
    const const_cached_iterator<Stream>& it1,
    const const_cached_iterator<Stream>& it2);

/// less than
///
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator<= (
    const const_cached_iterator<Stream>& it1,
    const const_cached_iterator<Stream>& it2);

/// greater than
///
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator>= (
    const const_cached_iterator<Stream>& it1,
    const const_cached_iterator<Stream>& it2);

/// equality test
///
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator== (
    const const_cached_iterator<Stream>& it1,
    const const_cached_iterator<Stream>& it2);

/// inequality test
///
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE bool operator!= (
    const const_cached_iterator<Stream>& it1,
    const const_cached_iterator<Stream>& it2);

/// pre-increment operator
///
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
const_cached_iterator<Stream>& operator++ (const_cached_iterator<Stream>& it);

/// post-increment operator
///
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
const_cached_iterator<Stream> operator++ (const_cached_iterator<Stream>& it, int dummy);

/// pre-decrement operator
///
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
const_cached_iterator<Stream>& operator-- (const_cached_iterator<Stream>& it);

/// post-decrement operator
///
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
const_cached_iterator<Stream> operator-- (const_cached_iterator<Stream>& it, int dummy);

/// add offset
///
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
const_cached_iterator<Stream>& operator+= (const_cached_iterator<Stream>& it, const typename const_cached_iterator<Stream>::difference_type distance);

/// subtract offset
///
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
const_cached_iterator<Stream>& operator-= (const_cached_iterator<Stream>& it, const typename const_cached_iterator<Stream>::difference_type distance);

/// add offset
///
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
const_cached_iterator<Stream> operator+ (const const_cached_iterator<Stream> it, const typename const_cached_iterator<Stream>::difference_type distance);

/// subtract offset
///
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
const_cached_iterator<Stream> operator- (const const_cached_iterator<Stream> it, const typename const_cached_iterator<Stream>::difference_type distance);

/// difference
///
template <typename Stream>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
typename const_cached_iterator<Stream>::difference_type operator- (const const_cached_iterator<Stream> it1, const const_cached_iterator<Stream> it2);

///@} Iterators
///@} Basic

} // namespace nvbio

#include <nvbio/basic/cached_iterator_inl.h>
