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

#include <nvbio/basic/packedstream.h>
#include <nvbio/basic/types.h>
#include <nvbio/basic/numbers.h>
#include <nvbio/basic/strided_iterator.h>
#include <nvbio/basic/cuda/arch.h>
#include <nvbio/basic/iterator.h>

namespace nvbio {

template <uint32 CACHE_SIZE> struct lmem_cache_tag {};

struct uncached_tag  {};


///@addtogroup Basic
///@{

///@addtogroup PackedStreams
///@{

///@addtogroup PackedStringLoaders
///@{

///
/// A utility wrapper to cache a window of a packed-string and present a wrapper string iterator
/// to the cached entries.
/// NOTE: the default implementation doesn't actually perform any caching.
///
/// \tparam StorageIterator     the underlying stream of words used to hold the packed stream (e.g. uint32, uint4)
/// \tparam SYMBOL_SIZE_T       the number of bits needed for each symbol
/// \tparam BIG_ENDIAN_T        the "endianness" of the words: if true, symbols will be packed from right to left within each word
/// \tparam Tag                 a tag to specify the type of cache at compile-time
///
template <typename StorageIterator, uint32 SYMBOL_SIZE_T, bool BIG_ENDIAN_T, typename Tag = uncached_tag>
struct PackedStringLoader {};

///
/// \relates PackedStringLoader
/// A utility wrapper to cache a window of a packed-string into a local memory buffer and present
/// a wrapper string iterator to the cached entries.
///
/// \tparam StorageIterator     the underlying stream of words used to hold the packed stream (e.g. uint32, uint4)
/// \tparam SYMBOL_SIZE_T       the number of bits needed for each symbol
/// \tparam BIG_ENDIAN_T        the "endianness" of the words: if true, symbols will be packed from right to left within each word
/// \tparam CACHE_SIZE          the size of the local memory cache, in words
///
template <typename StorageIterator, uint32 SYMBOL_SIZE_T, bool BIG_ENDIAN_T, uint32 CACHE_SIZE>
struct PackedStringLoader<StorageIterator,SYMBOL_SIZE_T,BIG_ENDIAN_T,lmem_cache_tag<CACHE_SIZE> >
{
    typedef PackedStream<StorageIterator,uint8,SYMBOL_SIZE_T,BIG_ENDIAN_T>                                                  input_stream;
    typedef input_stream                                                                                                    input_iterator;
    typedef typename std::iterator_traits<StorageIterator>::value_type                                                      storage_type;
    typedef typename PackedStream<const_cached_iterator<const storage_type*>,uint8,SYMBOL_SIZE_T,BIG_ENDIAN_T>::iterator    iterator;

    /// given a packed stream, load part of it starting at the given offset, and return an iterator
    /// to the first loaded symbol
    ///
    /// \param stream       input stream storage
    /// \param offset       offset to the first symbol to load
    /// \param length       length of the substring to load
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    iterator load(const input_stream stream, const uint32 length);

    /// given a packed stream, and a window of symbols that is virtually mapped to the cache,
    /// load a substring of it and return an iterator to the first symbol of the window.
    ///
    /// \verbatim
    /// [....|*****|####|****|...]
    ///       ^    ^    ^    ^
    ///   offset=5 |    |    offset+length
    ///            |    |
    ///        loaded_range = (6,10)
    /// \endverbatim
    ///
    /// \param stream           input stream storage
    /// \param length           length of the mapped substring
    /// \param loaded_range     range of the substring to load
    /// \param rev_flag         true if the range is specified wrt reversed coordinates
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    iterator load(
        const input_stream      stream,
        const uint32            length,
        const uint2             loaded_range,
        const uint32            rev_flag);

    storage_type lmem[CACHE_SIZE];
};

///
/// A no-op specialization of the PackedStringLoader, which actually doesn't perform any caching
///
template <typename StorageIterator, uint32 SYMBOL_SIZE_T, bool BIG_ENDIAN_T>
struct PackedStringLoader<StorageIterator,SYMBOL_SIZE_T,BIG_ENDIAN_T,uncached_tag>
{
    typedef typename std::iterator_traits<StorageIterator>::value_type                           storage_type;
    typedef PackedStream<StorageIterator,uint8,SYMBOL_SIZE_T,BIG_ENDIAN_T>                       input_stream;
    typedef input_stream                                                                         input_iterator;
    typedef typename PackedStream<StorageIterator,uint8,SYMBOL_SIZE_T,BIG_ENDIAN_T>::iterator    iterator;

    /// given a packed stream, load part of it starting at the given offset, and return an iterator
    /// to the first loaded symbol
    ///
    /// \param stream       input stream storage
    /// \param length       length of the substring to load
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    iterator load(const input_stream stream, const uint32 length);

    /// given a packed stream, and a window of symbols that is virtually mapped to the cache,
    /// load a substring of it and return an iterator to the first symbol of the window.
    ///
    /// \verbatim
    /// [....|*****|####|****|...]
    ///       ^    ^    ^    ^
    ///   offset=5 |    |    offset+length
    ///            |    |
    ///        loaded_range = (6,10)
    /// \endverbatim
    ///
    /// \param stream           input stream storage
    /// \param length           length of the mapped substring
    /// \param loaded_range     range of the substring to load
    /// \param rev_flag         true if the range is specified wrt reversed coordinates
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    iterator load(
        const input_stream      stream,
        const uint32            length,
        const uint2             substring_range,
        const uint32            rev_flag);
};

///@} PackedStringLoaders
///@} PackedStreams
///@} Basic

} // namespace nvbio

#include <nvbio/basic/packedstream_loader_inl.h>
