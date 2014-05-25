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
#include <nvbio/basic/packedstream.h>
#include <nvbio/basic/packedstream_loader.h>
#include <nvbio/basic/vector_view.h>
#include <nvbio/strings/infix.h>


namespace nvbio {

///@addtogroup Strings
///@{

///@defgroup StringPrefetchers String Prefetchers
///
/// This module implements a set of string <i>prefetchers</i> for common string types.
/// The idea behind prefetching is that on some CUDA architectures it's often useful
/// to pre-load the words of memory where strings are stored in a local-memory cache before
/// running expensive algorithms on them (especially with packed-strings).
/// This is because local-memory reads guarantee fully coalesced accesses and implement
/// efficient L1 caching.
///
///@{

///
/// A class to prefetch a string using a given caching strategy
///
/// \tparam StringType      the input string type
/// \tparam CacheTag        the cache type
///
template <typename StringType, typename CacheTag>
struct StringPrefetcher
{
    typedef StringType  input_string_type;
    typedef StringType        string_type;

    /// given a string, prefetch all its content and return a new string object
    /// wrapping the cached version
    ///
    /// \param string       input string
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    const string_type& load(const input_string_type& string) { return string; }

    /// given a string, prefetch the contents of a substring and return a new string object
    /// wrapping the cached version
    ///
    /// \param string           input string
    /// \param range            range of the substring to load
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    const string_type& load(
        const input_string_type& string,
        const uint2              range) { return string; }
};

///
/// A class to prefetch a packed string using a local-memory cache
///
/// \tparam StorageIterator     the underlying packed string storage iterator
/// \tparam SYMBOL_SIZE_T       the size of the packed symbols, in bits
/// \tparam BIG_ENDIAN_T        the endianness of the packing
/// \tparam CACHE_SIZE          the local-memory cache size, in words
///
template <typename StorageIterator, uint32 SYMBOL_SIZE_T, bool BIG_ENDIAN_T, uint32 CACHE_SIZE>
struct StringPrefetcher<
    vector_view< PackedStream<StorageIterator,uint8,SYMBOL_SIZE_T,BIG_ENDIAN_T> >,
    lmem_cache_tag<CACHE_SIZE> >
{
    typedef vector_view< PackedStream<StorageIterator,uint8,SYMBOL_SIZE_T,BIG_ENDIAN_T> >                           input_string_type;
    typedef PackedStringLoader<StorageIterator,SYMBOL_SIZE_T,BIG_ENDIAN_T,lmem_cache_tag<CACHE_SIZE> >              loader_type;
    typedef vector_view<typename loader_type::iterator>                                                             string_type;

    /// given a string, prefetch all its content and return a new string object
    /// wrapping the cached version
    ///
    /// \param string       input string
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    string_type load(const input_string_type& string)
    {
        return string_type(
            string.size(),
            loader.load( string.base(),
                         string.size() ) );
    }

    /// given a string, prefetch the contents of a substring and return a new string object
    /// wrapping the cached version
    ///
    /// \param string           input string
    /// \param range            range of the substring to load
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    string_type load(
        const input_string_type& string,
        const uint2              range)
    {
        return string_type(
            string.size(),
            loader.load( string.base(),
                         string.size(),
                         range,
                         false ) );
    }

    loader_type loader;
};

///
/// A class to prefetch an infix built on top of a PackedStream using a local-memory cache
///
/// \tparam StorageIterator     the underlying packed string storage iterator
/// \tparam SYMBOL_SIZE_T       the size of the packed symbols, in bits
/// \tparam BIG_ENDIAN_T        the endianness of the packing
/// \tparam CACHE_SIZE          the local-memory cache size, in words
///
template <typename InfixCoordType, typename StorageIterator, uint32 SYMBOL_SIZE_T, bool BIG_ENDIAN_T, uint32 CACHE_SIZE>
struct StringPrefetcher<
    Infix< PackedStream<StorageIterator,uint8,SYMBOL_SIZE_T,BIG_ENDIAN_T>,
           InfixCoordType >,
    lmem_cache_tag<CACHE_SIZE> >
{
    typedef Infix< PackedStream<StorageIterator,uint8,SYMBOL_SIZE_T,BIG_ENDIAN_T>,
                   InfixCoordType>                                                                                  input_string_type;
    typedef PackedStringLoader<StorageIterator,SYMBOL_SIZE_T,BIG_ENDIAN_T,lmem_cache_tag<CACHE_SIZE> >              loader_type;
    typedef vector_view<typename loader_type::iterator>                                                             string_type;

    /// given a string, prefetch all its content and return a new string object
    /// wrapping the cached version
    ///
    /// \param string       input string
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    string_type load(const input_string_type& string)
    {
        return string_type(
            string.size(),
            loader.load( string.m_string + string.range().x,
                         string.size() ) );
    }

    /// given a string, prefetch the contents of a substring and return a new string object
    /// wrapping the cached version
    ///
    /// \param string           input string
    /// \param range            range of the substring to load
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    string_type load(
        const input_string_type& string,
        const uint2              range)
    {
        return string_type(
            string.size(),
            loader.load( string.m_string + string.range().x,
                         string.size(),
                         range,
                         false ) );
    }

    loader_type loader;
};

///
/// A class to prefetch an infix built on top of a vector_view of a PackedStream using a local-memory cache
///
/// \tparam StorageIterator     the underlying packed string storage iterator
/// \tparam SYMBOL_SIZE_T       the size of the packed symbols, in bits
/// \tparam BIG_ENDIAN_T        the endianness of the packing
/// \tparam CACHE_SIZE          the local-memory cache size, in words
///
template <typename InfixCoordType, typename StorageIterator, uint32 SYMBOL_SIZE_T, bool BIG_ENDIAN_T, uint32 CACHE_SIZE>
struct StringPrefetcher<
    Infix< vector_view< PackedStream<StorageIterator,uint8,SYMBOL_SIZE_T,BIG_ENDIAN_T> >,
           InfixCoordType >,
    lmem_cache_tag<CACHE_SIZE> >
{
    typedef Infix< vector_view< PackedStream<StorageIterator,uint8,SYMBOL_SIZE_T,BIG_ENDIAN_T> >,
                   InfixCoordType>                                                                                  input_string_type;
    typedef PackedStringLoader<StorageIterator,SYMBOL_SIZE_T,BIG_ENDIAN_T,lmem_cache_tag<CACHE_SIZE> >              loader_type;
    typedef vector_view<typename loader_type::iterator>                                                             string_type;

    /// given a string, prefetch all its content and return a new string object
    /// wrapping the cached version
    ///
    /// \param string       input string
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    string_type load(const input_string_type& string)
    {
        return string_type(
            string.size(),
            loader.load( string.m_string.base() + string.range().x,
                         string.size() ) );
    }

    /// given a string, prefetch the contents of a substring and return a new string object
    /// wrapping the cached version
    ///
    /// \param string           input string
    /// \param range            range of the substring to load
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    string_type load(
        const input_string_type& string,
        const uint2              range)
    {
        return string_type(
            string.size(),
            loader.load( string.m_string.base() + string.range().x,
                         string.size(),
                         range,
                         false ) );
    }

    loader_type loader;
};

///@} StringPrefetchers
///@} Strings

} // namespace nvbio
