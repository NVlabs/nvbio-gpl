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

namespace priv {

// copy the support of a packed string in local memory
template <typename StreamType,
          typename SymbolType,
          uint32 SYMBOL_SIZE_T,
          bool   BIG_ENDIAN_T,
          typename W>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
typename PackedStream<const_cached_iterator<const W*>,SymbolType,SYMBOL_SIZE_T,BIG_ENDIAN_T>::iterator
make_local_string(
    const PackedStream<StreamType,SymbolType,SYMBOL_SIZE_T,BIG_ENDIAN_T>    in_stream,
    const uint32                                                            in_offset,
    const uint32                                                            N,
    W*                                                                      lmem)
{
    typedef typename std::iterator_traits<StreamType>::value_type word_type;

    const StreamType in_storage = in_stream.stream();

    const uint32 SYMBOLS_PER_WORD = (sizeof(word_type)*8) / SYMBOL_SIZE_T;
    const uint32 word_offset      = in_offset & (SYMBOLS_PER_WORD-1);
    const uint32 begin_word       = in_offset / SYMBOLS_PER_WORD;
    const uint32 end_word         = (in_offset + N + SYMBOLS_PER_WORD-1) / SYMBOLS_PER_WORD;
    //NVBIO_CUDA_DEBUG_ASSERT( (end_word - begin_word) <= LMEM_STRING_WORDS, "make_local_string(): out of bounds!\n  (%u, %u)\n", begin_word, end_word );

    for (uint32 word = begin_word; word < end_word; ++word)
        lmem[ word - begin_word ] = in_storage[ word ];

    typedef PackedStream<const_cached_iterator<const W*>,uint8,SYMBOL_SIZE_T,BIG_ENDIAN_T> const_stream_type;
    const_stream_type clmem_stream( lmem );

    return clmem_stream.begin() + word_offset;
}

// copy the support of a window of a packed string in local memory
template <typename StreamType,
          typename SymbolType,
          uint32 SYMBOL_SIZE_T,
          bool   BIG_ENDIAN_T,
          typename W>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
typename PackedStream<const_cached_iterator<const W*>,SymbolType,SYMBOL_SIZE_T,BIG_ENDIAN_T>::iterator
make_local_string(
    const PackedStream<StreamType,SymbolType,SYMBOL_SIZE_T,BIG_ENDIAN_T>    in_stream,
    const uint32                                                            in_offset,
    const uint32                                                            N,
    const uint2                                                             substring_range,
    const uint32                                                            rev_flag,
    W*                                                                      lmem)
{
    typedef typename std::iterator_traits<StreamType>::value_type word_type;

    const StreamType in_storage = in_stream.stream();

    const uint32 SYMBOLS_PER_WORD = (sizeof(word_type)*8) / SYMBOL_SIZE_T;
    const uint32 word_offset      = in_offset & (SYMBOLS_PER_WORD-1);
    const uint32 base_word        = in_offset / SYMBOLS_PER_WORD;
    const uint32 begin_word       = (in_offset + (rev_flag ? N - substring_range.y : substring_range.x)) / SYMBOLS_PER_WORD;
    const uint32 end_word         = (in_offset + (rev_flag ? N - substring_range.x : substring_range.y) + SYMBOLS_PER_WORD-1) / SYMBOLS_PER_WORD;
    //NVBIO_CUDA_DEBUG_ASSERT( (begin_word - base_word) < LMEM_STRING_WORDS && (end_word - base_word) <= LMEM_STRING_WORDS, "make_local_string(): out of bounds!\n  (%u, %u)\n", begin_word, end_word );

    for (uint32 word = begin_word; word < end_word; ++word)
        lmem[ word - base_word ] = in_storage[ word ];

    typedef PackedStream<const_cached_iterator<const W*>,uint8,SYMBOL_SIZE_T,BIG_ENDIAN_T> const_stream_type;
    const_stream_type clmem_stream( lmem );

    return clmem_stream.begin() + word_offset;
}

} // namespace priv

//
// A utility wrapper to cache a packed-string into a local memory buffer and present a wrapper
// string iterator.
//
template <typename StorageIterator, uint32 SYMBOL_SIZE_T, bool BIG_ENDIAN_T, uint32 CACHE_SIZE>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
typename PackedStringLoader<StorageIterator,SYMBOL_SIZE_T,BIG_ENDIAN_T,lmem_cache_tag<CACHE_SIZE> >::iterator
PackedStringLoader<StorageIterator,SYMBOL_SIZE_T,BIG_ENDIAN_T,lmem_cache_tag<CACHE_SIZE> >::load(const input_stream stream, const uint32 offset, const uint32 length)
{
    return priv::make_local_string( stream, offset, length, lmem );
}

template <typename StorageIterator, uint32 SYMBOL_SIZE_T, bool BIG_ENDIAN_T, uint32 CACHE_SIZE>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
typename PackedStringLoader<StorageIterator,SYMBOL_SIZE_T,BIG_ENDIAN_T,lmem_cache_tag<CACHE_SIZE> >::iterator
PackedStringLoader<StorageIterator,SYMBOL_SIZE_T,BIG_ENDIAN_T,lmem_cache_tag<CACHE_SIZE> >::load(
    const input_stream      stream,
    const uint32            offset,
    const uint32            length,
    const uint2             substring_range,
    const uint32            rev_flag)
{
    return priv::make_local_string( stream, offset, length, substring_range, rev_flag, lmem );
}

//
// A utility wrapper to cache a packed-string into a local memory buffer and present a wrapper
// string iterator.
//
template <typename StorageIterator, uint32 SYMBOL_SIZE_T, bool BIG_ENDIAN_T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
typename PackedStringLoader<StorageIterator,SYMBOL_SIZE_T,BIG_ENDIAN_T,uncached_tag>::iterator
PackedStringLoader<StorageIterator,SYMBOL_SIZE_T,BIG_ENDIAN_T,uncached_tag>::load(const input_stream stream, const uint32 offset, const uint32 length)
{
    return stream + offset;
}

template <typename StorageIterator, uint32 SYMBOL_SIZE_T, bool BIG_ENDIAN_T>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
typename PackedStringLoader<StorageIterator,SYMBOL_SIZE_T,BIG_ENDIAN_T,uncached_tag>::iterator
PackedStringLoader<StorageIterator,SYMBOL_SIZE_T,BIG_ENDIAN_T,uncached_tag>::load(
    const input_stream      stream,
    const uint32            offset,
    const uint32            length,
    const uint2             substring_range,
    const uint32            rev_flag)
{
    return stream + offset;
}

} // namespace nvbio
