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
#include <nvbio/basic/cached_iterator.h>
#include <nvbio/basic/vector_view.h>

namespace nvbio {

template <typename StringType>
struct lmem_selector {};

template <typename StreamType,
          typename SymbolType,
          uint32 SYMBOL_SIZE_T,
          bool   BIG_ENDIAN_T>
struct lmem_selector< vector_view< PackedStream<StreamType,SymbolType,SYMBOL_SIZE_T,BIG_ENDIAN_T> > >
{
    typedef typename std::iterator_traits<StreamType>::value_type type;

    static const uint32 SYMBOLS_PER_WORD = (sizeof(type)*8)/SYMBOL_SIZE_T;
    static const uint32 WORDS            = 512 / SYMBOLS_PER_WORD;
};

template <typename StreamType,
          typename SymbolType,
          uint32 SYMBOL_SIZE_T,
          bool   BIG_ENDIAN_T,
          typename W>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
vector_view< typename PackedStream<const_cached_iterator<const W*>,SymbolType,SYMBOL_SIZE_T,BIG_ENDIAN_T>::iterator >
make_local_string(
    vector_view< PackedStream<StreamType,SymbolType,SYMBOL_SIZE_T,BIG_ENDIAN_T> > string,
    W* lmem)
{
    const StreamType in_stream = string.begin().stream();
    const uint32     in_offset = string.begin().index();
    const uint32     N         = string.length();

    typedef typename std::iterator_traits<StreamType>::value_type word_type;

    const uint32 SYMBOLS_PER_WORD = (sizeof(word_type)*8) / SYMBOL_SIZE_T;
          uint32 word_offset      = in_offset & (SYMBOLS_PER_WORD-1);
          uint32 begin_word       = in_offset / SYMBOLS_PER_WORD;
          uint32 end_word         = (in_offset + N + SYMBOLS_PER_WORD-1) / SYMBOLS_PER_WORD;

    for (uint32 word = begin_word; word < end_word; ++word)
        lmem[word - begin_word] = in_stream[ word ];

    typedef PackedStream<const_cached_iterator<const W*>,uint8,SYMBOL_SIZE_T,BIG_ENDIAN_T> const_stream_type;
    const_stream_type clmem_stream( lmem );

    return vector_view<const_stream_type>( N, clmem_stream + word_offset );
}

} // namespace nvbio
