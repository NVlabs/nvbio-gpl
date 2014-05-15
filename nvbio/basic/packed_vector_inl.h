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

namespace nvbio {

// constructor
//
template <typename SystemTag, uint32 SYMBOL_SIZE_T, bool BIG_ENDIAN_T, typename IndexType>
PackedVector<SystemTag,SYMBOL_SIZE_T,BIG_ENDIAN_T,IndexType>::PackedVector(const index_type size) :
    m_storage( util::divide_ri( size, SYMBOLS_PER_WORD ) ), m_size( size )
{}

// reserve
//
template <typename SystemTag, uint32 SYMBOL_SIZE_T, bool BIG_ENDIAN_T, typename IndexType>
void PackedVector<SystemTag,SYMBOL_SIZE_T,BIG_ENDIAN_T,IndexType>::reserve(const index_type size)
{
    if (m_storage.size() < util::divide_ri( m_size, SYMBOLS_PER_WORD ))
        m_storage.resize( util::divide_ri( m_size, SYMBOLS_PER_WORD ) );
}

// resize
//
template <typename SystemTag, uint32 SYMBOL_SIZE_T, bool BIG_ENDIAN_T, typename IndexType>
void PackedVector<SystemTag,SYMBOL_SIZE_T,BIG_ENDIAN_T,IndexType>::resize(const index_type size)
{
    m_size = size;
    reserve(size);
}


// clear
//
template <typename SystemTag, uint32 SYMBOL_SIZE_T, bool BIG_ENDIAN_T, typename IndexType>
void PackedVector<SystemTag,SYMBOL_SIZE_T,BIG_ENDIAN_T,IndexType>::clear(void)
{
    resize(0);
}

// return the begin iterator
//
template <typename SystemTag, uint32 SYMBOL_SIZE_T, bool BIG_ENDIAN_T, typename IndexType>
typename PackedVector<SystemTag,SYMBOL_SIZE_T,BIG_ENDIAN_T,IndexType>::iterator
PackedVector<SystemTag,SYMBOL_SIZE_T,BIG_ENDIAN_T,IndexType>::begin()
{
    stream_type stream( &m_storage.front() );
    return stream.begin();
}

// return the end iterator
//
template <typename SystemTag, uint32 SYMBOL_SIZE_T, bool BIG_ENDIAN_T, typename IndexType>
typename PackedVector<SystemTag,SYMBOL_SIZE_T,BIG_ENDIAN_T,IndexType>::iterator
PackedVector<SystemTag,SYMBOL_SIZE_T,BIG_ENDIAN_T,IndexType>::end()
{
    stream_type stream( &m_storage.front() );
    return stream.begin() + m_size;
}

// return the begin iterator
//
template <typename SystemTag, uint32 SYMBOL_SIZE_T, bool BIG_ENDIAN_T, typename IndexType>
typename PackedVector<SystemTag,SYMBOL_SIZE_T,BIG_ENDIAN_T,IndexType>::const_iterator
PackedVector<SystemTag,SYMBOL_SIZE_T,BIG_ENDIAN_T,IndexType>::begin() const
{
    stream_type stream( &m_storage.front() );
    return stream.begin();
}

// return the end iterator
//
template <typename SystemTag, uint32 SYMBOL_SIZE_T, bool BIG_ENDIAN_T, typename IndexType>
typename PackedVector<SystemTag,SYMBOL_SIZE_T,BIG_ENDIAN_T,IndexType>::const_iterator
PackedVector<SystemTag,SYMBOL_SIZE_T,BIG_ENDIAN_T,IndexType>::end() const
{
    stream_type stream( &m_storage.front() );
    return stream.begin() + m_size;
}

// push back a symbol
//
template <typename SystemTag, uint32 SYMBOL_SIZE_T, bool BIG_ENDIAN_T, typename IndexType>
void PackedVector<SystemTag,SYMBOL_SIZE_T,BIG_ENDIAN_T,IndexType>::push_back(const uint8 s)
{
    if (m_storage.size() < util::divide_ri( m_size+1, SYMBOLS_PER_WORD ))
        m_storage.resize( util::divide_ri( m_size+1, SYMBOLS_PER_WORD ) );

    begin()[ m_size++ ] = s;
}

// return the base address of a symbol in the stream
// note that several symbols may share the same base address
template <typename SystemTag, uint32 SYMBOL_SIZE_T, bool BIG_ENDIAN_T, typename IndexType>
void *PackedVector<SystemTag, SYMBOL_SIZE_T, BIG_ENDIAN_T, IndexType>::addrof(const index_type i)
{
    index_type off = i / SYMBOLS_PER_WORD;
    return &m_storage[off];
}

} // namespace nvbio
