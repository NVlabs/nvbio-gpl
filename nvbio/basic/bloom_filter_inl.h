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

namespace nvbio {

template <
    uint32   K,         // number of hash functions { Hash1 + i * Hash2 | i : 0, ..., K-1 }
    typename Hash1,     // first hash generator function
    typename Hash2,     // second hash generator function
    typename Iterator,  // storage iterator - must dereference to uint32
    typename OrOperator>
bloom_filter<K,Hash1,Hash2,Iterator,OrOperator>::bloom_filter(
    const uint64    size,
    Iterator        storage,
    const Hash1     hash1,
    const Hash2     hash2) :
    m_size( size ),
    m_storage( storage ),
    m_hash1( hash1 ),
    m_hash2( hash2 ) {}

template <
    uint32   K,         // number of hash functions { Hash1 + i * Hash2 | i : 0, ..., K-1 }
    typename Hash1,     // first hash generator function
    typename Hash2,     // second hash generator function
    typename Iterator,  // storage iterator - must dereference to uint32
    typename OrOperator>
template <typename Key>
void bloom_filter<K,Hash1,Hash2,Iterator,OrOperator>::insert(const Key key, const OrOperator or_op)
{
    const uint64 h0 = m_hash1( key );
    const uint64 h1 = m_hash2( key );

    #if defined(__CUDA_ARCH__)
    #pragma unroll
    #endif
    for (uint64 i = 0; i < K; ++i)
    {
        const uint64 r = (h0 + i * h1) & (m_size-1u);

        const uint32 word = uint32(r >> 5);
        const uint32 bit  = uint32(r) & 31u;

        or_op( &m_storage[word], (1u << bit) );
    }
}

template <
    uint32   K,         // number of hash functions { Hash1 + i * Hash2 | i : 0, ..., K-1 }
    typename Hash1,     // first hash generator function
    typename Hash2,     // second hash generator function
    typename Iterator,  // storage iterator - must dereference to uint32
    typename OrOperator>
template <typename Key>
bool bloom_filter<K,Hash1,Hash2,Iterator,OrOperator>::has(const Key key) const
{
    const uint64 h0 = m_hash1( key );
    const uint64 h1 = m_hash2( key );

    #if defined(__CUDA_ARCH__)
    #pragma unroll
    #endif
    for (uint64 i = 0; i < K; ++i)
    {
        const uint64 r = (h0 + i * h1) & (m_size-1u);

        const uint32 word = uint32(r >> 5);
        const uint32 bit  = uint32(r) & 31u;

        if ((m_storage[word] & (1u << bit)) == 0u)
            return false;
    }
    return true;
}

} // namespace nvbio
