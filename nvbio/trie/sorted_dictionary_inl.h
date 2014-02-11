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

#include <nvbio/basic/numbers.h>
#include <nvbio/basic/transform_iterator.h>
#include <nvbio/basic/algorithms.h>

namespace nvbio {

// constructor
//
template <uint32 ALPHABET_SIZE_T, typename Iterator>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
SortedDictionarySuffixTrie<ALPHABET_SIZE_T,Iterator>::SortedDictionarySuffixTrie(const Iterator seq, const uint32 size) :
    m_seq( seq ), m_size( size )
{}

// return the root node of the dictionary seen as a trie
//
template <uint32 ALPHABET_SIZE_T, typename Iterator>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
typename SortedDictionarySuffixTrie<ALPHABET_SIZE_T,Iterator>::node_type
SortedDictionarySuffixTrie<ALPHABET_SIZE_T,Iterator>::root() const
{
    return node_type( 0u, m_size, length( m_seq[m_size-1] ) );
}

// visit the children of a given node
//
template <uint32 ALPHABET_SIZE_T, typename Iterator>
template <typename Visitor>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
void SortedDictionarySuffixTrie<ALPHABET_SIZE_T,Iterator>::children(const node_type node, Visitor& visitor) const
{
    // check subranges
    const uint8 lc = m_seq[ node.begin ][ node.level-1u ];
    const uint8 rc = m_seq[ node.end-1 ][ node.level-1u ];

    uint32 l_boundary = node.begin;
    uint8  c = lc;
    while (c < rc)
    {
        const uint32 r_boundary = uint32( upper_bound(
            c,
            make_transform_iterator( m_seq, get_char_functor<string_type>( node.level-1u ) ) + l_boundary,
            node.end - l_boundary ) - make_transform_iterator( m_seq, get_char_functor<string_type>( node.level-1u ) ) );

        visitor.visit( c, node_type( l_boundary, r_boundary, node.level-1u ) );

        l_boundary = r_boundary;
        c = m_seq[r_boundary][node.level-1u];
    }
    // last child
    visitor.visit( rc, node_type( l_boundary, node.end, node.level-1u ) );
}

// return true if the node is a leaf
//
template <uint32 ALPHABET_SIZE_T, typename Iterator>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
bool SortedDictionarySuffixTrie<ALPHABET_SIZE_T,Iterator>::is_leaf(const node_type node) const
{
    return node.level == 0u;
}

// return the size of a node
//
template <uint32 ALPHABET_SIZE_T, typename Iterator>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
uint32 SortedDictionarySuffixTrie<ALPHABET_SIZE_T,Iterator>::size(const node_type node) const
{
    return node.end - node.begin;
}

} // namespace nvbio
