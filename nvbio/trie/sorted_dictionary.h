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
#include <nvbio/basic/iterator.h>

namespace nvbio {

///@addtogroup TriesModule
///@{

///@addtogroup SortedDictionarySuffixTriesModule Sorted Dictionary Suffix Tries
///@{

///
/// A node of a SortedDictionarySuffixTrie
///
struct SortedDictionaryNode
{
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    SortedDictionaryNode() {}

    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    SortedDictionaryNode(const uint32 _begin, const uint32 _end, const uint32 _level) :
        begin( _begin ), end( _end ), level( _level ) {}

    uint32 begin;
    uint32 end;
    uint32 level;
};

///
/// A suffix trie type built on a generic dictionary of sorted strings
///
/// \tparam ALPHABET_SIZE_T     the size of the alphabet
/// \tparam Iterator            an iterator to the sorted string dictionary
///
template <uint32 ALPHABET_SIZE_T, typename Iterator>
struct SortedDictionarySuffixTrie
{
    const static uint32 ALPHABET_SIZE = ALPHABET_SIZE_T;

    typedef typename std::iterator_traits<Iterator>::value_type string_type;
    typedef SortedDictionaryNode                                node_type;

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    SortedDictionarySuffixTrie(const Iterator seq, const uint32 size);

    /// return the root node of the dictionary seen as a trie
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    node_type root() const;

    /// visit the children of a given node
    ///
    /// \tparam Visitor     a visitor implementing the following interface:
    /// \code
    /// struct Visitor
    /// {
    ///     // do something with the node corresponding to character c
    ///     void visit(const uint8 c, const NodeType node);
    /// }
    /// \endcode
    ///
    template <typename Visitor>
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    void children(const node_type node, Visitor& visitor) const;

    /// return true if the node is a leaf
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    bool is_leaf(const node_type node) const;

    /// return the size of a node
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 size(const node_type node) const;

private:
    Iterator m_seq;
    uint32   m_size;
};

///@} // SortedDictionarySuffixTriesModule
///@} // TriesModule

} // namespace nvbio

#include <nvbio/trie/sorted_dictionary_inl.h>
