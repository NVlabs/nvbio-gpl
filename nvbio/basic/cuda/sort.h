/*
 * nvbio
 * Copyright (C) 2011-2013, NVIDIA Corporation
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

/*! \file sort.h
 *   \brief Define CUDA based sort primitives.
 */

#pragma once

#include <nvbio/basic/types.h>

namespace nvbio {
namespace cuda {

/// \page sorting_page Sorting
///
/// The SortEnactor provides a convenient wrapper around the fastest CUDA sorting library available,
/// allowing to perform both key-only and key-value pair sorting of arrays with the following
/// data-types:
///
/// - uint8
/// - uint16
/// - uint32
/// - uint64
/// - (uint8,uint32)
/// - (uint16,uint32)
/// - (uint32,uint32)
/// - (uint64,uint32)
///
///

///@addtogroup Basic
///@{

///@defgroup SortEnactors Sort Enactors
/// This module implements simple classes to sort device-memory buffers of key/value pairs of various primitive types.
///@{

/// A sorting buffer to hold vectors of key-value pairs
///
template <typename Keys, typename Values = null_type>
struct SortBuffers
{
    /// constructor
    ///
    SortBuffers() : selector(0) {}

    uint32  selector;
    Keys    keys[2];
    Values  values[2];
};

/// A simple class to enact sorts of various kinds
///
struct SortEnactor
{
    /// constructor
    ///
    SortEnactor();

    /// destructor
    ///
    ~SortEnactor();

    void sort(const uint32 count, SortBuffers<uint8*, uint32*>& buffers, const uint32 begin_bit = 0, const uint32 end_bit = 8);
    void sort(const uint32 count, SortBuffers<uint16*,uint32*>& buffers, const uint32 begin_bit = 0, const uint32 end_bit = 16);
    void sort(const uint32 count, SortBuffers<uint32*,uint32*>& buffers, const uint32 begin_bit = 0, const uint32 end_bit = 32);
    void sort(const uint32 count, SortBuffers<uint32*,uint64*>& buffers, const uint32 begin_bit = 0, const uint32 end_bit = 32);
    void sort(const uint32 count, SortBuffers<uint64*,uint32*>& buffers, const uint32 begin_bit = 0, const uint32 end_bit = 64);
    void sort(const uint32 count, SortBuffers<uint8*>&          buffers, const uint32 begin_bit = 0, const uint32 end_bit = 8);
    void sort(const uint32 count, SortBuffers<uint16*>&         buffers, const uint32 begin_bit = 0, const uint32 end_bit = 16);
    void sort(const uint32 count, SortBuffers<uint32*>&         buffers, const uint32 begin_bit = 0, const uint32 end_bit = 32);
    void sort(const uint32 count, SortBuffers<uint64*>&         buffers, const uint32 begin_bit = 0, const uint32 end_bit = 64);

private:
    void*  m_impl;
};

///@} SortEnactors
///@} Basic

} // namespace cuda
} // namespace nvbio
