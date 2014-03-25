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

#include <nvbio/basic/string_set.h>
#include <nvbio/basic/thrust_view.h>
#include <nvbio/basic/cuda/sort.h>
#include <thrust/device_vector.h>
#include <thrust/transform_scan.h>
#include <thrust/sort.h>


namespace nvbio {

struct BWTParams
{
    BWTParams() :
        host_memory(8u*1024u*1024u*1024llu),
        device_memory(1024u*1024u*1024llu) {}

    uint64 host_memory;
    uint64 device_memory;
};

namespace cuda {

/// Sort the suffixes of all the strings in the given string_set
///
template <typename string_set_type, typename output_handler>
void suffix_sort(
    const string_set_type&   string_set,
          output_handler&    output,
    BWTParams*               params = NULL);

/// Build the bwt of a device-side string set
///
template <uint32 SYMBOL_SIZE, bool BIG_ENDIAN, typename storage_type, typename output_handler>
void bwt(
    const ConcatenatedStringSet<
        PackedStreamIterator< PackedStream<storage_type,uint8,SYMBOL_SIZE,BIG_ENDIAN,uint64> >,
        uint64*>                    string_set,
        output_handler&             output,
        BWTParams*                  params = NULL);

} // namespace cuda

/// Build the bwt of a large host-side string set
///
template <uint32 SYMBOL_SIZE, bool BIG_ENDIAN, typename word_type, typename output_handler>
void large_bwt(
    const ConcatenatedStringSet<
        PackedStreamIterator< PackedStream<word_type*,uint8,SYMBOL_SIZE,BIG_ENDIAN,uint64> >,
        uint64*>                    string_set,
        output_handler&             output,
        BWTParams*                  params = NULL);

} // namespace nvbio

#include <nvbio/sufsort/sufsort_inl.h>
