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

///\page sufsort_page Sufsort Module
///\htmlonly
/// <img src="nvidia_cubes.png" style="position:relative; bottom:-10px; border:0px;"/>
///\endhtmlonly
///\par
/// This module contains a series of novel parallel algorithms to perform suffix-sorting
/// and BWT construction of very large texts and text collections.
///\par
/// For example, the single string BWT construction can be used to index the
/// whole human genome on a GPU with 5GB of memory - while the string-set
/// BWT construction algorithm has been tested with up to 500M x 100bp reads
/// on a system with as little as 32GB of system memory and 5GB of GPU memory.
///\par
/// The functions are split into two groups: functions that operate on host-side
/// strings and string-sets, and functions that operate on device-side strings
/// and string-sets. The latter are grouped into the <em>cuda</em> namespace.
///\par
/// The large string set BWT construction algorithm is a new derivation, inspired by:
/// "GPU-Accelerated BWT Construction for Large Collection of Short Reads" -
/// C.M. Liu, R.Luo, T-W. Lam
/// http://arxiv.org/abs/1401.7457
///\par
/// The large string BWT construction uses a GPU implementation of J.Kärkkäainen's
/// Blockwise Suffix Sorting framework, customized around a new GPU-based block sorter
/// that employs a mixture of a novel, high-performance MSB radix sorting algorithm
/// and a high-period DCS sorter.
/// The resulting algorithm can sort strings containing several billion characters at
/// up to 70M suffixes/s on a Tesla K40, and is practically insensitive to LCP length.
///
/// \section TechnicalOverviewSection Technical Overview
///\par
/// A complete list of the classes and functions in this module is given in the \ref Sufsort documentation.
///

///
///@defgroup Sufsort Sufsort Module
/// This module contains a series of functions to perform suffix-sorting
/// and for BWT construction of very large texts and text collections.
///

namespace nvbio {

///@addtogroup Sufsort
///@{

/// BWT construction parameters
///
struct BWTParams
{
    BWTParams() :
        host_memory(8u*1024u*1024u*1024llu),
        device_memory(2u*1024u*1024u*1024llu) {}

    uint64 host_memory;
    uint64 device_memory;
};

///@}

///@addtogroup Sufsort
///@{
namespace cuda {
///@}

///@addtogroup Sufsort
///@{

/// return the position of the primary suffix of a string
///
template <typename string_type>
typename string_type::index_type find_primary(
    const typename string_type::index_type  string_len,
    const string_type                       string);

/// Sort all the suffixes of a given string.
/// This function using an adaptation of Larsson and Sadanake's algorith, and requires
/// roughly 16b of device memory per symbol.
///
/// \tparam string_type             an iterator to the string
/// \tparam output_iterator         an iterator for the output list of suffixes
///
///
/// \param string_len               the length of the given string
/// \param string                   a device-side string
/// \param output                   iterator to the output suffixes
/// \param params                   construction parameters
///
template <typename string_type, typename output_iterator>
void suffix_sort(
    const typename stream_traits<string_type>::index_type   string_len,
    const string_type                                       string,
    output_iterator                                         output,
    BWTParams*                                              params);

/// Compute the bwt of a device-side string.
/// This function computes the bwt using an adaptation of the Blockwise Suffix Sorting
/// by J.Kärkkäinen, and can hence work in a confined amount of host and device memory
/// (as specified by \ref BWTParams).
///
/// \tparam string_type             an iterator to the string
/// \tparam output_iterator         an iterator for the output list of symbols
///
///
/// \param string_len               the length of the given string
/// \param string                   a device-side string
/// \param output                   iterator to the output suffixes
/// \param params                   construction parameters
/// \return                         position of the primary suffix / $ symbol
///
template <typename string_type, typename output_iterator>
typename string_type::index_type bwt(
    const typename string_type::index_type  string_len,
    string_type                             string,
    output_iterator                         output,
    BWTParams*                              params);

/// Sort the suffixes of all the strings in the given string set
///
/// \param string_set               a device-side packed-concatenated string-set
/// \param output                   output handler
/// \param params                   construction parameters
///
template <typename string_set_type, typename output_handler>
void suffix_sort(
    const string_set_type&   string_set,
          output_handler&    output,
    BWTParams*               params = NULL);

/// Build the bwt of a device-side string set
///
/// \tparam SYMBOL_SIZE             alphabet size, in bits per symbol
/// \tparam storage_type            underlying storage iterator (e.g. uint32*)
/// \tparam output_handler          an output handler, exposing the following interface:
///
/// \code
/// struct BWTOutputHandler
/// {
///     // process a batch of BWT symbols
///     void process(
///        const uint32  n_suffixes,        // number of sorted suffixes emitted
///        const uint8*  h_bwt,             // host-side BWT of the suffixes
///        const uint8*  d_bwt,             // device-side BWT of the suffixes
///        const uint2*  h_suffixes,        // host-side suffixes
///        const uint2*  d_suffixes,        // device-side suffixes
///        const uint32* d_indices);        // device-side sorting index into the suffixes (possibly NULL)
/// };
/// \endcode
///
///
/// \param string_set               a device-side packed-concatenated string-set
/// \param output                   output handler
/// \param params                   construction parameters
///
template <uint32 SYMBOL_SIZE, bool BIG_ENDIAN, typename storage_type, typename output_handler>
void bwt(
    const ConcatenatedStringSet<
        PackedStreamIterator< PackedStream<storage_type,uint8,SYMBOL_SIZE,BIG_ENDIAN,uint64> >,
        uint64*>                    string_set,
        output_handler&             output,
        BWTParams*                  params = NULL);

///@}

} // namespace cuda

///@addtogroup Sufsort
///@{

/// Build the bwt of a large host-side string set - the string set might not fit into GPU memory.
///
/// \tparam SYMBOL_SIZE             alphabet size, in bits per symbol
/// \tparam storage_type            underlying storage iterator (e.g. uint32*)
/// \tparam output_handler          an output handler, exposing the following interface:
///
/// \code
/// struct LargeBWTOutputHandler
/// {
///     // process a batch of BWT symbols
///     void process(
///        const uint32  n_suffixes,        // number of sorted suffixes emitted
///        const uint8*  h_bwt,             // host-side BWT of the suffixes
///        const uint8*  d_bwt,             // device-side BWT of the suffixes
///        const uint2*  h_suffixes,        // host-side suffixes
///        const uint2*  d_suffixes,        // device-side suffixes
///        const uint32* d_indices);        // device-side sorting index into the suffixes (possibly NULL)
/// };
/// \endcode
///
///
/// \param string_set               a host-side packed-concatenated string-set
/// \param output                   output handler
/// \param params                   construction parameters
///
template <uint32 SYMBOL_SIZE, bool BIG_ENDIAN, typename storage_type, typename output_handler>
void large_bwt(
    const ConcatenatedStringSet<
        PackedStreamIterator< PackedStream<storage_type,uint8,SYMBOL_SIZE,BIG_ENDIAN,uint64> >,
        uint64*>                    string_set,
        output_handler&             output,
        BWTParams*                  params = NULL);

///@}

} // namespace nvbio

#include <nvbio/sufsort/sufsort_inl.h>
