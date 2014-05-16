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
#include <nvbio/basic/numbers.h>
#include <nvbio/basic/console.h>
#include <nvbio/basic/thrust_view.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/iterator/constant_iterator.h>

#if defined(__CUDACC__)
#include <nvbio/basic/cuda/primitives.h>
#endif

/// \page primitives_page Parallel Primitives
///
/// This module provides a set of convenience wrappers to invoke system-wide
/// CUB's parallel primitives without worrying about the memory management.
/// All temporary storage is in fact allocated within a single nvbio::vector
/// passed by the user, which can be safely reused across function calls.
///
/// - any()
/// - all()
/// - is_sorted()
/// - is_segment_sorted()
/// - reduce()
/// - inclusive_scan()
/// - exclusive_scan()
/// - copy_flagged()
/// - copy_if()
/// - runlength_encode()
///

namespace nvbio {

///@addtogroup Basic
///@{

///@defgroup Primitives Parallel Primitives
/// This module provides a set of convenience wrappers to invoke system-wide
/// CUB's parallel primitives without worrying about the memory management.
/// All temporary storage is in fact allocated within a single thrust::system_vector
/// passed by the user, which can be safely reused across function calls.
///@{

/// return true if any item in the range [0,n) evaluates to true
///
template <typename system_tag, typename PredicateIterator>
bool any(
    const uint32            n,
    const PredicateIterator pred);

/// return true if all items in the range [0,n) evaluate to true
///
template <typename system_tag, typename PredicateIterator>
bool all(
    const uint32            n,
    const PredicateIterator pred);

/// return true if the items in the range [0,n) are sorted
///
template <typename system_tag, typename Iterator>
bool is_sorted(
    const uint32            n,
    const Iterator          values);

/// return true if the items in the range [0,n) are sorted by segment, where
/// the beginning of each segment is identified by a set head flag
///
template <typename system_tag, typename Iterator, typename Headflags>
bool is_segment_sorted(
    const uint32            n,
    const Iterator          values,
    const Headflags         flags);

/// system-wide reduce
///
/// \param n                    number of items to reduce
/// \param in                   a system iterator
/// \param op                   the binary reduction operator
/// \param temp_storage         some temporary storage
///
template <typename system_tag, typename InputIterator, typename BinaryOp>
typename std::iterator_traits<InputIterator>::value_type reduce(
    const uint32                        n,
    InputIterator                       in,
    BinaryOp                            op,
    nvbio::vector<system_tag,uint8>&    temp_storage);

/// system-wide inclusive scan
///
/// \param n                    number of items to reduce
/// \param in                   a system input iterator
/// \param out                  a system output iterator
/// \param op                   the binary reduction operator
/// \param temp_storage         some temporary storage
///
template <typename system_tag, typename InputIterator, typename OutputIterator, typename BinaryOp>
void inclusive_scan(
    const uint32                        n,
    InputIterator                       in,
    OutputIterator                      out,
    BinaryOp                            op,
    nvbio::vector<system_tag,uint8>&    temp_storage);

/// system-wide exclusive scan
///
/// \param n                    number of items to reduce
/// \param in                   a system input iterator
/// \param out                  a system output iterator
/// \param op                   the binary reduction operator
/// \param identity             the identity element
/// \param temp_storage         some temporary storage
///
template <typename system_tag, typename InputIterator, typename OutputIterator, typename BinaryOp, typename Identity>
void exclusive_scan(
    const uint32                        n,
    InputIterator                       in,
    OutputIterator                      out,
    BinaryOp                            op,
    Identity                            identity,
    nvbio::vector<system_tag,uint8>&    temp_storage);

/// system-wide copy of flagged items
///
/// \param n                    number of input items
/// \param in                   a system input iterator
/// \param flags                a system flags iterator
/// \param out                  a system output iterator
/// \param temp_storage         some temporary storage
///
/// \return                     the number of copied items
///
template <typename system_tag, typename InputIterator, typename FlagsIterator, typename OutputIterator>
uint32 copy_flagged(
    const uint32                        n,
    InputIterator                       in,
    FlagsIterator                       flags,
    OutputIterator                      out,
    nvbio::vector<system_tag,uint8>&    temp_storage);

/// system-wide copy of predicated items
///
/// \param n                    number of input items
/// \param in                   a system input iterator
/// \param out                  a system output iterator
/// \param pred                 a unary predicate functor
/// \param temp_storage         some temporary storage
///
/// \return                     the number of copied items
///
template <typename system_tag, typename InputIterator, typename OutputIterator, typename Predicate>
uint32 copy_if(
    const uint32                        n,
    InputIterator                       in,
    OutputIterator                      out,
    const Predicate                     pred,
    nvbio::vector<system_tag,uint8>&    temp_storage);

/// system-wide run-length encode
///
/// \param n                    number of input items
/// \param in                   a system input iterator
/// \param out                  a system output iterator
/// \param counts               a system output count iterator
/// \param temp_storage         some temporary storage
///
/// \return                     the number of copied items
///
template <typename system_tag, typename InputIterator, typename OutputIterator, typename CountIterator>
uint32 runlength_encode(
    const uint32                        n,
    InputIterator                       in,
    OutputIterator                      out,
    CountIterator                       counts,
    nvbio::vector<system_tag,uint8>&    temp_storage);

/// system-wide run-length encode
///
/// \param n                    number of input items
/// \param keys_in              a system input iterator
/// \param values_in            a system input iterator
/// \param keys_out             a system output iterator
/// \param values_out           a system output iterator
/// \param reduction_op         a reduction operator
/// \param temp_storage         some temporary storage
///
/// \return                     the number of copied items
///
template <typename system_tag, typename KeyIterator, typename ValueIterator, typename OutputKeyIterator, typename OutputValueIterator, typename ReductionOp>
uint32 reduce_by_key(
    const uint32                        n,
    KeyIterator                         keys_in,
    ValueIterator                       values_in,
    OutputKeyIterator                   keys_out,
    OutputValueIterator                 values_out,
    ReductionOp                         reduction_op,
    nvbio::vector<system_tag,uint8>&    temp_storage);

///@} // end of the Primitives group
///@} // end of the Basic group

} // namespace nvbio

#include <nvbio/basic/primitives_inl.h>
