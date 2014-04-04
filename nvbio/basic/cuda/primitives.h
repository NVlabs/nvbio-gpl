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
#include <nvbio/basic/console.h>
#include <nvbio/basic/thrust_view.h>
#include <thrust/device_vector.h>
#include <cub/cub.cuh>

/// \page primitives_page Parallel Primitives
///
/// This module provides a set of convenience wrappers to invoke device-wide
/// CUB's parallel primitives without worrying about the memory management.
/// All temporary storage is in fact allocated within a single thrust::device_vector
/// passed by the user, which can be safely reused across function calls.
///

namespace nvbio {
namespace cuda {

///@addtogroup Basic
///@{

///@defgroup Primitives Parallel Primitives
/// This module provides a set of convenience wrappers to invoke device-wide
/// CUB's parallel primitives without worrying about the memory management.
/// All temporary storage is in fact allocated within a single thrust::device_vector
/// passed by the user, which can be safely reused across function calls.
///@{

/// make sure a given buffer is big enough
///
template <typename VectorType>
void alloc_temp_storage(VectorType& vec, const uint64 size)
{
    if (vec.size() < size)
    {
        try
        {
            vec.clear();
            vec.resize( size );
        }
        catch (...)
        {
            log_error(stderr,"alloc_temp_storage() : allocation failed!\n");
            throw;
        }
    }
}

/// device-wide reduce
///
/// \param n                    number of items to reduce
/// \param d_in                 a device iterator
/// \param op                   the binary reduction operator
/// \param d_temp_storage       some temporary storage
///
template <typename InputIterator, typename BinaryOp>
typename std::iterator_traits<InputIterator>::value_type reduce(
    const uint32                  n,
    InputIterator                 d_in,
    BinaryOp                      op,
    thrust::device_vector<uint8>& d_temp_storage)
{
    typedef typename std::iterator_traits<InputIterator>::value_type value_type;

    thrust::device_vector<value_type> d_out(1);

    size_t temp_bytes = 0;

    cub::DeviceReduce::Reduce(
        (void*)NULL, temp_bytes,
        d_in,
        d_out.begin(),
        int(n),
        op );

    temp_bytes = nvbio::max( uint64(temp_bytes), uint64(16) );
    alloc_temp_storage( d_temp_storage, temp_bytes );

    cub::DeviceReduce::Reduce(
        (void*)nvbio::plain_view( d_temp_storage ), temp_bytes,
        d_in,
        d_out.begin(),
        int(n),
        op );

    return d_out[0];
}

/// device-wide inclusive scan
///
/// \param n                    number of items to reduce
/// \param d_in                 a device input iterator
/// \param d_out                a device output iterator
/// \param op                   the binary reduction operator
/// \param d_temp_storage       some temporary storage
///
template <typename InputIterator, typename OutputIterator, typename BinaryOp>
void inclusive_scan(
    const uint32                  n,
    InputIterator                 d_in,
    OutputIterator                d_out,
    BinaryOp                      op,
    thrust::device_vector<uint8>& d_temp_storage)
{
    size_t temp_bytes = 0;

    cub::DeviceScan::InclusiveScan(
        (void*)NULL, temp_bytes,
        d_in,
        d_out,
        op,
        int(n) );

    temp_bytes = nvbio::max( uint64(temp_bytes), uint64(16) );
    alloc_temp_storage( d_temp_storage, temp_bytes );

    cub::DeviceScan::InclusiveScan(
        (void*)nvbio::plain_view( d_temp_storage ), temp_bytes,
        d_in,
        d_out,
        op,
        int(n) );
}

/// device-wide exclusive scan
///
/// \param n                    number of items to reduce
/// \param d_in                 a device input iterator
/// \param d_out                a device output iterator
/// \param op                   the binary reduction operator
/// \param identity             the identity element
/// \param d_temp_storage       some temporary storage
///
template <typename InputIterator, typename OutputIterator, typename BinaryOp, typename Identity>
void exclusive_scan(
    const uint32                  n,
    InputIterator                 d_in,
    OutputIterator                d_out,
    BinaryOp                      op,
    Identity                      identity,
    thrust::device_vector<uint8>& d_temp_storage)
{
    size_t temp_bytes = 0;

    cub::DeviceScan::ExclusiveScan(
        (void*)NULL, temp_bytes,
        d_in,
        d_out,
        op,
        identity,
        int(n) );

    temp_bytes = nvbio::max( uint64(temp_bytes), uint64(16) );
    alloc_temp_storage( d_temp_storage, temp_bytes );

    cub::DeviceScan::ExclusiveScan(
        (void*)nvbio::plain_view( d_temp_storage ), temp_bytes,
        d_in,
        d_out,
        op,
        identity,
        int(n) );
}

/// device-wide copy of flagged items
///
/// \param n                    number of input items
/// \param d_in                 a device input iterator
/// \param d_flags              a device flags iterator
/// \param d_out                a device output iterator
/// \param d_temp_storage       some temporary storage
///
/// \return                     the number of copied items
///
template <typename InputIterator, typename FlagsIterator, typename OutputIterator>
uint32 copy_flagged(
    const uint32                  n,
    InputIterator                 d_in,
    FlagsIterator                 d_flags,
    OutputIterator                d_out,
    thrust::device_vector<uint8>& d_temp_storage)
{
    size_t                         temp_bytes = 0;
    thrust::device_vector<int>     d_num_selected(1);

    cub::DeviceSelect::Flagged(
        (void*)NULL, temp_bytes,
        d_in,
        d_flags,
        d_out,
        nvbio::plain_view( d_num_selected ),
        int(n) );

    temp_bytes = nvbio::max( uint64(temp_bytes), uint64(16) );
    alloc_temp_storage( d_temp_storage, temp_bytes );

    cub::DeviceSelect::Flagged(
        (void*)nvbio::plain_view( d_temp_storage ), temp_bytes,
        d_in,
        d_flags,
        d_out,
        nvbio::plain_view( d_num_selected ),
        int(n) );

    return uint32( d_num_selected[0] );
};

/// device-wide copy of predicated items
///
/// \param n                    number of input items
/// \param d_in                 a device input iterator
/// \param d_out                a device output iterator
/// \param pred                 a unary predicate functor
/// \param d_temp_storage       some temporary storage
///
/// \return                     the number of copied items
///
template <typename InputIterator, typename OutputIterator, typename Predicate>
uint32 copy_if(
    const uint32                  n,
    InputIterator                 d_in,
    OutputIterator                d_out,
    const Predicate               pred,
    thrust::device_vector<uint8>& d_temp_storage)
{
    size_t                         temp_bytes = 0;
    thrust::device_vector<int>     d_num_selected(1);

    cub::DeviceSelect::If(
        (void*)NULL, temp_bytes,
        d_in,
        d_out,
        nvbio::plain_view( d_num_selected ),
        int(n),
        pred );

    temp_bytes = nvbio::max( uint64(temp_bytes), uint64(16) );
    alloc_temp_storage( d_temp_storage, temp_bytes );

    cub::DeviceSelect::If(
        (void*)nvbio::plain_view( d_temp_storage ), temp_bytes,
        d_in,
        d_out,
        nvbio::plain_view( d_num_selected ),
        int(n),
        pred );

    return uint32( d_num_selected[0] );
};

/// device-wide run-length encode
///
/// \param n                    number of input items
/// \param d_in                 a device input iterator
/// \param d_out                a device output iterator
/// \param d_counts             a device output count iterator
/// \param pred                 a unary predicate functor
/// \param d_temp_storage       some temporary storage
///
/// \return                     the number of copied items
///
template <typename InputIterator, typename OutputIterator, typename CountIterator, typename Predicate>
uint32 runlength_encode(
    const uint32                  n,
    InputIterator                 d_in,
    OutputIterator                d_out,
    CountIterator                 d_counts,
    thrust::device_vector<uint8>& d_temp_storage)
{
    size_t                         temp_bytes = 0;
    thrust::device_vector<int>     d_num_selected(1);

    cub::DeviceReduce::RunLengthEncode(
        (void*)NULL, temp_bytes,
        d_in,
        d_out,
        d_counts,
        nvbio::plain_view( d_num_selected ),
        int(n),
        pred );

    temp_bytes = nvbio::max( uint64(temp_bytes), uint64(16) );
    alloc_temp_storage( d_temp_storage, temp_bytes );

    cub::DeviceReduce::RunLengthEncode(
        (void*)nvbio::plain_view( d_temp_storage ), temp_bytes,
        d_in,
        d_out,
        d_counts,
        nvbio::plain_view( d_num_selected ),
        int(n),
        pred );

    return uint32( d_num_selected[0] );
};

///@} // end of the Primitives group
///@} // end of the Basic group

} // namespace cuda
} // namespace nvbio
