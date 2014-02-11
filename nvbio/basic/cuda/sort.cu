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

#include <nvbio/basic/cuda/sort.h>
#include <nvbio/basic/thrust_view.h>
#include <cub/cub.cuh>

namespace nvbio {
namespace cuda {

SortEnactor::SortEnactor()
{
    m_impl = NULL; // we might want to use this later for the temp storage
}
SortEnactor::~SortEnactor()
{
}

void SortEnactor::sort(const uint32 count, SortBuffers<uint8*, uint32*>& buffers)
{
    cub::DoubleBuffer<uint8>  key_buffers;
    cub::DoubleBuffer<uint32> value_buffers;

    // Create ping-pong storage wrapper
    key_buffers.selector       = buffers.selector;
    value_buffers.selector     = buffers.selector;
    key_buffers.d_buffers[0]   = buffers.keys[0];
    key_buffers.d_buffers[1]   = buffers.keys[1];
    value_buffers.d_buffers[0] = buffers.values[0];
    value_buffers.d_buffers[1] = buffers.values[1];

    size_t temp_storage_bytes = 0;

    // gauge the amount of temp storage we need
    cub::DeviceRadixSort::SortPairs( NULL, temp_storage_bytes, key_buffers, value_buffers, count );

    thrust::device_vector<uint8> d_temp( temp_storage_bytes );

    // do the real run
    cub::DeviceRadixSort::SortPairs( nvbio::plain_view( d_temp ), temp_storage_bytes, key_buffers, value_buffers, count );

    // keep track of the current buffer
    buffers.selector = key_buffers.selector;
}
void SortEnactor::sort(const uint32 count, SortBuffers<uint16*,uint32*>& buffers)
{
    cub::DoubleBuffer<uint16> key_buffers;
    cub::DoubleBuffer<uint32> value_buffers;

    // Create ping-pong storage wrapper
    key_buffers.selector       = buffers.selector;
    value_buffers.selector     = buffers.selector;
    key_buffers.d_buffers[0]   = buffers.keys[0];
    key_buffers.d_buffers[1]   = buffers.keys[1];
    value_buffers.d_buffers[0] = buffers.values[0];
    value_buffers.d_buffers[1] = buffers.values[1];

    size_t temp_storage_bytes = 0;

    // gauge the amount of temp storage we need
    cub::DeviceRadixSort::SortPairs( NULL, temp_storage_bytes, key_buffers, value_buffers, count );

    thrust::device_vector<uint8> d_temp( temp_storage_bytes );

    // do the real run
    cub::DeviceRadixSort::SortPairs( nvbio::plain_view( d_temp ), temp_storage_bytes, key_buffers, value_buffers, count );

    // keep track of the current buffer
    buffers.selector = key_buffers.selector;
}
void SortEnactor::sort(const uint32 count, SortBuffers<uint32*,uint32*>& buffers)
{
    cub::DoubleBuffer<uint32> key_buffers;
    cub::DoubleBuffer<uint32> value_buffers;

    // Create ping-pong storage wrapper
    key_buffers.selector       = buffers.selector;
    value_buffers.selector     = buffers.selector;
    key_buffers.d_buffers[0]   = buffers.keys[0];
    key_buffers.d_buffers[1]   = buffers.keys[1];
    value_buffers.d_buffers[0] = buffers.values[0];
    value_buffers.d_buffers[1] = buffers.values[1];

    size_t temp_storage_bytes = 0;

    // gauge the amount of temp storage we need
    cub::DeviceRadixSort::SortPairs( NULL, temp_storage_bytes, key_buffers, value_buffers, count );

    thrust::device_vector<uint8> d_temp( temp_storage_bytes );

    // do the real run
    cub::DeviceRadixSort::SortPairs( nvbio::plain_view( d_temp ), temp_storage_bytes, key_buffers, value_buffers, count );

    // keep track of the current buffer
    buffers.selector = key_buffers.selector;
}
void SortEnactor::sort(const uint32 count, SortBuffers<uint64*,uint32*>& buffers)
{
    cub::DoubleBuffer<uint64> key_buffers;
    cub::DoubleBuffer<uint32> value_buffers;

    // Create ping-pong storage wrapper
    key_buffers.selector       = buffers.selector;
    value_buffers.selector     = buffers.selector;
    key_buffers.d_buffers[0]   = buffers.keys[0];
    key_buffers.d_buffers[1]   = buffers.keys[1];
    value_buffers.d_buffers[0] = buffers.values[0];
    value_buffers.d_buffers[1] = buffers.values[1];

    size_t temp_storage_bytes = 0;

    // gauge the amount of temp storage we need
    cub::DeviceRadixSort::SortPairs( NULL, temp_storage_bytes, key_buffers, value_buffers, count );

    thrust::device_vector<uint8> d_temp( temp_storage_bytes );

    // do the real run
    cub::DeviceRadixSort::SortPairs( nvbio::plain_view( d_temp ), temp_storage_bytes, key_buffers, value_buffers, count );

    // keep track of the current buffer
    buffers.selector = key_buffers.selector;
}

void SortEnactor::sort(const uint32 count, SortBuffers<uint8*>& buffers)
{
    cub::DoubleBuffer<uint8> key_buffers;

    // Create ping-pong storage wrapper
    key_buffers.selector       = buffers.selector;
    key_buffers.d_buffers[0]   = buffers.keys[0];
    key_buffers.d_buffers[1]   = buffers.keys[1];

    size_t temp_storage_bytes = 0;

    // gauge the amount of temp storage we need
    cub::DeviceRadixSort::SortKeys( NULL, temp_storage_bytes, key_buffers, count );

    thrust::device_vector<uint8> d_temp( temp_storage_bytes );

    // do the real run
    cub::DeviceRadixSort::SortKeys( nvbio::plain_view( d_temp ), temp_storage_bytes, key_buffers, count );

    // keep track of the current buffer
    buffers.selector = key_buffers.selector;

}
void SortEnactor::sort(const uint32 count, SortBuffers<uint16*>& buffers)
{
    cub::DoubleBuffer<uint16> key_buffers;

    // Create ping-pong storage wrapper
    key_buffers.selector       = buffers.selector;
    key_buffers.d_buffers[0]   = buffers.keys[0];
    key_buffers.d_buffers[1]   = buffers.keys[1];

    size_t temp_storage_bytes = 0;

    // gauge the amount of temp storage we need
    cub::DeviceRadixSort::SortKeys( NULL, temp_storage_bytes, key_buffers, count );

    thrust::device_vector<uint8> d_temp( temp_storage_bytes );

    // do the real run
    cub::DeviceRadixSort::SortKeys( nvbio::plain_view( d_temp ), temp_storage_bytes, key_buffers, count );

    // keep track of the current buffer
    buffers.selector = key_buffers.selector;
}
void SortEnactor::sort(const uint32 count, SortBuffers<uint32*>& buffers)
{
    cub::DoubleBuffer<uint32> key_buffers;

    // Create ping-pong storage wrapper
    key_buffers.selector       = buffers.selector;
    key_buffers.d_buffers[0]   = buffers.keys[0];
    key_buffers.d_buffers[1]   = buffers.keys[1];

    size_t temp_storage_bytes = 0;

    // gauge the amount of temp storage we need
    cub::DeviceRadixSort::SortKeys( NULL, temp_storage_bytes, key_buffers, count );

    thrust::device_vector<uint8> d_temp( temp_storage_bytes );

    // do the real run
    cub::DeviceRadixSort::SortKeys( nvbio::plain_view( d_temp ), temp_storage_bytes, key_buffers, count );

    // keep track of the current buffer
    buffers.selector = key_buffers.selector;
}
void SortEnactor::sort(const uint32 count, SortBuffers<uint64*>& buffers)
{
    cub::DoubleBuffer<uint64> key_buffers;

    // Create ping-pong storage wrapper
    key_buffers.selector       = buffers.selector;
    key_buffers.d_buffers[0]   = buffers.keys[0];
    key_buffers.d_buffers[1]   = buffers.keys[1];

    size_t temp_storage_bytes = 0;

    // gauge the amount of temp storage we need
    cub::DeviceRadixSort::SortKeys( NULL, temp_storage_bytes, key_buffers, count );

    thrust::device_vector<uint8> d_temp( temp_storage_bytes );

    // do the real run
    cub::DeviceRadixSort::SortKeys( nvbio::plain_view( d_temp ), temp_storage_bytes, key_buffers, count );

    // keep track of the current buffer
    buffers.selector = key_buffers.selector;
}

} // namespace cuda
} // namespace nvbio
