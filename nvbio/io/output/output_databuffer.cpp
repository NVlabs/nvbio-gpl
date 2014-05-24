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

#include <nvbio/io/output/output_bam.h>
#include <nvbio/basic/numbers.h>

#include <stdio.h>
#include <stdarg.h>

namespace nvbio {
namespace io {

DataBuffer::DataBuffer()
    : pos(0)
{
    buffer = (char *)malloc(BUFFER_SIZE + BUFFER_EXTRA);
    NVBIO_CUDA_ASSERT(buffer);
}

DataBuffer::~DataBuffer()
{
    if (buffer)
    {
        free(buffer);
        buffer = NULL;
    }
}

void DataBuffer::append_data(const void *data, int size)
{
    NVBIO_CUDA_ASSERT(pos + size < BUFFER_SIZE + BUFFER_EXTRA);

    memcpy(buffer + pos, data, size);
    pos += size;
}

void DataBuffer::append_int32(int32 val)
{
    append_data(&val, sizeof(val));
}

void DataBuffer::append_uint32(uint32 val)
{
    append_data(&val, sizeof(val));
}

void DataBuffer::append_int8(int8 val)
{
    append_data(&val, sizeof(val));
}

void DataBuffer::append_uint8(uint8 val)
{
    append_data(&val, sizeof(val));
}

void DataBuffer::append_formatted_string(const char *fmt, ...)
{
    int bytes_left = BUFFER_SIZE + BUFFER_EXTRA - pos;
    int bytes_written;
    va_list args;

    va_start(args, fmt);
    bytes_written = vsnprintf(&buffer[pos], bytes_left, fmt, args);

    // if we hit the very end, the string was likely truncated
    NVBIO_CUDA_ASSERT(bytes_written < bytes_left);

    pos += bytes_written;
}

void DataBuffer::append_string(const char *str)
{
    append_data(str, strlen(str));
}

int DataBuffer::get_pos(void)
{
    return pos;
}

void DataBuffer::skip_ahead(int n)
{
    NVBIO_CUDA_ASSERT(pos + n < BUFFER_SIZE + BUFFER_EXTRA);

    pos += n;
}

void DataBuffer::poke_data(int offset, const void *data, int size)
{
    NVBIO_CUDA_ASSERT(offset + size < BUFFER_SIZE + BUFFER_EXTRA);

    memcpy(buffer + offset, data, size);
}

void DataBuffer::poke_int32(int offset, int32 val)
{
    poke_data(offset, &val, sizeof(val));
}

void DataBuffer::poke_uint16(int offset, uint16 val)
{
    poke_data(offset, &val, sizeof(val));
}

bool DataBuffer::is_full(void)
{
    if (pos > BUFFER_SIZE)
    {
        return true;
    } else {
        return false;
    }
}

void DataBuffer::rewind(void)
{
    pos = 0;
}

void *DataBuffer::get_base_ptr(void)
{
    return buffer;
}

void *DataBuffer::get_cur_ptr(void)
{
    return &buffer[pos];
}

int DataBuffer::get_remaining_size(void)
{
    return BUFFER_SIZE + BUFFER_EXTRA - pos;
}

} // namespace io
} // namespace nvbio

