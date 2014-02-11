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

#include <nvbio/io/output/output_types.h>

#include <stdio.h>

namespace nvbio {
namespace io {

// xxxnsubtil: BAM is little-endian, and so is x86/amd64
// we do no endinaness conversion here, though we provide
// methods to fill out integers in case we ever need to
struct DataBuffer
{
    // the maximum size for a BGZF block is 64kb + 1, which seems like a horrible design decision
    static const int BUFFER_SIZE = 60 * 1024;
    static const int BUFFER_EXTRA = 64 * 1024 - BUFFER_SIZE;

    char *buffer;
    int pos;

    DataBuffer();
    ~DataBuffer();

    // append raw data to the buffer
    void append_data(const void *data, int size);
    // apend integral values to the data buffer
    // we explicitly implement each of these instead of using a template to make sure we always
    // output the correct types
    void append_int32(int32 value);
    void append_uint32(uint32 value);
    void append_int8(int8 value);
    void append_uint8(uint8 value);

    // append a formatted string to the buffer (note: this can fail if there's too much data!)
    void append_formatted_string(const char *fmt, ...);
    // append a plain string to the buffer
    void append_string(const char *str);
    // append a linebreak
    void append_linebreak();

    // get current offset
    int get_pos(void);
    // move write pointer forward by num bytes
    void skip_ahead(int n);
    // poke data at a given offset
    void poke_data(int offset, const void *data, int size);
    void poke_int32(int offset, int32 val);
    void poke_uint16(int offset, uint16 val);
    // check whether buffer is full
    bool is_full(void);

    // grab the current pointer
    void *get_cur_ptr(void);
    // grab the base pointer
    void *get_base_ptr(void);
    // get number of bytes remaining in this buffer
    int get_remaining_size(void);

    // rewind pos back to 0
    void rewind(void);
};

} // namespace io
} // namespace nvbio
