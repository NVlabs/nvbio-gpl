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

#include <nvbio/sufsort/file_bwt.h>

namespace nvbio {

struct LZ4FileWriter
{
    /// constructor
    ///
    LZ4FileWriter(FILE* _file = NULL);

    /// destructor
    ///
    ~LZ4FileWriter();

    /// open a session
    ///
    void open(FILE* _file);

    /// close a session
    ///
    void close();

    /// write a block to the output
    ///
    void write(uint32 n_bytes, const void* _src);

private:
    /// encode a given block and write it to the output
    ///
    void encode_block(uint32 n_bytes, const uint8* src);

    FILE*              m_file;
    std::vector<uint8> m_buffer;
    std::vector<uint8> m_comp_buffer;
    uint32             m_buffer_size;
};

/// A class to output the BWT to an LZ4-compressed binary file
///
struct BWTLZ4Writer
{
    /// constructor
    ///
    BWTLZ4Writer();

    /// destructor
    ///
    ~BWTLZ4Writer();

    /// open
    void open(const char* output_name, const char* index_name, const char* compression);

    /// write to the bwt
    ///
    uint32 bwt_write(const uint32 n_bytes, const void* buffer);

    /// write to the index
    ///
    uint32 index_write(const uint32 n_bytes, const void* buffer);

    /// return whether the file is in a good state
    ///
    bool is_ok() const;

private:
    FILE*           output_file;
    FILE*           index_file;
    LZ4FileWriter   output_file_writer;
    LZ4FileWriter   index_file_writer;
};

} // namespace nvbio
