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
#include <nvbio/io/output/output_databuffer.h>

#include <zlib/zlib.h>
#include <stdio.h>

namespace nvbio {
namespace io {

struct GzipCompressor
{
    GzipCompressor();

    void start_block(DataBuffer& output);
    void compress(DataBuffer& output, DataBuffer& input);
    virtual void end_block(DataBuffer& output);

protected:
    // the zlib stream for this object
    z_stream stream;
    // gzip header for the stream
    gz_header_s gzh;
};

struct BGZFCompressor : public GzipCompressor
{
    struct bgzf_extra_data
    {
        uint8  SI1;     // gzip subfield identifier 1
        uint8  SI2;     // gzip subfield identifier 2
        uint16 SLEN;    // length of subfield data

        uint16 BSIZE;   // BAM total block size - 1
    } extra_data;

    BGZFCompressor();

    virtual void end_block(DataBuffer& output);
};

} // namespace io
} // namespace nvbio
