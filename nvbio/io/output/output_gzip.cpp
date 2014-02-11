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

#include <nvbio/io/output/output_gzip.h>
#include <nvbio/io/fmi.h>
#include <nvbio/basic/numbers.h>
#include <nvbio/basic/types.h>

#include <stdio.h>
#include <stdarg.h>

namespace nvbio {
namespace io {

GzipCompressor::GzipCompressor()
{
    // initialize the gzip header
    // note that we don't actually care about most of these fields
    gzh.text = 0;
    gzh.time = 0;
    gzh.xflags = 0;
    gzh.extra = Z_NULL;
    gzh.extra_len = 0;
    gzh.os = 255;       // meaning unknown OS
    gzh.name = Z_NULL;
    gzh.comment = Z_NULL;
    gzh.hcrc = 0;
}

void GzipCompressor::start_block(DataBuffer& output)
{
    NVBIO_VAR_UNUSED int ret;

    // initialize the zlib stream
    stream.zalloc = Z_NULL;
    stream.zfree = Z_NULL;
    stream.opaque = Z_NULL;

    stream.next_in = NULL;
    stream.avail_in = 0;

    stream.next_out = (Bytef *) output.get_base_ptr();
    stream.avail_out = output.get_remaining_size();

    ret = deflateInit2(&stream,                 // stream object
                       Z_DEFAULT_COMPRESSION,   // compression level (0-9, default = 6)
                       Z_DEFLATED,              // compression method (no other choice...)
                       15 + 16,                 // log2 of compression window size + 16 to switch zlib to gzip format
                       9,                       // memlevel (1..9, default 8: 1 uses less memory but is slower, 9 uses more memory and is faster)
                       Z_DEFAULT_STRATEGY);     // compression strategy, may affect compression ratio and/or performance
                                                // xxxnsubtil: we might want to shmoo the compression strategy
    NVBIO_CUDA_ASSERT(ret == Z_OK);

    // set our custom gzip header
    ret = deflateSetHeader(&stream, &gzh);
    NVBIO_CUDA_ASSERT(ret == Z_OK);

    output.pos = stream.total_out;
}

// compress all of input into the current position of the output buffer
// note that this assumes output has enough space!
void GzipCompressor::compress(DataBuffer& output, DataBuffer& input)
{
    NVBIO_VAR_UNUSED int status;

    stream.next_in = (Bytef *) input.get_base_ptr();
    stream.avail_in = input.get_pos();

    stream.next_out = (Bytef *) output.get_cur_ptr();
    stream.avail_out = output.get_remaining_size();

    status = deflate(&stream, Z_NO_FLUSH);
    NVBIO_CUDA_ASSERT(status == Z_OK);
    NVBIO_CUDA_ASSERT(stream.avail_in == 0);

    output.pos = stream.total_out;
}

// finish writing this block
void GzipCompressor::end_block(DataBuffer& output)
{
    NVBIO_VAR_UNUSED int ret;

    stream.next_in = NULL;
    NVBIO_CUDA_ASSERT(stream.avail_in == 0);

    stream.next_out = (Bytef *) output.get_cur_ptr();
    stream.avail_out = output.get_remaining_size();
    NVBIO_CUDA_ASSERT(stream.avail_out);

    ret = deflate(&stream, Z_FINISH);
    NVBIO_CUDA_ASSERT(status == Z_STREAM_END);

    output.pos = stream.total_out;

    ret = deflateEnd(&stream);
    NVBIO_CUDA_ASSERT(status == Z_OK);

    output.pos = stream.total_out;
}


BGZFCompressor::BGZFCompressor()
    : GzipCompressor()
{
    // set up our gzip extra data field
    // these values are defined in the samtools spec (http://samtools.sourceforge.net/SAMv1.pdf)
    extra_data.SI1  = 66;
    extra_data.SI2  = 67;
    extra_data.SLEN = 2;

    // we set the actual BAM-specific BSIZE field to 0, since we don't know
    // ahead of time how big the output block will be; this will be updated
    // after the entire block has been compressed
    extra_data.BSIZE = 0;

    gzh.extra = (Bytef *) &extra_data;
    gzh.extra_len = sizeof(extra_data);
}

void BGZFCompressor::end_block(DataBuffer& output)
{
    // finish up the gzip block
    GzipCompressor::end_block(output);

    NVBIO_CUDA_ASSERT((output.get_pos() - 1) < 0xffff);
    // poke the BAM-specific BSIZE value in the header
    output.poke_uint16(16, (uint16)output.get_pos() - 1);
}

} // namespace io
} // namespace nvbio
