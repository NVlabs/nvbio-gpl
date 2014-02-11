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

#include <zlib/zlib.h>

#include <nvbio/io/bam_format.h>
#include <nvbio/io/reads/reads.h>
#include <nvbio/io/reads/reads_priv.h>
#include <nvbio/basic/console.h>

namespace nvbio {
namespace io {

///@addtogroup IO
///@{

///@addtogroup ReadsIO
///@{

///@addtogroup ReadsIODetail
///@{

/// ReadDataFile from a BAM file
///
struct ReadDataFile_BAM : public ReadDataFile
{
    /// constructor
    ///
    ReadDataFile_BAM(const char *read_file_name,
                     const uint32 max_reads,
                     const uint32 max_read_len);

    /// read the next chunk
    ///
    virtual int nextChunk(ReadDataRAM *output, uint32 max_reads);

    /// initialize the stream
    ///
    bool init(void);

private:
    /// small utility function to read data from the gzip stream
    ///
    bool readData(void *output, unsigned int len);

    // our file pointer
    gzFile fp;
};

///@} // ReadsIODetail
///@} // ReadsIO
///@} // IO

} // namespace io
} // namespace nvbio
