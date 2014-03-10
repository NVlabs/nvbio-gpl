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

#include <nvbio/io/reads/reads.h>

namespace nvbio {
namespace io {

///@addtogroup IO
///@{

///@addtogroup ReadsIO
///@{

///@addtogroup ReadsIODetail
///@{

/// abstract file-backed ReadDataStream
///
struct ReadDataFile : public ReadDataStream
{
    /// enum describing various possible file states
    ///
    typedef enum {
        // not yet opened (initial state)
        FILE_NOT_READY,
        // unable to open file (e.g., file not found)
        FILE_OPEN_FAILED,
        // ready to read
        FILE_OK,
        // reached EOF
        FILE_EOF,
        // file stream error (e.g., gzip CRC failure)
        FILE_STREAM_ERROR,
        // file format parsing error (e.g., bad FASTQ file)
        FILE_PARSE_ERROR
    } FileState;

protected:
    ReadDataFile(const uint32 max_reads,
                 const uint32 truncate_read_len,
                 const ReadEncoding flags)
      : ReadDataStream(truncate_read_len),
        m_max_reads(max_reads),
        m_flags(flags),
        m_loaded(0),
        m_file_state(FILE_NOT_READY)
    {};

public:
    /// grab the next batch of reads into a host memory buffer
    ///
    virtual ReadData *next(const uint32 batch_size);

    /// returns true if the stream is ready to read from
    ///
    virtual bool is_ok(void)
    {
        return m_file_state == FILE_OK;
    };

protected:
    virtual int nextChunk(ReadDataRAM *output, uint32 max_reads) = 0;

    uint32                  m_max_reads;
    ReadEncoding            m_flags;
    uint32                  m_loaded;

    // current file state
    FileState               m_file_state;
};

///@} // ReadsIODetail
///@} // ReadsIO
///@} // IO

} // namespace io
} // namespace nvbio
