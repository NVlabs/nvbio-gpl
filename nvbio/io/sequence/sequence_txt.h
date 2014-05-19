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

#include <nvbio/io/sequence/sequence.h>
#include <nvbio/io/sequence/sequence_priv.h>
#include <nvbio/basic/console.h>

#include <zlib/zlib.h>

namespace nvbio {
namespace io {

///@addtogroup IO
///@{

///@addtogroup SequenceIO
///@{

///@addtogroup SequenceIODetail
///@{

// SequenceDataFile from a FASTQ file
// contains the code to parse FASTQ files and dump the results into a SequenceDataRAM object
// file access is done via derived classes
struct SequenceDataFile_TXT : public SequenceDataFile
{
protected:
    SequenceDataFile_TXT(
        const char*                 read_file_name,
        const QualityEncoding       quality_encoding,
        const uint32                max_reads,
        const uint32                max_read_len,
        const SequenceEncoding      flags,
        const uint32                buffer_size = 64536u)
      : SequenceDataFile(max_reads, max_read_len, flags),
        m_file_name(read_file_name),
        m_quality_encoding(quality_encoding),
        m_buffer(buffer_size),
        m_buffer_size(buffer_size),
        m_buffer_pos(buffer_size),
        m_line(0)
    {};

    // get next read chunk from file and parse it (up to max reads)
    // this can cause m_file_state to change
    virtual int nextChunk(struct SequenceDataEncoder* output, uint32 max_reads, uint32 max_bps);

    // fill m_buffer with data from the file, return the new file state
    // this should only report EOF when no more bytes could be read
    // derived classes should override this method to return actual file data
    virtual FileState fillBuffer(void) = 0;

private:
    // get next character from file
    uint8 get();

protected:
    // file name we're reading from
    const char *            m_file_name;
    // the quality encoding we're using (for FASTQ, this comes from the command line or defaults to Phred33)
    QualityEncoding         m_quality_encoding;

    // buffers input from the fastq file
    std::vector<char>       m_buffer;
    uint32                  m_buffer_size;
    uint32                  m_buffer_pos;

    // counter for which line we're at
    uint32                  m_line;

    // error reporting from the parser: stores the character that generated an error
    uint8                   m_error_char;

    // temp buffers for data coming in from the FASTQ file: read name, base pairs and qualities
    std::vector<char>  m_name;
    std::vector<uint8> m_read_bp;
    std::vector<uint8> m_read_q;
};

// loader for gzipped files
// this also works for plain uncompressed files, as zlib does that transparently
struct SequenceDataFile_TXT_gz : public SequenceDataFile_TXT
{
    SequenceDataFile_TXT_gz(const char *read_file_name,
                          const QualityEncoding qualities,
                          const uint32 max_reads,
                          const uint32 max_read_len,
                          const SequenceEncoding flags,
                          const uint32 buffer_size = 64536u);

    virtual FileState fillBuffer(void);

private:
    gzFile m_file;
};

///@} // SequenceIODetail
///@} // SequenceIO
///@} // IO

inline uint8 SequenceDataFile_TXT::get(void)
{
    if (m_buffer_pos >= m_buffer_size)
    {
        // grab more data from the underlying file
        m_file_state = fillBuffer();
        m_buffer_pos = 0;

        // if we failed to read more data, return \0
        if (m_file_state != FILE_OK)
            return 0;
    }

    return m_buffer[m_buffer_pos++];
}

} // namespace io
} // namespace nvbio
