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

#include <nvbio/io/reads/reads_fastq.h>
#include <nvbio/basic/types.h>

#include <string.h>
#include <ctype.h>

namespace nvbio {
namespace io {

///@addtogroup IO
///@{

///@addtogroup ReadsIO
///@{

///@addtogroup ReadsIODetail
///@{

int ReadDataFile_FASTQ_parser::nextChunk(ReadDataRAM *output, uint32 max)
{
    uint32 n = 0;
    uint8  marker;

    while (n < max)
    {
        // consume spaces & newlines
        do {
            marker = get();

            // check for EOF or read errors
            if (m_file_state != FILE_OK)
            {
                break;
            }

            // count lines
            if (marker == '\n')
            {
                m_line++;
            }
        } while (marker == '\n' || marker == ' ');

        // check for EOF or read errors
        if (m_file_state != FILE_OK)
            break;

        // if the newlines didn't end in a read marker,
        // issue a parsing error...
        if (marker != '@')
        {
            m_file_state = FILE_PARSE_ERROR;
            m_error_char = marker;
            return uint32(-1);
        }

        // read all the line
        m_name.erase( m_name.begin(), m_name.end() );
        for (uint8 c = get(); c != '\n'; c = get())
        {
            if (m_file_state != FILE_OK)
            {
                log_error(stderr, "incomplete read!\n");
                m_name.erase(m_name.begin(), m_name.end());

                m_error_char = c;
                return uint32(-1);
            }

            m_name.push_back(c);
        }

        m_name.push_back('\0');

        m_line++;

        // start reading the bp read
        m_read_bp.erase( m_read_bp.begin(), m_read_bp.end() );
        for (uint8 c = get(); c != '+'; c = get())
        {
            if (m_file_state != FILE_OK)
            {
                log_error(stderr, "incomplete read!\n");
                m_name.erase(m_name.begin(), m_name.end());

                m_error_char = c;
                return uint32(-1);
            }

            if (isgraph(c))
                m_read_bp.push_back( c );

            if (c == '\n')
                m_line++;
        }

        // read all the line
        for(uint8 c = get(); c != '\n'; c = get())
        {
            if (m_file_state != FILE_OK)
            {
                log_error(stderr, "incomplete read!\n");
                m_name.erase(m_name.begin(), m_name.end());
                m_read_bp.erase(m_read_bp.begin(), m_read_bp.end());

                m_error_char = c;
                return uint32(-1);
            }
        }

        m_line++;

        // start reading the quality read
        m_read_q.erase( m_read_q.begin(), m_read_q.end() );
        for (uint8 c = get(); c != '\n'; c = get())
        {
            if (m_file_state != FILE_OK)
            {
                log_error(stderr, "incomplete read!\n");
                m_name.erase(m_name.begin(), m_name.end());
                m_read_bp.erase(m_read_bp.begin(), m_read_bp.end());

                m_error_char = c;
                return uint32(-1);
            }

            m_read_q.push_back( c );
        }

        m_line++;

        if (m_flags & FORWARD)
        {
            output->push_back(uint32( m_read_bp.size() ),
                              &m_name[0],
                              &m_read_bp[0],
                              &m_read_q[0],
                              m_quality_encoding,
                              m_truncate_read_len,
                              FORWARD );
        }
        if (m_flags & REVERSE)
        {
            output->push_back(uint32( m_read_bp.size() ),
                              &m_name[0],
                              &m_read_bp[0],
                              &m_read_q[0],
                              m_quality_encoding,
                              m_truncate_read_len,
                              REVERSE );
        }
        if (m_flags & FORWARD_COMPLEMENT)
        {
            output->push_back(uint32( m_read_bp.size() ),
                              &m_name[0],
                              &m_read_bp[0],
                              &m_read_q[0],
                              m_quality_encoding,
                              m_truncate_read_len,
                              ReadEncoding( FORWARD | COMPLEMENT ) );
        }
        if (m_flags & REVERSE_COMPLEMENT)
        {
            output->push_back(uint32( m_read_bp.size() ),
                              &m_name[0],
                              &m_read_bp[0],
                              &m_read_q[0],
                              m_quality_encoding,
                              m_truncate_read_len,
                              ReadEncoding( REVERSE | COMPLEMENT ) );
        }

        ++n;
    }

    return n;
}

ReadDataFile_FASTQ_gz::ReadDataFile_FASTQ_gz(const char *read_file_name,
                                             const QualityEncoding qualities,
                                             const uint32 max_reads,
                                             const uint32 max_read_len,
                                             const ReadEncoding flags)
    : ReadDataFile_FASTQ_parser(read_file_name, qualities, max_reads, max_read_len, flags)
{
    m_file = gzopen(read_file_name, "r");
    if (!m_file) {
        m_file_state = FILE_OPEN_FAILED;
    } else {
        m_file_state = FILE_OK;
    }

    gzbuffer(m_file, m_buffer_size);
}

ReadDataFile_FASTQ_parser::FileState ReadDataFile_FASTQ_gz::fillBuffer(void)
{
    m_buffer_size = gzread(m_file, &m_buffer[0], (uint32)m_buffer.size());
    if (m_buffer_size <= 0)
    {
        // check for EOF separately; zlib will not always return Z_STREAM_END at EOF below
        if (gzeof(m_file))
        {
            return FILE_EOF;
        } else {
            // ask zlib what happened and inform the user
            int err;
            const char *msg;

            msg = gzerror(m_file, &err);
            // we're making the assumption that we never see Z_STREAM_END here
            assert(err != Z_STREAM_END);

            log_error(stderr, "error processing FASTQ file: zlib error %d (%s)\n", err, msg);
            return FILE_STREAM_ERROR;
        }
    }

    return FILE_OK;
}

///@} // ReadsIODetail
///@} // ReadsIO
///@} // IO

} // namespace io
} // namespace nvbio
