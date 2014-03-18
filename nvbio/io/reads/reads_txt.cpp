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

#include <nvbio/io/reads/reads_txt.h>
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

int ReadDataFile_TXT::nextChunk(ReadDataRAM *output, uint32 max)
{
    const char* name = "";

    enum State {
        SPACES   = 0,
        IN_READ  = 1
    };

    State state = SPACES;

    uint32 n = 0;

    while (1)
    {
        const uint8 c = get();

        // eat whitespaces and newlines
        if (m_file_state != FILE_OK || c == '\n' || c == ' ')
        {
            // state transition: IN_READ -> SPACES
            if (state == IN_READ)
            {
                //
                // emit a read
                //

                if (m_read_q.size() < m_read_bp.size())
                {
                    // extend the quality score vector if needed
                    m_read_q.reserve( m_read_bp.size() );
                    for (size_t i = m_read_q.size(); i < m_read_bp.size(); ++i)
                        m_read_q.push_back( char(255) );
                }

                if (m_flags & FORWARD)
                {
                    output->push_back(uint32( m_read_bp.size() ),
                                      name,
                                      &m_read_bp[0],
                                      &m_read_q[0],
                                      m_quality_encoding,
                                      m_truncate_read_len,
                                      FORWARD );
                }
                if (m_flags & REVERSE)
                {
                    output->push_back(uint32( m_read_bp.size() ),
                                      name,
                                      &m_read_bp[0],
                                      &m_read_q[0],
                                      m_quality_encoding,
                                      m_truncate_read_len,
                                      REVERSE );
                }
                if (m_flags & FORWARD_COMPLEMENT)
                {
                    output->push_back(uint32( m_read_bp.size() ),
                                      name,
                                      &m_read_bp[0],
                                      &m_read_q[0],
                                      m_quality_encoding,
                                      m_truncate_read_len,
                                      ReadEncoding( FORWARD | COMPLEMENT ) );
                }
                if (m_flags & REVERSE_COMPLEMENT)
                {
                    output->push_back(uint32( m_read_bp.size() ),
                                      name,
                                      &m_read_bp[0],
                                      &m_read_q[0],
                                      m_quality_encoding,
                                      m_truncate_read_len,
                                      ReadEncoding( REVERSE | COMPLEMENT ) );
                }

                // reset the read
                m_read_bp.erase( m_read_bp.begin(), m_read_bp.end() );

                ++n;

                // bail-out if we reached our reads quota
                if (n == max)
                    break;
            }

            // check for EOF
            if (m_file_state != FILE_OK)
                break;

            if (c == '\n')
                ++m_line;

            state = SPACES;
        }
        else
        {
            // add a character to the current read
            if (isgraph(c))
                m_read_bp.push_back( c );

            state = IN_READ;
        }
    }
    return n;
}

ReadDataFile_TXT_gz::ReadDataFile_TXT_gz(const char *read_file_name,
                                             const QualityEncoding qualities,
                                             const uint32 max_reads,
                                             const uint32 max_read_len,
                                             const ReadEncoding flags,
                                             const uint32 buffer_size)
    : ReadDataFile_TXT(read_file_name, qualities, max_reads, max_read_len, flags, buffer_size)
{
    m_file = gzopen(read_file_name, "r");
    if (!m_file) {
        m_file_state = FILE_OPEN_FAILED;
    } else {
        m_file_state = FILE_OK;
    }

    gzbuffer(m_file, m_buffer_size);
}

ReadDataFile_TXT::FileState ReadDataFile_TXT_gz::fillBuffer(void)
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

            log_error(stderr, "error processing TXT file: zlib error %d (%s)\n", err, msg);
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
