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

#include <nvbio/io/sequence/sequence_fasta.h>
#include <nvbio/io/sequence/sequence_encoder.h>
#include <nvbio/basic/types.h>
#include <nvbio/basic/timer.h>

#include <string.h>
#include <ctype.h>

namespace nvbio {
namespace io {

///@addtogroup IO
///@{

///@addtogroup SequenceIO
///@{

///@addtogroup SequenceIODetail
///@{

namespace { // anonymous namespace

struct FASTAHandler
{
    FASTAHandler(
        SequenceDataEncoder*    output,
        const uint32            flags,
        const uint32            truncate_read_len) :
        m_output( output ),
        m_flags( flags ),
        m_truncate_read_len( truncate_read_len ) {}

    void push_back(const char* id, const uint32 read_len, const uint8* bp)
    {
        if (m_quals.size() < size_t( read_len ))
            m_quals.resize( read_len, 50u );

        if (m_flags & FORWARD)
        {
            m_output->push_back(
                read_len,
                id,
                bp,
                &m_quals[0],
                Phred,
                m_truncate_read_len,
                SequenceDataEncoder::NO_OP );
        }
        if (m_flags & REVERSE)
        {
            m_output->push_back(
                read_len,
                id,
                bp,
                &m_quals[0],
                Phred,
                m_truncate_read_len,
                SequenceDataEncoder::REVERSE_OP );
        }
        if (m_flags & FORWARD_COMPLEMENT)
        {
            m_output->push_back(
                read_len,
                id,
                bp,
                &m_quals[0],
                Phred,
                m_truncate_read_len,
                SequenceDataEncoder::COMPLEMENT_OP );
        }
        if (m_flags & REVERSE_COMPLEMENT)
        {
            m_output->push_back(
                read_len,
                id,
                bp,
                &m_quals[0],
                Phred,
                m_truncate_read_len,
                SequenceDataEncoder::REVERSE_COMPLEMENT_OP );
        }
    }

    SequenceDataEncoder*    m_output;
    const uint32            m_flags;
    const uint32            m_truncate_read_len;
    std::vector<uint8>      m_quals;
};

} // anonymous namespace

// constructor
//
SequenceDataFile_FASTA_gz::SequenceDataFile_FASTA_gz(
    const char*             read_file_name,
    const QualityEncoding   qualities,
    const uint32            max_reads,
    const uint32            max_read_len,
    const SequenceEncoding  flags) :
    SequenceDataFile( max_reads, max_read_len, flags ),
    m_fasta_reader( read_file_name )
{}

// get a chunk of reads
//
int SequenceDataFile_FASTA_gz::nextChunk(SequenceDataEncoder *output, uint32 max_reads, uint32 max_bps)
{
    const uint32 read_mult =
        ((m_flags & FORWARD)            ? 1u : 0u) +
        ((m_flags & REVERSE)            ? 1u : 0u) +
        ((m_flags & FORWARD_COMPLEMENT) ? 1u : 0u) +
        ((m_flags & REVERSE_COMPLEMENT) ? 1u : 0u);

    // build a writer
    FASTAHandler writer( output, m_flags, m_truncate_read_len );

    return m_fasta_reader.read( max_reads / read_mult, writer );
}

///@} // SequenceIODetail
///@} // SequenceIO
///@} // IO

} // namespace io
} // namespace nvbio
