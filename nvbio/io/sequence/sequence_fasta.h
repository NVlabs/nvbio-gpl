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
#include <nvbio/fasta/fasta.h>
#include <nvbio/basic/console.h>

namespace nvbio {
namespace io {

///@addtogroup IO
///@{

///@addtogroup SequenceIO
///@{

///@addtogroup SequenceIODetail
///@{

///
/// loader for FASTA files, possibly gzipped
///
struct SequenceDataFile_FASTA_gz : public SequenceDataFile
{
    /// constructor
    ///
    SequenceDataFile_FASTA_gz(
        const char*             read_file_name,
        const QualityEncoding   qualities,
        const uint32            max_reads,
        const uint32            max_read_len,
        const SequenceEncoding  flags);

    /// get a chunk of reads
    ///
    int nextChunk(SequenceDataEncoder *output, uint32 max_reads, uint32 max_bps);

private:
    FASTA_reader m_fasta_reader; ///< the FASTA file parser
};

///@} // SequenceIODetail
///@} // SequenceIO
///@} // IO

} // namespace io
} // namespace nvbio
