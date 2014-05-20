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

namespace nvbio {
namespace io {

/// load a sequence file
///
/// \param sequence_file_name   the file to open
/// \param qualities            the encoding of the qualities
/// \param max_seqs             maximum number of reads to input
/// \param max_sequence_len     maximum read length - reads will be truncated
/// \param flags                a set of flags indicating which strands to encode
///                             in the batch for each read.
///                             For example, passing FORWARD | REVERSE_COMPLEMENT
///                             will result in a stream containing BOTH the forward
///                             and reverse-complemented strands.
///
bool load_pac(
    const SequenceAlphabet      alphabet,
    SequenceDataHost*           sequence_data,
    const char*                 prefix,
    const SequenceFlags         load_flags,
    const QualityEncoding       qualities);

/// load a sequence file
///
/// \param sequence_file_name   the file to open
/// \param qualities            the encoding of the qualities
/// \param max_seqs             maximum number of reads to input
/// \param max_sequence_len     maximum read length - reads will be truncated
/// \param flags                a set of flags indicating which strands to encode
///                             in the batch for each read.
///                             For example, passing FORWARD | REVERSE_COMPLEMENT
///                             will result in a stream containing BOTH the forward
///                             and reverse-complemented strands.
///
bool load_pac(
    const SequenceAlphabet      alphabet,
    SequenceDataMMAPServer*     sequence_data,
    const char*                 prefix,
    const char*                 mapped_name,
    const SequenceFlags         load_flags,
    const QualityEncoding       qualities);

///@} // SequenceIO
///@} // IO

} // namespace io
} // namespace nvbio
