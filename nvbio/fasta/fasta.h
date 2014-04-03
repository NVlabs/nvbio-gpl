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

#include <nvbio/basic/types.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <zlib/zlib.h>

namespace nvbio {

///\page fasta_page FASTA Parsers
///\htmlonly
/// <img src="nvidia_cubes.png" style="position:relative; bottom:-10px; border:0px;"/>
///\endhtmlonly
///
///\n
/// This module contains bare bones configurable parsers and accessory classes for FASTA files.
///
///\section AtAGlanceSection At a Glance
///
/// - FASTA_reader
/// - FASTA_inc_reader
///
///\section TechnicalDetailsSection Technical Details
///
/// For technical details, see the \ref FASTAModule module documentation
///

///\defgroup FASTAModule FASTA
///
/// This module contains bare bones configurable parsers for FASTA files
///

///@addtogroup FASTAModule
///@{

///
/// An incremental FASTA reader, which parses the reads incrementally
/// without ever storing them internally, and providing them to the
/// output handler base by base.
///
struct FASTA_inc_reader
{
    /// constructor
    ///
    FASTA_inc_reader(const char* filename, const uint32 buffer_size = 64536u);

    /// destructor
    ///
    ~FASTA_inc_reader();

    /// return whether the file is valid
    ///
    bool valid() const { return m_file != NULL; }

    /// read a batch of bp reads
    ///
    /// \tparam Writer          an output handler class, which must
    ///                         implement the following interface:
    ///
    /// \code
    /// struct Writer
    /// {
    ///     // called before starting to parse a new read
    ///     void begin_read();
    ///
    ///     // called upon completion of a single read
    ///     void end_read();
    ///
    ///     // provide the next character of the read id
    ///     void id(const char c);
    ///
    ///     // provide the next base of the read
    ///     void read(const char c);
    /// }
    /// \endcode
    ///
    template <typename Writer>
    uint32 read(const uint32 n_reads, Writer& writer);

    /// get the next character, or 255 if EOF
    ///
    uint8 get();

private:
    gzFile m_file;

    std::vector<uint8> m_buffer;
    uint32             m_buffer_size;
    uint32             m_buffer_pos;
};

///
/// A non-incremental FASTA reader, which parses the reads atomically
/// and provides them to the output handler one by one.
///
struct FASTA_reader
{
    /// constructor
    ///
    FASTA_reader(const char* filename, const uint32 buffer_size = 64536u);

    /// destructor
    ///
    ~FASTA_reader();

    /// return whether the file is valid
    ///
    bool valid() const { return m_file != NULL; }

    /// read a batch of bp reads
    ///
    /// \tparam Writer          an output handler class, which must
    ///                         implement the following interface:
    ///
    /// \code
    /// struct Writer
    /// {
    ///     // called whenever a new read has been parsed
    ///     void push_back(
    ///         const char*     id,
    ///         const uint32    read_length,
    ///         const uint8*    read);
    /// }
    /// \endcode
    ///
    template <typename Writer>
    uint32 read(const uint32 n_reads, Writer& writer);

    /// get the next character, or 255 if EOF
    ///
    uint8 get();

private:
    gzFile m_file;

    std::vector<char>  m_id;
    std::vector<uint8> m_read;
    std::vector<uint8> m_buffer;
    uint32             m_buffer_size;
    uint32             m_buffer_pos;
};

///@} // FASTAModule

} // namespace nvbio

#include <nvbio/fasta/fasta_inl.h>
