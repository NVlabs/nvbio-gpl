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
#include <nvbio/basic/console.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>

namespace nvbio {

#define FASTQ_EOF 255u

///\page fastq_page FASTQ Parsers
///\htmlonly
/// <img src="nvidia_cubes.png" style="position:relative; bottom:-10px; border:0px;"/>
///\endhtmlonly
///
///\n
/// This module contains bare bones configurable parsers and accessory classes for FASTQ files.
///
///\section AtAGlanceSection At a Glance
///
/// - FASTQ_reader
/// - FASTQ_file
/// - FASTQ_gzfile
///
///\section TechnicalDetailsSection Technical Details
///
/// For technical details, see the \ref FASTQModule module documentation
///

///\defgroup FASTQModule FASTQ
///
/// This module contains bare bones configurable parsers for FASTQ files
///

///@addtogroup FASTQModule
///@{

///
/// A gzipped FASTQ file implementing the \ref FASTQ_stream model needed by FASTQ_reader
///
struct FASTQ_gzfile
{
    /// constructor
    ///
    FASTQ_gzfile(const uint32 buffer_size = 64536u);

    /// constructor
    ///
    FASTQ_gzfile(const char* filename, const uint32 buffer_size = 64536u);

    /// destructor
    ///
    ~FASTQ_gzfile();

    /// open a new file
    ///
    void open(const char* filename);

    /// return whether the file is valid
    ///
    bool valid() const;

    /// get the next character, or 255 if EOF
    ///
    uint8 get();

    /// read a buffer
    ///
    uint32 read(uint8* buffer, const uint32 n);

    struct Impl;

private:
    Impl*              m_impl;
    std::vector<uint8> m_buffer;
    uint32             m_buffer_size;
    uint32             m_buffer_pos;
};

///
/// A FASTQ file implementing the \ref FASTQ_stream model needed by FASTQ_reader
///
struct FASTQ_file
{
    /// constructor
    ///
    FASTQ_file(const uint32 buffer_size = 64536u);
        
    /// constructor
    ///
    FASTQ_file(const char* filename, const uint32 buffer_size = 64536u);

    /// open a new file
    ///
    void open(const char* filename);

    /// destructor
    ///
    ~FASTQ_file();

    /// return whether the file is valid
    ///
    bool valid() const { return m_file != NULL; }

    /// get the next character, or 255 if EOF
    ///
    uint8 get();

private:
    FILE*              m_file;
    std::vector<uint8> m_buffer;
    uint32             m_buffer_size;
    uint32             m_buffer_pos;
};

///
/// A simple FASTQ_reader, templated over an input stream class
///
/// \tparam FASTQ_stream        the templated input stream, which must
///                             provide a single method:
///
/// \anchor FASTQ_stream
/// \code
/// struct FASTQ_stream
/// {
///     // consume and return the next character
///     uint8 get();
/// }
/// \endcode
///
///
template <typename FASTQ_stream>
struct FASTQ_reader
{
    /// constructor
    ///
    FASTQ_reader(FASTQ_stream& stream);

    /// destructor
    ///
    ~FASTQ_reader() {}

    /// read a batch of bp reads
    ///
    /// \tparam Writer          the output FASTQ writer, which must implement
    ///                         the following interface
    /// \code
    /// struct Writer
    /// {
    ///     void push_back(
    ///         const uint32    read_length,
    ///         const char*     read_name,
    ///         const uint8*    bps,
    ///         const uint8*    qualities);
    /// }
    /// \endcode
    ///
    template <typename Writer>
    uint32 read(const uint32 n_reads, Writer& writer);

    /// get the next character, or 255 if EOF
    ///
    uint8 get() { return m_stream->get(); };

    /// return the error string
    ///
    void error_string(char* error);

private:
    FASTQ_stream* m_stream;

    std::vector<char>  m_name;
    std::vector<uint8> m_read_bp;
    std::vector<uint8> m_read_q;

    uint32             m_error;
    char               m_error_char;
    uint32             m_line;
};

///
/// A helper buffer class
///
struct FASTQ_buffer
{
    /// constructor
    ///
    FASTQ_buffer(const uint32 buffer_size, const char* buffer) :
        m_buffer( buffer ),
        m_buffer_size( buffer_size ),
        m_buffer_pos( 0 ) {}

    /// get the next character, or 255 if EOF
    ///
    uint8 get()
    {
        return (m_buffer_pos < m_buffer_size) ? m_buffer[ m_buffer_pos++ ] : FASTQ_EOF;
    };

    const char* m_buffer;
    uint32      m_buffer_size;
    uint32      m_buffer_pos;
};

///@} // FASTQModule

} // namespace nvbio

#include <nvbio/fastq/fastq_inl.h>
