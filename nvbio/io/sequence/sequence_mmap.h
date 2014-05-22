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
#include <nvbio/basic/mmap.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>

namespace nvbio {
namespace io {

///@addtogroup IO
///@{

///@addtogroup SequenceIO
///@{

///
/// A system mapped memory server for sequence data
///
struct SequenceDataMMAPServer
{
    /// load a sequence from file
    ///
    /// \param alphabet                 the alphabet to use for encoding
    /// \param file_name                file name
    /// \param mapped_name              memory mapped object name
    bool load(
        const SequenceAlphabet  alphabet,
        const char*             file_name,
        const char*             mapped_name,
        const SequenceFlags     load_flags,
        const QualityEncoding   qualities = Phred33);

    static std::string info_file_name(const char* name);
    static std::string sequence_file_name(const char* name);
    static std::string sequence_index_file_name(const char* name);
    static std::string qual_file_name(const char* name);
    static std::string name_file_name(const char* name);
    static std::string name_index_file_name(const char* name);

    ServerMappedFile m_info_file;                    ///< internal memory-mapped info object server
    ServerMappedFile m_sequence_file;                ///< internal memory-mapped genome object server
    ServerMappedFile m_sequence_index_file;          ///< internal memory-mapped forward occurrence table object server
    ServerMappedFile m_qual_file;                    ///< internal memory-mapped reverse occurrence table object server
    ServerMappedFile m_name_file;                    ///< internal memory-mapped forward BWT object server
    ServerMappedFile m_name_index_file;              ///< internal memory-mapped reverse BWT object server
};

///
/// A concrete SequenceData storage implementation in system mapped memory
///
struct SequenceDataMMAP : public SequenceData
{
    typedef SequenceDataView                  plain_view_type;
    typedef ConstSequenceDataView       const_plain_view_type;

    /// constructor
    ///
    SequenceDataMMAP() :
        m_sequence_ptr( NULL ),
        m_sequence_index_ptr( NULL ),
        m_qual_ptr( NULL ),
        m_name_ptr( NULL ),
        m_name_index_ptr( NULL )
        {}

    /// load from a memory mapped object
    ///
    /// \param name          memory mapped object name
    bool load(
        const char* name);

    /// convert to a plain_view
    ///
    operator plain_view_type()
    {
        return plain_view_type(
            static_cast<const SequenceDataInfo&>( *this ),
            m_sequence_ptr,
            m_sequence_index_ptr,
            m_qual_ptr,
            m_name_ptr,
            m_name_index_ptr );
    }
    /// convert to a const plain_view
    ///
    operator const_plain_view_type() const
    {
        return const_plain_view_type(
            static_cast<const SequenceDataInfo&>( *this ),
            m_sequence_ptr,
            m_sequence_index_ptr,
            m_qual_ptr,
            m_name_ptr,
            m_name_index_ptr );
    }

    uint32* m_sequence_ptr;
    uint32* m_sequence_index_ptr;
    char*   m_qual_ptr;
    char*   m_name_ptr;
    uint32* m_name_index_ptr;

    MappedFile m_info_file;                    ///< internal memory-mapped info object server
    MappedFile m_sequence_file;                ///< internal memory-mapped genome object server
    MappedFile m_sequence_index_file;          ///< internal memory-mapped forward occurrence table object server
    MappedFile m_qual_file;                    ///< internal memory-mapped reverse occurrence table object server
    MappedFile m_name_file;                    ///< internal memory-mapped forward BWT object server
    MappedFile m_name_index_file;              ///< internal memory-mapped reverse BWT object server
};

/// map a sequence file
///
/// \param sequence_file_name   the file to open
///
SequenceDataMMAP* map_sequence_file(const char* sequence_file_name);

///@} // SequenceIO
///@} // IO

} // namespace io
} // namespace nvbio
