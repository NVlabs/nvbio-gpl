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

#include <nvbio/io/sequence/sequence_mmap.h>
#include <nvbio/basic/console.h>

namespace nvbio {
namespace io {

// load from a memory mapped object
//
// \param name          memory mapped object name
int SequenceDataMMAP::load(const char* file_name)
{
    log_visible(stderr, "SequenceData (MMAP) : loading... started\n");
    log_visible(stderr, "  file : %s\n", file_name);

    std::string infoName        = std::string("nvbio.") + std::string( file_name ) + ".seq_info";
    std::string seqName         = std::string("nvbio.") + std::string( file_name ) + ".seq";
    std::string seqIndexName    = std::string("nvbio.") + std::string( file_name ) + ".seq_index";
    std::string qualName        = std::string("nvbio.") + std::string( file_name ) + ".qual";
    std::string nameName        = std::string("nvbio.") + std::string( file_name ) + ".name";
    std::string nameIndexName   = std::string("nvbio.") + std::string( file_name ) + ".name_index";

    try {
        const SequenceDataInfo* info = (const SequenceDataInfo*)m_info_file.init( infoName.c_str(), sizeof(SequenceDataInfo) );

        this->SequenceDataInfo::operator=( *info );

        const uint64 index_file_size    = info->size()  * sizeof(uint32);
        const uint64 seq_file_size      = info->words() * sizeof(uint32);
        const uint64 qual_file_size     = info->qs()    * sizeof(char);
        const uint64 name_file_size     = info->m_name_stream_len * sizeof(char);

        m_sequence_ptr        = (uint32*)m_sequence_file.init( seqName.c_str(), seq_file_size );
        m_sequence_index_ptr  = (uint32*)m_sequence_index_file.init( seqIndexName.c_str(), index_file_size );
        m_qual_ptr            = qual_file_size ? (char*)m_qual_file.init( qualName.c_str(), qual_file_size ) : NULL;
        m_name_ptr            =   (char*)m_name_file.init( nameName.c_str(), name_file_size );
        m_name_index_ptr      = (uint32*)m_sequence_index_file.init( nameIndexName.c_str(), index_file_size );
    }
    catch (MappedFile::mapping_error error)
    {
        log_error(stderr, "SequenceDataMMAP: error mapping file \"%s\" (%d)!\n", error.m_file_name, error.m_code);
        return 0;
    }
    catch (MappedFile::view_error error)
    {
        log_error(stderr, "SequenceDataMMAP: error viewing file \"%s\" (%d)!\n", error.m_file_name, error.m_code);
        return 0;
    }
    catch (...)
    {
        log_error(stderr, "SequenceDataMMAP: error mapping file (unknown)!\n");
        return 0;
    }

    log_visible(stderr, "SequenceData (MMAP) : loading... done\n");
    return 1;
}

} // namespace io
} // namespace nvbio
