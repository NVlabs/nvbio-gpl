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

namespace nvbio {

/// \page memory_mapping_page Memory Mapping
///
/// This module implements basic server-client memory mapping functionality
///
/// \section AtAGlanceSection At a Glance
///
/// - MappedFile
/// - ServerMappedFile
///
/// \section MMAPExampleSection Example
///
/// A typical scenario would be to have a server process map a region of its own memory so
/// that other processes can access it:
///\code
/// // server program
/// void main()
/// {
///   // map a block of 100MB
///   ServerMappedFile* mapped_file = new ServerMappedFile();
///   mapped_file->init("my_file", 100*1024*1024, my_buffer);
///
///   // and loop until somebody tell us to stop
///   while (exit_signal() == false) {}
///
///   // delete the mapped_file object, releasing the memory mapping
///   delete mapped_file;
/// }
///\endcode
///
/// and a client read that mapped file:
///\code
/// // client program
/// void main()
/// {
///   // map the shared block in the client's memory space
///   MappedFile mapped_file;
///   void* mapped_buffer = mapped_file.init("my_file", 100*1024*1024);
///
///   // and do something with it
///   do_something( mapped_buffer );
/// }
///\endcode
///
/// \section TechnicalOverviewSection Technical Overview
///
/// See the \ref MemoryMappingModule module documentation.
///

///@addtogroup Basic
///@{

///@defgroup MemoryMappingModule Memory Mapping
/// This module implements basic server-client memory mapping functionality
///@{

///
/// A class to map a memory object into a client process.
/// See ServerMappedFile.
///
struct MappedFile
{
    struct mapping_error
    {
        mapping_error(const char* name, int32 code) : m_file_name( name ), m_code( code ) {}

        const char* m_file_name;
        int32       m_code;
    };
    struct view_error
    {
        view_error(const char* name, uint32 code) : m_file_name( name ), m_code( code ) {}

        const char* m_file_name;
        int32       m_code;
    };

    /// constructor
    ///
    MappedFile();

    /// destructor
    ///
    ~MappedFile();

    /// initialize the memory mapped file
    ///
    void* init(const char* name, const uint64 file_size);

private:
    struct Impl;
    Impl* impl;
};

///
/// A class to create a memory mapped object into a server process. The mapped file is destroyed
/// when the destructor is called.
/// See MappedFile.
///
struct ServerMappedFile
{
    struct mapping_error
    {
        mapping_error(const char* name, int32 code) : m_file_name( name ), m_code( code ) {}

        const char* m_file_name;
        int32       m_code;
    };
    struct view_error
    {
        view_error(const char* name, uint32 code) : m_file_name( name ), m_code( code ) {}

        const char* m_file_name;
        int32       m_code;
    };

    /// constructor
    ///
    ServerMappedFile();

    /// destructor
    ///
    ~ServerMappedFile();

    /// initialize the memory mapped file
    void* init(const char* name, const uint64 file_size, const void* src);

private:
    struct Impl;
    Impl* impl;
};

///@} MemoryMappingModule
///@} Basic

} // namespace nvbio
