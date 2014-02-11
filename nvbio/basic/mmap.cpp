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

#include <nvbio/basic/mmap.h>
#include <nvbio/basic/console.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <string>

#ifdef WIN32

#include <windows.h>
#include <conio.h>

namespace nvbio {

struct MappedFile::Impl
{
    Impl() : h_file( INVALID_HANDLE_VALUE ), buffer( NULL ) {} 

    HANDLE h_file;
    void*  buffer;
};
struct ServerMappedFile::Impl
{
    Impl() : h_file( INVALID_HANDLE_VALUE ), buffer( NULL ) {} 

    HANDLE h_file;
    void*  buffer;
};

MappedFile::MappedFile() : impl( new Impl() ) {}

void* MappedFile::init(const char* name, const uint64 file_size)
{
    std::string sname = std::string("Global\\") + std::string( name );
    std::wstring wname( sname.begin(), sname.end() );

    impl->h_file = OpenFileMapping(
        FILE_MAP_READ,       // read/write access
        FALSE,               // do not inherit the name
#ifdef UNICODE
        wname.c_str()        // name of mapping object
#else
        (LPCSTR)wname.c_str()
#endif
    );

    if (impl->h_file == NULL)
        throw mapping_error( name, GetLastError() );

    impl->buffer = MapViewOfFile(
        impl->h_file,   // handle to map object
        FILE_MAP_READ,  // read/write permission
        0,
        uint32(file_size >> 32),
        uint32(file_size & 0xFFFFFFFFu) );

    if (impl->buffer == NULL)
        throw view_error( name, GetLastError() );

    log_verbose(stderr, "created mapped file object \"%s\" (%.2f %s)\n", name, (file_size > 1024*1024 ? float(file_size)/float(1024*1024) : float(file_size)), (file_size > 1024*1024 ? TEXT("MB") : TEXT("B")));
    return impl->buffer;
}
MappedFile::~MappedFile()
{
    if (impl->buffer != NULL) UnmapViewOfFile( impl->buffer );
    if (impl->h_file != INVALID_HANDLE_VALUE) CloseHandle( impl->h_file );

    delete impl;
}

ServerMappedFile::ServerMappedFile() : impl( new Impl() ) {}

void* ServerMappedFile::init(const char* name, const uint64 file_size, const void* src)
{
    std::string sname = std::string("Global\\") + std::string( name );
    std::wstring wname( sname.begin(), sname.end() );

    impl->h_file = CreateFileMapping(
        INVALID_HANDLE_VALUE,               // use paging file
        NULL,                               // default security
        PAGE_READWRITE,                     // read/write access
        uint32(file_size >> 32),            // maximum object size (high-order DWORD)
        uint32(file_size & 0xFFFFFFFFu),    // maximum object size (low-order DWORD)
#ifdef UNICODE
        wname.c_str()                       // name of mapping object
#else
        (LPCSTR) wname.c_str()
#endif
    );

    if (impl->h_file == NULL)
        throw mapping_error( name, GetLastError() );

    impl->buffer = MapViewOfFile(
        impl->h_file,           // handle to map object
        FILE_MAP_WRITE,         // read/write permission
        0,
        uint32(file_size >> 32),
        uint32(file_size & 0xFFFFFFFFu) );

    if (impl->buffer == NULL)
        throw view_error( name, GetLastError() );

    if (src != NULL)
        CopyMemory( impl->buffer, src, file_size );

    log_verbose(stderr, "created file mapping object \"%s\" (%.2f %s)\n", name, (file_size > 1024*1024 ? float(file_size)/float(1024*1024) : float(file_size)), (file_size > 1024*1024 ? "MB" : "B"));
    return impl->buffer;
}
ServerMappedFile::~ServerMappedFile()
{
    if (impl->buffer != NULL) UnmapViewOfFile( impl->buffer );
    if (impl->h_file != INVALID_HANDLE_VALUE) CloseHandle( impl->h_file );

    delete impl;
}

} // namespace nvbio

#else

//
// POSIX shared memory files
//

#include <sys/mman.h>
#include <sys/stat.h> /* For mode constants */
#include <fcntl.h> /* For O_* constants */ 
#include <errno.h>
#include <unistd.h>
#include <sys/types.h>

namespace nvbio {

struct MappedFile::Impl
{
    Impl() : h_file( -1 ), buffer( NULL ) {} 

    int         h_file;
    void*       buffer;
    std::string file_name;
    uint64      file_size;
};
struct ServerMappedFile::Impl
{
    Impl() : h_file( -1 ), buffer( NULL ) {} 

    int    h_file;
    void*  buffer;
    std::string file_name;
    uint64      file_size;
};

MappedFile::MappedFile() : impl( new Impl() ) {}

void* MappedFile::init(const char* name, const uint64 file_size)
{
    impl->file_name = std::string("/") + std::string(name);
    impl->file_size = file_size;
    impl->h_file = shm_open(
        impl->file_name.c_str(),
        O_RDONLY,
        S_IRWXU );

    if (impl->h_file == -1)
        throw mapping_error( impl->file_name.c_str(), errno );

    ftruncate( impl->h_file, file_size );

    impl->buffer = mmap(
        NULL,
        file_size,
        PROT_READ,
        MAP_SHARED,
        impl->h_file,
        0 );

    if (impl->buffer == NULL)
        throw view_error( impl->file_name.c_str(), errno );

    log_verbose(stderr, "created file mapping object \"%s\" (%.2f %s)\n", name, (file_size > 1024*1024 ? float(file_size)/float(1024*1024) : float(file_size)), (file_size > 1024*1024 ? "MB" : "B"));
    return impl->buffer;
}
MappedFile::~MappedFile()
{
    if (impl->buffer != NULL) munmap( impl->buffer, impl->file_size );
    //if (impl->h_file != -1)   shm_unlink( impl->file_name.c_str() );

    delete impl;
}

ServerMappedFile::ServerMappedFile() : impl( new Impl() ) {}

void* ServerMappedFile::init(const char* name, const uint64 file_size, const void* src)
{
    impl->file_name = std::string("/") + std::string(name);
    impl->file_size = file_size;
    impl->h_file = shm_open(
        impl->file_name.c_str(),
        O_RDWR | O_CREAT,
        S_IRWXU );

    if (impl->h_file == -1)
        throw mapping_error( impl->file_name.c_str(), errno );

    ftruncate( impl->h_file, file_size );

    impl->buffer = mmap(
        NULL,
        file_size,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        impl->h_file,
        0 );

    if (impl->buffer == NULL)
        throw view_error( impl->file_name.c_str(), errno );

    if (src != NULL)
        memcpy( impl->buffer, src, file_size );

    log_verbose(stderr, "created file mapping object \"%s\" (%.2f %s)\n", name, (file_size > 1024*1024 ? float(file_size)/float(1024*1024) : float(file_size)), (file_size > 1024*1024 ? "MB" : "B"));
    return impl->buffer;
}
ServerMappedFile::~ServerMappedFile()
{
    if (impl->buffer != NULL) munmap( impl->buffer, impl->file_size );
    if (impl->h_file != -1)   shm_unlink( impl->file_name.c_str() );

    delete impl;
}

} // namespace nvbio

#endif
