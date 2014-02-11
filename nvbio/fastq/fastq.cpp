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

#include <nvbio/fastq/fastq.h>
#include <zlib/zlib.h>


namespace nvbio {

struct FASTQ_gzfile::Impl
{
    Impl(const char* filename, const uint32 buffer_size)
    {
        m_file = gzopen( filename, "r" );

        if (m_file)
            gzbuffer( m_file, buffer_size );
    }

    ~Impl()
    {
        if (m_file)
            gzclose( m_file );
    }

    gzFile m_file;
};

// constructor
//
FASTQ_gzfile::FASTQ_gzfile(const uint32 buffer_size) :
    m_buffer( buffer_size ),
    m_buffer_size( 0 ),
    m_buffer_pos( 0 )
{
    m_impl = NULL;
}

// constructor
//
FASTQ_gzfile::FASTQ_gzfile(const char* filename, const uint32 buffer_size) :
    m_buffer( buffer_size ),
    m_buffer_size( 0 ),
    m_buffer_pos( 0 )
{
    m_impl = new Impl( filename, buffer_size );
}

// destructor
//
FASTQ_gzfile::~FASTQ_gzfile() { delete m_impl; }

// open a new file
//
void FASTQ_gzfile::open(const char* filename)
{
    if (m_impl)
        delete m_impl;

    m_impl = new Impl( filename, uint32( m_buffer.size() ) );
}

// return whether the file is valid
//
bool FASTQ_gzfile::valid() const
{
    return m_impl->m_file != NULL;
}

// read a buffer
//
uint32 FASTQ_gzfile::read(uint8* buffer, const uint32 n)
{
    return gzread( m_impl->m_file, buffer, n );
}

} // namespace nvbio
