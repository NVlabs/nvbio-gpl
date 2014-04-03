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

namespace nvbio {

// constructor
//
inline
FASTA_inc_reader::FASTA_inc_reader(const char* filename, const uint32 buffer_size)
    : m_buffer( buffer_size ), m_buffer_size( 0 ), m_buffer_pos( 0 )
{
    m_file = gzopen( filename, "r" );
    if (m_file)
        gzbuffer( m_file, buffer_size );
}
// destructor
//
inline
FASTA_inc_reader::~FASTA_inc_reader()
{
    if (m_file)
        gzclose( m_file );
}

// read a batch of bp reads
//
// \tparam Writer          an output handler class, which must
//                         implement the following interface:
//
// \code
// struct Writer
// {
//     // called before starting to parse a new read
//     void begin_read();
//
//     // called upon completion of a single read
//     void end_read();
//
//     // provide the next character of the read id
//     void id(const char c);
//
//     // provide the next base of the read
//     void read(const char c);
// }
// \endcode
//
template <typename Writer>
uint32 FASTA_inc_reader::read(const uint32 n_reads, Writer& writer)
{
    uint32 n = 0;
    uint8 c;
    bool  read_sequence = false;

    writer.begin_read();

    while ((c = get()) != 255)
    {
        // start of a new sequence?
        if (c == '>')
        {
            // end the previous one
            if (read_sequence)
            {
                writer.end_read();
                writer.begin_read();

                if (n == n_reads)
                    return n;
            }

            n++;
            
            // read the id
            for (c = get(); c != ' ' && c != '\n'; c = get())
                writer.id( c );
            writer.id( '\0' );

            // read the rest of the line
            while (c != '\n') { c = get(); }

            read_sequence = true;
        }

        // save non-trivial characters into the read
        if (read_sequence && c != '\n' && c != ' ')
            writer.read( c );
    }
    // output the last sequence we've been reading
    writer.end_read();

    return n;
}

// get the next character, or 255 if EOF
//
inline
uint8 FASTA_inc_reader::get()
{
    if (m_buffer_pos >= m_buffer_size)
    {
        m_buffer_size = uint32( gzread( m_file, &m_buffer[0], m_buffer.size() ) );
        m_buffer_pos  = 0;
    }
    return (m_buffer_pos < m_buffer_size) ? m_buffer[ m_buffer_pos++ ] : 255u;
};

// constructor
//
inline
FASTA_reader::FASTA_reader(const char* filename, const uint32 buffer_size)
    : m_buffer( buffer_size ), m_buffer_size( 0 ), m_buffer_pos( 0 )
{
    m_file = gzopen( filename, "r" );
    if (m_file)
        gzbuffer( m_file, buffer_size );
}

// destructor
//
inline
FASTA_reader::~FASTA_reader()
{
    if (m_file)
        gzclose( m_file );
}

// read a batch of bp reads
//
// \tparam Writer          an output handler class, which must
//                         implement the following interface:
//
// \code
// struct Writer
// {
//     // called whenever a new read has been parsed
//     void push_back(
//         const char*     id,
//         const uint32    read_length,
//         const uint8*    read);
// }
// \endcode
//
template <typename Writer>
uint32 FASTA_reader::read(const uint32 n_reads, Writer& writer)
{
    uint32 n = 0;
    uint8 c;
    bool  read_sequence = false;

    while ((c = get()) != 255)
    {
        // start of a new sequence?
        if (c == '>')
        {
            // output the previous one
            if (read_sequence)
            {
                writer.push_back(
                    &m_id[0],
                    uint32( m_read.size() ),
                    &m_read[0] );

                if (n == n_reads)
                    return n;
            }

            n++;
            
            // read the id
            m_id.erase( m_id.begin(), m_id.end() );
            for (c = get(); c != ' ' && c != '\n'; c = get())
                m_id.push_back( char(c) );

            m_id.push_back('\0');

            // read the rest of the line
            while (c != '\n') { c = get(); }

            // reset the read
            m_read.erase( m_read.begin(), m_read.end() );

            read_sequence = true;
        }

        // save non-trivial characters into the read
        if (read_sequence && c != '\n' && c != ' ')
            m_read.push_back( c );
    }
    // output the last sequence we've been reading
    if (read_sequence)
    {
        writer.push_back(
            &m_id[0],
            uint32( m_read.size() ),
            &m_read[0] );
    }
    return n;
}

// get the next character, or 255 if EOF
//
inline
uint8 FASTA_reader::get()
{
    if (m_buffer_pos >= m_buffer_size)
    {
        m_buffer_size = uint32( gzread( m_file, &m_buffer[0], m_buffer.size() ) );
        m_buffer_pos  = 0;
    }
    return (m_buffer_pos < m_buffer_size) ? m_buffer[ m_buffer_pos++ ] : 255u;
}

} // namespace nvbio
