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
#include <nvbio/io/sequence/sequence_mmap.h>
#include <nvbio/basic/bnt.h>
#include <nvbio/basic/console.h>
#include <stdio.h>
#include <stdlib.h>

namespace nvbio {
namespace io {

namespace { // anonymous namespace

template <typename T>
uint64 block_fread(T* dst, const uint64 n, FILE* file)
{
#if defined(WIN32)
    // use blocked reads on Windows, which seems to otherwise become less responsive while reading.
    const uint64 BATCH_SIZE = 16*1024*1024;
    for (uint64 batch_begin = 0; batch_begin < n; batch_begin += BATCH_SIZE)
    {
        const uint64 batch_end = nvbio::min( batch_begin + BATCH_SIZE, n );
        const uint64 batch_size = batch_end - batch_begin;

        const uint64 n_words = fread( dst + batch_begin, sizeof(T), batch_size, file );
        if (n_words != batch_size)
            return batch_begin + n_words;
    }
    return n;
#else
    return fread( dst, sizeof(T), n, file );
#endif
}

template <SequenceAlphabet ALPHABET>
bool load_pac(
    const char*     file_name,
    uint32*         stream,
    uint32          seq_length,
    uint32          seq_words)
{
    FILE* file = fopen( file_name, "rb" );
    if (file == NULL)
    {
        log_warning(stderr, "unable to open %s\n", file_name);
        return false;
    }

    const bool wpac = (strcmp( file_name + strlen( file_name ) - 5u, ".wpac" ) == 0);

    typedef SequenceDataTraits<ALPHABET> sequence_traits;

    typedef PackedStream<
        uint32*,uint8,
        sequence_traits::SEQUENCE_BITS,
        sequence_traits::SEQUENCE_BIG_ENDIAN>   output_stream_type;

    if (wpac)
    {
        // read a .wpac file
        uint64 field;
        if (!fread( &field, sizeof(field), 1, file ))
        {
            log_error(stderr, "failed reading %s\n", file_name);
            return false;
        }

        const uint32 _seq_length = uint32(field);
        if (_seq_length != seq_length)
        {
            log_error(stderr, "mismatching sequence lengths in %s, expected: %u, found: %u\n", file_name, seq_length, _seq_length);
            return false;
        }

        if (ALPHABET == 2 && sequence_traits::SEQUENCE_BIG_ENDIAN == true)
        {
            // read the 2-bit per symbol words in the final destination
            const uint32 n_words = (uint32)block_fread( stream, seq_words, file );
            if (n_words != seq_words)
            {
                log_error(stderr, "failed reading %s\n", file_name);
                return false;
            }
        }
        else
        {
            // read the 2-bit per symbol words in a temporary array
            const uint32 pac_words = uint32( (seq_length + 15u)/16u );

            std::vector<uint32> pac_vec( pac_words );
            uint32* pac_stream = &pac_vec[0];

            const uint32 n_words = (uint32)block_fread( pac_stream, pac_words, file );
            if (n_words != pac_words)
            {
                log_error(stderr, "failed reading %s\n", file_name);
                return false;
            }

            // copy the pac stream into the output
            typedef PackedStream<uint32*,uint8,2,true> pac_stream_type;
            pac_stream_type pac( pac_stream );

            output_stream_type out( stream );
            assign( seq_length, pac, out );
        }
    }
    else
    {
        // read a .pac file
        fseek( file, -1, SEEK_END );
        const uint32 packed_file_len = ftell( file );
        uint8 last_byte_len;
        if (!fread( &last_byte_len, sizeof(unsigned char), 1, file ))
        {
            log_error(stderr, "failed reading %s\n", file_name);
            return false;
        }
        const uint32 _seq_length = (packed_file_len - 1u) * 4u + last_byte_len;
        if (_seq_length != seq_length)
        {
            log_error(stderr, "mismatching sequence lengths in %s, expected: %u, found: %u\n", file_name, seq_length, _seq_length);
            return false;
        }

        fseek( file, 0, SEEK_SET );

        const uint32 seq_bytes = uint32( (seq_length + 3u)/4u );

        std::vector<uint8> pac_vec( seq_bytes );
        uint8* pac_stream = &pac_vec[0];

        const uint64 n_bytes = block_fread( pac_stream, seq_bytes, file );
        if (n_bytes != seq_bytes)
        {
            log_error(stderr, "failed reading %s\n", file_name);
            return false;
        }

        // copy the pac stream into the output
        typedef PackedStream<uint8*,uint8,2,true> pac_stream_type;
        pac_stream_type pac( pac_stream );

        output_stream_type out( stream );
        assign( seq_length, pac, out );
    }
    fclose( file );
    return true;
}

struct BNTLoader : public nvbio::BNTSeqLoader
{
    typedef nvbio::vector<host_tag,uint32> IndexVector;
    typedef nvbio::vector<host_tag,char>   StringVector;

    BNTLoader(
        IndexVector&  index_vec,
        IndexVector&  name_index_vec,
        StringVector& name_vec) :
        m_name_vec( &name_vec ),
        m_name_index_vec( &name_index_vec ),
        m_index_vec( &index_vec ) {}

    void set_info(const nvbio::BNTInfo info)
    {
        m_info = info;
    }
    void read_ann(const nvbio::BNTAnnInfo& info, nvbio::BNTAnnData& data)
    {
        const uint32 name_offset = (uint32)m_name_vec->size();
        m_name_vec->resize( name_offset + info.name.length() + 1u );
        strcpy( &m_name_vec->front() + name_offset, info.name.c_str() );

        m_name_index_vec->push_back( name_offset );
        m_index_vec->push_back( uint32( data.offset ) );
    }

    void read_amb(const nvbio::BNTAmb& amb)
    {
    }

    nvbio::BNTInfo  m_info;
    StringVector*   m_name_vec;
    IndexVector*    m_name_index_vec;
    IndexVector*    m_index_vec;
};

} // anonymous namespace

// load a sequence file
//
// \param sequence_file_name   the file to open
// \param qualities            the encoding of the qualities
// \param max_seqs             maximum number of reads to input
// \param max_sequence_len     maximum read length - reads will be truncated
// \param flags                a set of flags indicating which strands to encode
//                             in the batch for each read.
//                             For example, passing FORWARD | REVERSE_COMPLEMENT
//                             will result in a stream containing BOTH the forward
//                             and reverse-complemented strands.
//
bool load_pac(
    const SequenceAlphabet      alphabet,
    SequenceDataHost*           sequence_data,
    const char*                 sequence_file_name,
    const SequenceFlags         load_flags,
    const QualityEncoding       qualities)
{
    std::string prefix = sequence_file_name;
    const size_t s = prefix.rfind( "." );
    if (s != std::string::npos)
        prefix[s] = '\0';

    // prepare the sequence index
    sequence_data->m_sequence_index_vec.resize( 1 );
    sequence_data->m_sequence_index_vec[0] = 0;

    // prepare the name index
    sequence_data->m_name_index_vec.resize( 1 );
    sequence_data->m_name_index_vec[0] = 0;

    BNTInfo bnt_info;
    load_bns_info( bnt_info, prefix.c_str() );

    BNTLoader loader( sequence_data->m_sequence_index_vec, sequence_data->m_name_index_vec, sequence_data->m_name_vec );
    load_bns( &loader, prefix.c_str() );

    const uint32 bits             = bits_per_symbol( alphabet );
    const uint32 symbols_per_word = 32 / bits;

    const uint32 seq_length         = uint32( bnt_info.l_pac );
    const uint32 seq_words          = uint32( util::divide_ri( seq_length, symbols_per_word ) );
    const uint32 aligned_seq_words  = align<4>( seq_words );

    // set all basic info
    sequence_data->m_alphabet               = alphabet;
    sequence_data->m_n_seqs                 = bnt_info.n_seqs;
    sequence_data->m_sequence_stream_len    = seq_length;
    sequence_data->m_sequence_stream_words  = aligned_seq_words;
    sequence_data->m_name_stream_len        = uint32( sequence_data->m_name_vec.size() );
    sequence_data->m_has_qualities          = false;
    // TODO: m_{avg,min,max}_sequence_len

    // alloc sequence storage
    sequence_data->m_sequence_vec.resize( sequence_data->m_sequence_stream_words );

    // initialize the alignment slack
    for (uint32 i = seq_words; i < aligned_seq_words; ++i)
        sequence_data->m_sequence_vec[i] = 0u;

    switch (alphabet)
    {
    case DNA:
        return load_pac<DNA>( sequence_file_name, &sequence_data->m_sequence_vec[0], seq_length, seq_words );
        break;
    case DNA_N:
        return load_pac<DNA>( sequence_file_name, &sequence_data->m_sequence_vec[0], seq_length, seq_words );
        break;
    case PROTEIN:
        return load_pac<PROTEIN>( sequence_file_name, &sequence_data->m_sequence_vec[0], seq_length, seq_words );
        break;
    }
    return false;
}

// load a sequence file
//
// \param sequence_file_name   the file to open
// \param qualities            the encoding of the qualities
// \param max_seqs             maximum number of reads to input
// \param max_sequence_len     maximum read length - reads will be truncated
// \param flags                a set of flags indicating which strands to encode
//                             in the batch for each read.
//                             For example, passing FORWARD | REVERSE_COMPLEMENT
//                             will result in a stream containing BOTH the forward
//                             and reverse-complemented strands.
//
bool load_pac(
    const SequenceAlphabet      alphabet,
    SequenceDataMMAPServer*     sequence_data,
    const char*                 sequence_file_name,
    const char*                 mapped_name,
    const SequenceFlags         load_flags,
    const QualityEncoding       qualities)
{
    std::string prefix = sequence_file_name;
    const size_t s = prefix.rfind( "." );
    if (s != std::string::npos)
        prefix[s] = '\0';

    std::string info_name           = std::string("nvbio.") + std::string( mapped_name ) + ".seq_info";
    std::string sequence_name       = std::string("nvbio.") + std::string( mapped_name ) + ".seq";
    std::string sequence_index_name = std::string("nvbio.") + std::string( mapped_name ) + ".seq_index";
    std::string name_name           = std::string("nvbio.") + std::string( mapped_name ) + ".name";
    std::string name_index_name     = std::string("nvbio.") + std::string( mapped_name ) + ".name_index";

    // prepare the sequence index
    nvbio::vector<host_tag,uint32> sequence_index_vec;
    nvbio::vector<host_tag,uint32> name_index_vec;
    nvbio::vector<host_tag,char>   name_vec;

    sequence_index_vec.resize( 1 );
    sequence_index_vec[0] = 0;

    // prepare the name index
    name_index_vec.resize( 1 );
    name_index_vec[0] = 0;

    BNTInfo bnt_info;
    load_bns_info( bnt_info, prefix.c_str() );

    BNTLoader loader( sequence_index_vec, name_index_vec, name_vec );
    load_bns( &loader, prefix.c_str() );

    const uint32 bits             = bits_per_symbol( alphabet );
    const uint32 symbols_per_word = 32 / bits;

    const uint32 n_seqs             = bnt_info.n_seqs;
    const uint32 seq_length         = uint32( bnt_info.l_pac );
    const uint32 seq_words          = uint32( util::divide_ri( seq_length, symbols_per_word ) );
    const uint32 aligned_seq_words  = align<4>( seq_words );

    // setup all basic info
    SequenceDataInfo info;
    info.m_alphabet                 = alphabet;
    info.m_n_seqs                   = n_seqs;
    info.m_sequence_stream_len      = seq_length;
    info.m_sequence_stream_words    = aligned_seq_words;
    info.m_name_stream_len          = uint32( name_vec.size() );
    info.m_has_qualities            = false;
    // TODO: m_{avg,min,max}_sequence_len

    // alloc sequence storage
    uint32* sequence_ptr = (uint32*)sequence_data->m_sequence_file.init(
        sequence_name.c_str(),
        aligned_seq_words * sizeof(uint32),
        NULL );

    // initialize the alignment slack
    for (uint32 i = seq_words; i < aligned_seq_words; ++i)
        sequence_ptr[i] = 0u;

    // alloc sequence_index storage
    uint32* sequence_index_ptr = (uint32*)sequence_data->m_sequence_index_file.init(
        sequence_index_name.c_str(),
        sequence_index_vec.size() * sizeof(uint32),
        NULL );

    // alloc name_index storage
    uint32* name_index_ptr = (uint32*)sequence_data->m_name_index_file.init(
        name_index_name.c_str(),
        name_index_vec.size() * sizeof(uint32),
        NULL );

    // alloc name storage
    char* name_ptr = (char*)sequence_data->m_name_file.init(
        name_name.c_str(),
        name_vec.size() * sizeof(char),
        NULL );

    // alloc info storage
    SequenceDataInfo* info_ptr = (SequenceDataInfo*)sequence_data->m_info_file.init(
        info_name.c_str(),
        sizeof(SequenceDataInfo),
        NULL );

    // copy the loaded index and names vectors
    memcpy( sequence_index_ptr, &sequence_index_vec[0], sequence_index_vec.size() * sizeof(uint32) );
    memcpy( name_index_ptr,     &name_index_vec[0],     name_index_vec.size()     * sizeof(uint32) );
    memcpy( name_ptr,           &name_vec[0],           name_vec.size()           * sizeof(char) );

    *info_ptr = info;

    // load the actual sequence
    switch (alphabet)
    {
    case DNA:
        return load_pac<DNA>( sequence_file_name, sequence_ptr, seq_length, seq_words );
        break;
    case DNA_N:
        return load_pac<DNA>( sequence_file_name, sequence_ptr, seq_length, seq_words );
        break;
    case PROTEIN:
        return load_pac<PROTEIN>( sequence_file_name, sequence_ptr, seq_length, seq_words );
        break;
    }
    return false;
}

///@} // SequenceIO
///@} // IO

} // namespace io
} // namespace nvbio
