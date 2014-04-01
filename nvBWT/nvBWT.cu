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

// nvBWT.cu
//

#define NVBIO_CUDA_DEBUG

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <algorithm>
#include <crc/crc.h>
#include <nvbio/basic/console.h>
#include <nvbio/basic/bnt.h>
#include <nvbio/basic/numbers.h>
#include <nvbio/basic/timer.h>
#include <nvbio/basic/packedstream.h>
#include <nvbio/basic/thrust_view.h>
#include <nvbio/fmindex/dna.h>
#include <nvbio/fmindex/bwt.h>
#include <nvbio/fasta/fasta.h>
#include <nvbio/io/fmi.h>
#include <nvbio/sufsort/sufsort.h>
#include "filelist.h"

using namespace nvbio;

unsigned char nst_nt4_table[256] = {
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 5 /*'-'*/, 4, 4,
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  3, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 0, 4, 1,  4, 4, 4, 2,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  3, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4, 
	4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4,  4, 4, 4, 4
};


#define RAND    0
#define RAND48  1

#if (GENERATOR == RAND) || ((GENERATOR == RAND48) && defined(WIN32))

// generate random base pairs using rand()
inline void  srand_bp(const unsigned int s) { srand(s); }
inline float frand() { return float(rand()) / float(RAND_MAX); }
inline uint8 rand_bp() { return uint8( frand() * 4 ); }

#elif (GENERATOR == RAND48)

// generate random base pairs using rand48()
inline void  srand_bp(const unsigned int s) { srand48(s); }
inline uint8 rand_bp() { return uint8( drand48() * 4 ); }

#endif

struct Counter
{
    Counter() : m_size(0), m_reads(0) {}

    void begin_read() { m_reads++; }
    void end_read() {}

    void id(const uint8 c) {}
    void read(const uint8 c) { m_size++; }

    uint64 m_size;
    uint32 m_reads;
};

struct Writer
{
    typedef PackedStream<uint32*,uint8,2,true,uint64> stream_type;

    Writer(uint32* storage, const uint32 reads, const uint64 max_size) :
        m_max_size(max_size), m_size(0), m_stream( storage )
    {
        m_bntseq.seed = 11;
        m_bntseq.anns_data.resize( reads );
        m_bntseq.anns_info.resize( reads );

        srand_bp( m_bntseq.seed );
    }

    void begin_read()
    {
        BNTAnnData& ann_data = m_bntseq.anns_data[ m_bntseq.n_seqs ];
        ann_data.len    = 0;
        ann_data.gi     = 0;
        ann_data.offset = m_size;
        ann_data.n_ambs = 0;

        BNTAnnInfo& ann_info = m_bntseq.anns_info[ m_bntseq.n_seqs ];
        ann_info.anno   = "null";

        m_lasts = 0;
    }
    void end_read()
    {
        m_bntseq.n_seqs++;
    }

    void id(const uint8 c)
    {
        m_bntseq.anns_info[ m_bntseq.n_seqs ].name.push_back(char(c));
    }
    void read(const uint8 s)
    {
        if (m_size < m_max_size)
        {
            const uint8 c = nst_nt4_table[s];

            m_stream[ m_size ] = c < 4 ? c : rand_bp();

            if (c >= 4) // we have an N
            {
                if (m_lasts == s) // contiguous N
                {
                    // increment length of the last hole
                    ++m_bntseq.ambs.back().len;
                }
                else
                {
                    // beginning of a new hole
                    BNTAmb amb;
                    amb.len    = 1;
                    amb.offset = m_size;
                    amb.amb    = s;

                    m_bntseq.ambs.push_back( amb );

                    ++m_bntseq.anns_data[ m_bntseq.n_seqs ].n_ambs;
                    ++m_bntseq.n_holes;
                }
            }
            // save last symbol
            m_lasts = s;

            // update sequence length
            BNTAnnData& ann_data = m_bntseq.anns_data[ m_bntseq.n_seqs ];
            ann_data.len++;
        }

        m_bntseq.l_pac++;

        m_size++;
    }

    uint64      m_max_size;
    uint64      m_size;
    stream_type m_stream;

    BNTSeq      m_bntseq;
    uint8       m_lasts;
};

template <typename StreamType>
bool save_stream(FILE* output_file, const uint64 seq_words, const StreamType* stream)
{
    for (uint64 words = 0; words < seq_words; words += 1024)
    {
        const uint32 n_words = (uint32)nvbio::min( uint64(1024u), uint64(seq_words - words) );
        if (fwrite( stream + words, sizeof(StreamType), n_words, output_file ) != n_words)
            return false;
    }
    return true;
}

int build(
    const char*  input_name,
    const char*  output_name,
    const char*  pac_name,
    const char*  rpac_name,
    const char*  bwt_name,
    const char*  rbwt_name,
    const char*  sa_name,
    const char*  rsa_name,
    const uint64 max_length)
{
    std::vector<std::string> sortednames;
    list_files(input_name, sortednames);

    uint32 n_inputs = (uint32)sortednames.size();
    log_info(stderr, "\ncounting bps... started\n");
    // count entire sequence length
    Counter counter;

    for (uint32 i = 0; i < n_inputs; ++i)
    {
        log_info(stderr, "  counting \"%s\"\n", sortednames[i].c_str());

        FASTA_inc_reader fasta( sortednames[i].c_str() );
        if (fasta.valid() == false)
        {
            log_error(stderr, "  unable to open file\n");
            exit(1);
        }

        while (fasta.read( 1024, counter ) == 1024);
    }
    log_info(stderr, "counting bps... done\n");

    const uint64 seq_length   = nvbio::min( (uint64)counter.m_size, (uint64)max_length );
    const uint32 bps_per_word = sizeof(uint32)*4u;
    const uint64 seq_words    = (seq_length + bps_per_word - 1u) / bps_per_word;

    log_info(stderr, "\nstats:\n");
    log_info(stderr, "  reads           : %u\n", counter.m_reads );
    log_info(stderr, "  sequence length : %llu bps (%.1f MB)\n",
        seq_length,
        float(seq_words*sizeof(uint32))/float(1024*1024));
    log_info(stderr, "  buffer size     : %.1f MB\n",
        2*seq_words*sizeof(uint32)/1.0e6f );

    const uint32 sa_intv = nvbio::io::FMIndexData::SA_INT;
    const uint32 ssa_len = (seq_length + sa_intv) / sa_intv;

    // allocate the actual storage
    thrust::host_vector<uint32> h_base_storage( seq_words );
    thrust::host_vector<uint32> h_bwt_storage( seq_words );
    thrust::host_vector<uint32> h_ssa( ssa_len );

    uint32* h_base_stream = nvbio::plain_view( h_base_storage );
    uint32* h_bwt_stream  = nvbio::plain_view( h_bwt_storage );

    typedef PackedStream<const uint32*,uint8,2,true,uint64> const_stream_type;
    typedef PackedStream<      uint32*,uint8,2,true,uint64>       stream_type;

    stream_type h_string( h_base_stream );
    stream_type h_bwt( h_bwt_stream );

    log_info(stderr, "\nbuffering bps... started\n");
    // read all files
    {
        Writer writer( h_base_stream, counter.m_reads, seq_length );

        for (uint32 i = 0; i < n_inputs; ++i)
        {
            log_info(stderr, "  buffering \"%s\"\n", sortednames[i].c_str());

            FASTA_inc_reader fasta( sortednames[i].c_str() );
            if (fasta.valid() == false)
            {
                log_error(stderr, "  unable to open file!\n");
                exit(1);
            }

            while (fasta.read( 1024, writer ) == 1024);
        }

        save_bns( writer.m_bntseq, output_name );
    }
    log_info(stderr, "buffering bps... done\n");
    {
        const uint32 crc = crcCalc( h_string.begin(), uint32(seq_length) );
        log_info(stderr, "  crc: %u\n", crc);
    }

    // writing
    if (pac_name)
    {
        log_info(stderr, "\nwriting \"%s\"... started\n", pac_name);

        const uint32 bps_per_byte = 4u;
        const uint64 seq_bytes    = (seq_length + bps_per_byte - 1u) / bps_per_byte;

        //
        // .pac file
        //

        FILE* output_file = fopen( pac_name, "wb" );
        if (output_file == NULL)
        {
            log_error(stderr, "  could not open output file \"%s\"!\n", pac_name );
            exit(1);
        }

        if (save_stream( output_file, seq_bytes, (uint8*)h_base_stream ) == false)
        {
            log_error(stderr, "  writing failed!\n");
            exit(1);
        }
		// the following code makes the pac file size always (l_pac/4+1+1)
        if (seq_length % 4 == 0)
        {
		    const uint8 ct = 0;
		    fwrite( &ct, 1, 1, output_file );
        }
        {
            const uint8 ct = seq_length % 4;
	        fwrite( &ct, 1, 1, output_file );
        }

        fclose( output_file );

        //
        // .rpac file
        //

        output_file = fopen( rpac_name, "wb" );
        if (output_file == NULL)
        {
            log_error(stderr, "  could not open output file \"%s\"!\n", rpac_name );
            exit(1);
        }

        // reuse the bwt storage to build the reverse
        uint32* h_rbase_stream = h_bwt_stream;
        stream_type h_rstring( h_rbase_stream );

        // reverse the string
        for (uint32 i = 0; i < seq_length; ++i)
            h_rstring[i] = h_string[ seq_length - i - 1u ];

        if (save_stream( output_file, seq_bytes, (uint8*)h_rbase_stream ) == false)
        {
            log_error(stderr, "  writing failed!\n");
            exit(1);
        }
		// the following code makes the pac file size always (l_pac/4+1+1)
        if (seq_length % 4 == 0)
        {
		    const uint8 ct = 0;
		    fwrite( &ct, 1, 1, output_file );
        }
        {
            const uint8 ct = seq_length % 4;
	        fwrite( &ct, 1, 1, output_file );
        }

        fclose( output_file );

        log_info(stderr, "writing \"%s\"... done\n", pac_name);
    }

    try
    {
        BWTParams params;
        uint32    primary;

        thrust::device_vector<uint32> d_base_storage( h_base_storage );
        thrust::device_vector<uint32> d_bwt_storage( seq_words );

        const_stream_type d_string( nvbio::plain_view( d_base_storage ) );
              stream_type d_bwt( nvbio::plain_view( d_bwt_storage ) );

        Timer timer;

        log_info(stderr, "\nbuilding forward BWT... started\n");
        timer.start();
        {
            StringBWTSSAHandler<const_stream_type::iterator,stream_type::iterator,uint32*> output(
                seq_length,                         // string length
                d_string.begin(),                   // string
                sa_intv,                            // SSA sampling interval
                d_bwt.begin(),                      // output bwt iterator
                nvbio::plain_view( h_ssa ) );       // output ssa iterator

            cuda::blockwise_suffix_sort(
                seq_length,
                d_string.begin(),
                output,
                &params );

            // remove the dollar symbol
            output.remove_dollar();

            primary = output.primary();
        }
        timer.stop();
        log_info(stderr, "building forward BWT... done: %um:%us\n", uint32(timer.seconds()/60), uint32(timer.seconds())%60);
        log_info(stderr, "  primary: %u\n", primary);

        // save it to disk
        {
            // copy to the host
            h_bwt_storage = d_bwt_storage;

            const uint32 cumFreq[4] = { 0, 0, 0, 0 };
            log_info(stderr, "\nwriting \"%s\"... started\n", bwt_name);
            FILE* output_file = fopen( bwt_name, "wb" );
            if (output_file == NULL)
            {
                log_error(stderr, "  could not open output file \"%s\"!\n", bwt_name );
                exit(1);
            }
            fwrite( &primary, sizeof(uint32), 1, output_file );
            fwrite( cumFreq,  sizeof(uint32), 4, output_file );
            if (save_stream( output_file, seq_words, h_bwt_stream ) == false)
            {
                log_error(stderr, "  writing failed!\n");
                exit(1);
            }
            fclose( output_file );
            log_info(stderr, "writing \"%s\"... done\n", bwt_name);
        }
        {
            log_info(stderr, "\nwriting \"%s\"... started\n", sa_name);
            FILE* output_file = fopen( sa_name, "wb" );
            if (output_file == NULL)
            {
                log_error(stderr, "  could not open output file \"%s\"!\n", sa_name );
                exit(1);
            }

            const uint32 L2[4] = { 0, 0, 0, 0 };

            fwrite( &primary,       sizeof(uint32),     1u,         output_file );
            fwrite( &L2,            sizeof(uint32),     4u,         output_file );
            fwrite( &sa_intv,       sizeof(uint32),     1u,         output_file );
            fwrite( &seq_length,    sizeof(uint32),     1u,         output_file );
            fwrite( &h_ssa[1],      sizeof(uint32),     ssa_len-1,  output_file );
            fclose( output_file );
            log_info(stderr, "writing \"%s\"... done\n", sa_name);
        }

        // reverse the string in h_base_storage
        {
            // reuse the bwt storage to build the reverse
            uint32* h_rbase_stream = h_bwt_stream;
            stream_type h_rstring( h_rbase_stream );

            // reverse the string
            for (uint32 i = 0; i < seq_length; ++i)
                h_rstring[i] = h_string[ seq_length - i - 1u ];

            // and now swap the vectors
            h_bwt_storage.swap( h_base_storage );
            std::swap( h_base_stream, h_bwt_stream );

            // and copy back the new string to the device
            d_base_storage = h_base_storage;
        }

        log_info(stderr, "\nbuilding reverse BWT... started\n");
        timer.start();
        {
            StringBWTSSAHandler<const_stream_type::iterator,stream_type::iterator,uint32*> output(
                seq_length,                         // string length
                d_string.begin(),                   // string
                sa_intv,                            // SSA sampling interval
                d_bwt.begin(),                      // output bwt iterator
                nvbio::plain_view( h_ssa ) );       // output ssa iterator

            cuda::blockwise_suffix_sort(
                seq_length,
                d_string.begin(),
                output,
                &params );

            // remove the dollar symbol
            output.remove_dollar();

            primary = output.primary();
        }
        timer.stop();
        log_info(stderr, "building reverse BWT... done: %um:%us\n", uint32(timer.seconds()/60), uint32(timer.seconds())%60);
        log_info(stderr, "  primary: %u\n", primary);

        // save it to disk
        {
            // copy to the host
            h_bwt_storage = d_bwt_storage;

            const uint32 cumFreq[4] = { 0, 0, 0, 0 };
            log_info(stderr, "\nwriting \"%s\"... started\n", rbwt_name);
            FILE* output_file = fopen( rbwt_name, "wb" );
            if (output_file == NULL)
            {
                log_error(stderr, "  could not open output file \"%s\"!\n", rbwt_name );
                exit(1);
            }
            fwrite( &primary, sizeof(uint32), 1, output_file );
            fwrite( cumFreq,  sizeof(uint32), 4, output_file );
            if (save_stream( output_file, seq_words, h_bwt_stream ) == false)
            {
                log_error(stderr, "  writing failed!\n");
                exit(1);
            }
            fclose( output_file );
            log_info(stderr, "writing \"%s\"... done\n", rbwt_name);
        }
        {
            log_info(stderr, "\nwriting \"%s\"... started\n", rsa_name);
            FILE* output_file = fopen( rsa_name, "wb" );
            if (output_file == NULL)
            {
                log_error(stderr, "  could not open output file \"%s\"!\n", rsa_name );
                exit(1);
            }

            const uint32 L2[4] = { 0, 0, 0, 0 };

            fwrite( &primary,       sizeof(uint32),     1u,         output_file );
            fwrite( &L2,            sizeof(uint32),     4u,         output_file );
            fwrite( &sa_intv,       sizeof(uint32),     1u,         output_file );
            fwrite( &seq_length,    sizeof(uint32),     1u,         output_file );
            fwrite( &h_ssa[1],      sizeof(uint32),     ssa_len-1,  output_file );
            fclose( output_file );
            log_info(stderr, "writing \"%s\"... done\n", rsa_name);
        }
    }
    catch (...)
    {
        log_info(stderr,"error: unknown exception!\n");
        exit(1);
    }
    return 0;
}

int main(int argc, char* argv[])
{
    crcInit();

    if (argc < 2)
    {
        log_info(stderr, "please specify input and output file names, e.g:\n");
        log_info(stderr, "  nvBWT [options] myinput.*.fa output-prefix\n");
        log_info(stderr, "  options:\n");
        log_info(stderr, "    -m     max_length\n");
        exit(0);
    }

    const char* file_names[2] = { NULL, NULL };
    uint64 max_length = uint64(-1);

    uint32 n_files = 0;
    for (int32 i = 1; i < argc; ++i)
    {
        const char* arg = argv[i];

        if (strcmp( arg, "-m" ) == 0)
        {
            max_length = atoi( argv[i+1] );
            ++i;
        }
        else
            file_names[ n_files++ ] = argv[i];
    }

    const char* input_name  = file_names[0];
    const char* output_name = file_names[1];
    std::string pac_string  = std::string( output_name ) + ".pac";
    const char* pac_name    = pac_string.c_str();
    std::string rpac_string = std::string( output_name ) + ".rpac";
    const char* rpac_name   = rpac_string.c_str();
    std::string bwt_string  = std::string( output_name ) + ".bwt";
    const char* bwt_name    = bwt_string.c_str();
    std::string rbwt_string = std::string( output_name ) + ".rbwt";
    const char* rbwt_name   = rbwt_string.c_str();
    std::string sa_string   = std::string( output_name ) + ".sa";
    const char* sa_name     = sa_string.c_str();
    std::string rsa_string  = std::string( output_name ) + ".rsa";
    const char* rsa_name    = rsa_string.c_str();

    log_info(stderr, "max length : %lld\n", max_length);
    log_info(stderr, "input      : \"%s\"\n", input_name);
    log_info(stderr, "output     : \"%s\"\n", output_name);

    return build( input_name, output_name, pac_name, rpac_name, bwt_name, rbwt_name, sa_name, rsa_name, max_length );
}
