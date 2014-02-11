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

// rank_test.cu
//

#define MOD_NAMESPACE
#define MOD_NAMESPACE_NAME fmitest
#define MOD_NAMESPACE_BEGIN namespace fmitest {
#define MOD_NAMESPACE_END   }

//#define CUFMI_CUDA_DEBUG
//#define CUFMI_CUDA_ASSERTS

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <nvbio/basic/timer.h>
#include <nvbio/basic/console.h>
#include <nvbio/basic/vector_wrapper.h>
#include <nvbio/fmindex/dna.h>
#include <nvbio/basic/cached_iterator.h>
#include <nvbio/basic/packedstream.h>
#include <nvbio/basic/deinterleaved_iterator.h>
#include <nvbio/fmindex/bwt.h>
#include <nvbio/fmindex/rank_dictionary.h>

namespace nvbio {
namespace { // anonymous namespace

template <typename rank_dict_type, typename index_type>
void do_test(const index_type LEN, const rank_dict_type& dict)
{
    typedef typename vector_type<index_type,4>::type vec4;

    index_type counts[4] = { 0, 0, 0, 0 };
    for (index_type i = 0; i < LEN; ++i)
    {
        counts[ dict.text[i] ]++;

        for (uint8 c = 0; c < 4; ++c)
        {
            const index_type r = rank( dict, i, c );

            if (r != counts[c])
            {
                log_error(stderr, "  rank mismatch at [%u:%u]: expected %u, got %u\n", uint32(i), uint32(c), uint32(counts[c]), uint32(r));
                exit(1);
            }
        }

        const vec4 r4 = rank4( dict, i );
        if (r4.x != counts[0] ||
            r4.y != counts[1] ||
            r4.z != counts[2] ||
            r4.w != counts[3])
        {
            log_error(stderr, "  rank mismatch at [%u]: expected (%u,%u,%u,%u), got (%u,%u,%u,%u)\n", uint32(i),
                (uint32)counts[0], (uint32)counts[1], (uint32)counts[2], (uint32)counts[3],
                (uint32)r4.x, (uint32)r4.y, (uint32)r4.z, (uint32)r4.w);
            exit(1);
        }
    }
}

void synthetic_test(const uint32 LEN)
{
    // 32-bits test
    {
        fprintf(stderr, "  32-bit test\n");
        const uint32 OCC_INT   = 64;
        const uint32 WORDS     = (LEN+15)/16;
        const uint32 OCC_WORDS = ((LEN+OCC_INT-1) / OCC_INT) * 4;

        Timer timer;

        const uint64 memory_footprint =
            sizeof(uint32)*WORDS +
            sizeof(uint32)*OCC_WORDS;

        fprintf(stderr, "    memory  : %.1f MB\n", float(memory_footprint)/float(1024*1024));

        thrust::host_vector<uint32> text_storage( align<4>(WORDS), 0u );
        thrust::host_vector<uint32> occ(  align<4>(WORDS), 0u );
        thrust::host_vector<uint32> count_table( 256 );

        // initialize the text
        {
            typedef PackedStream<uint32*,uint8,2,true> stream_type;
            stream_type text( &text_storage[0] );

            for (uint32 i = 0; i < LEN; ++i)
                text[i] = (rand() % 4);

            // print the string
            if (LEN < 64)
            {
                char string[64];
                dna_to_string(
                    text.begin(),
                    text.begin() + LEN,
                    string );

                fprintf(stderr, "  string : %s\n", string);
            }

            uint32 L2[5];

            // build the occurrence table
            build_occurrence_table<OCC_INT>(
                text.begin(),
                text.begin() + LEN,
                &occ[0],
                &L2[1] );
        }

        // generate the count table
        gen_bwt_count_table( &count_table[0] );

        // test uint32 support
        {
            typedef PackedStream<const uint32*,uint8,2,true> stream_type;
            stream_type text( &text_storage[0] );

            typedef rank_dictionary<2u, OCC_INT, stream_type, const uint32*, const uint32*> rank_dict_type;
            rank_dict_type dict(
                text,
                &occ[0],
                &count_table[0] );

            do_test( LEN, dict );
        }
        // test uint4 support
        {
            typedef PackedStream<const uint4*,uint8,2,true> stream_type;
            stream_type text( (const uint4*)&text_storage[0] );

            typedef rank_dictionary<2u, OCC_INT, stream_type, const uint4*, const uint32*> rank_dict_type;
            rank_dict_type dict(
                text,
                (const uint4*)&occ[0],
                &count_table[0] );

            do_test( LEN, dict );
        }
    }
    // 64-bits test
    {
        fprintf(stderr, "  64-bit test\n");
        const uint32 OCC_INT   = 128;
        const uint32 WORDS     = (LEN+31)/32;
        const uint32 OCC_WORDS = ((LEN+OCC_INT-1) / OCC_INT) * 4;

        Timer timer;

        const uint64 memory_footprint =
            sizeof(uint64)*WORDS +
            sizeof(uint64)*OCC_WORDS;

        fprintf(stderr, "    memory  : %.1f MB\n", float(memory_footprint)/float(1024*1024));

        thrust::host_vector<uint64> text_storage( align<4>(WORDS), 0u );
        thrust::host_vector<uint64> occ(  align<4>(WORDS), 0u );
        thrust::host_vector<uint32> count_table( 256 );

        // initialize the text
        {
            typedef PackedStream<uint64*,uint8,2,true,uint64> stream_type;
            stream_type text( &text_storage[0] );

            for (uint32 i = 0; i < LEN; ++i)
                text[i] = (rand() % 4);

            // print the string
            if (LEN < 64)
            {
                char string[64];
                dna_to_string(
                    text.begin(),
                    text.begin() + LEN,
                    string );

                fprintf(stderr, "  string : %s\n", string);
            }

            uint64 L2[5];

            // build the occurrence table
            build_occurrence_table<OCC_INT>(
                text.begin(),
                text.begin() + LEN,
                &occ[0],
                &L2[1] );
        }

        // generate the count table
        gen_bwt_count_table( &count_table[0] );

        // test uint64 support
        {
            typedef PackedStream<const uint64*,uint8,2,true,uint64> stream_type;
            stream_type text( &text_storage[0] );

            typedef rank_dictionary<2u, OCC_INT, stream_type, const uint64*, const uint32*> rank_dict_type;
            rank_dict_type dict(
                text,
                &occ[0],
                &count_table[0] );

            do_test( uint64(LEN), dict );
        }
    }
}

} // anonymous namespace

int rank_test(int argc, char* argv[])
{
    uint32 len = 10000000;

    for (int i = 0; i < argc; ++i)
    {
        if (strcmp( argv[i], "-length" ) == 0)
            len = atoi( argv[++i] )*1000;
    }

    fprintf(stderr, "rank test... started\n");

    synthetic_test( len );

    fprintf(stderr, "rank test... done\n");
    return 0;
}

} // namespace nvbio
