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

// fmindex_test.cpp
//

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <nvbio/basic/timer.h>
#include <nvbio/basic/dna.h>
#include <nvbio/basic/packedstream.h>
#include <nvbio/fmindex/bwt.h>

using namespace nvbio;

int bwt_test()
{
    fprintf(stderr, "bwt test... started\n");
    const int32  LEN = 10000000;
    const uint32 WORDS = (LEN+16)/16;

    const uint64 memory_footprint =
        sizeof(uint32)*uint64(WORDS) +
        sizeof(uint32)*uint64(LEN);

    fprintf(stderr, "  arch    : %lu bit\n", sizeof(void*)*8u);
    fprintf(stderr, "  length  : %.2f M bps\n", float(LEN)/1.0e6f);
    fprintf(stderr, "  memory  : %.1f MB\n", float(memory_footprint)/float(1024*1024));

    std::vector<int32> buffer( LEN+1 + WORDS, 0u );
    int32*  bwt_temp    = &buffer[0];
    uint32* base_stream = (uint32*)&buffer[0] + LEN+1;

    typedef PackedStream<uint32*,uint8,2,true> stream_type;
    stream_type stream( base_stream );

    srand(0);
    for (int32 i = 0; i < LEN; ++i)
    {
        const uint32 s = rand() % 4;
        stream[i] = s;
    }

    fprintf(stderr, "  construction... started\n");

    Timer timer;
    timer.start();

    gen_bwt( LEN, stream.begin(), &bwt_temp[0], stream.begin() );

    timer.stop();
    fprintf(stderr, "  construction... done: %um:%us\n", uint32(timer.seconds()/60), uint32(timer.seconds())%60);

    fprintf(stderr, "bwt test... done\n");
    return 0;
}