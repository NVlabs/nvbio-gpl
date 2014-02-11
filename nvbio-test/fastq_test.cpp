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

#include <nvbio/fastq/fastq.h>

using namespace nvbio;

namespace {

struct Writer
{
    Writer() : n(0) {}

    void push_back(const uint32 read_len, const char* name, const uint8* bp, const uint8* q)
    {
#if 0
        if ((n & 0x00FF) == 0)
        {
            char read[16*1024];
            for (uint32 i = 0; i < read_len; ++i)
                read[i] = bp[i];
            read[read_len] = '\0';

            fprintf( stderr, "  len: %u, read: %s\n", read_len, read );
        }
#endif
        n++;
    }

    uint32 n;
};

} // anonymous namespace

int fastq_test(const char* filename)
{
    fprintf(stderr, "FASTQ test... started\n");

    FASTQ_file fastq_file( filename );
    if (fastq_file.valid() == false)
    {
        fprintf(stderr, "*** error *** : file \"%s\" not found\n", filename);
        return 1;
    }

    FASTQ_reader<FASTQ_file> fastq( fastq_file );

    Writer writer;

    int n;

    while ((n = fastq.read( 100u, writer )))
    {
        if (n < 0)
        {
            fprintf(stderr, "*** parsing error ***\n");
            char error[1024];
            fastq.error_string( error );
            fprintf(stderr, "  %s\n", error);
            return 1;
        }
    }

    fprintf(stderr, "FASTQ test... done: %u reads\n", writer.n);
    return 0;
}
