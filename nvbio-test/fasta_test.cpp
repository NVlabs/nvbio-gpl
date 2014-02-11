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

#include <nvbio/fasta/fasta.h>

using namespace nvbio;

namespace {

struct Writer
{
    Writer() : n(0) {}

    void push_back(const char* id, const uint32 read_len, const uint8* bp)
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

int fasta_test(const char* filename)
{
    fprintf(stderr, "FASTA test... started\n");

    FASTA_reader fasta( filename );
    if (fasta.valid() == false)
    {
        fprintf(stderr, "*** error *** : file \"%s\" not found\n", filename);
        return 1;
    }

    Writer writer;

    int n;

    while ((n = fasta.read( 100u, writer )))
    {
        if (n < 0)
        {
            fprintf(stderr, "*** parsing error ***\n");
            return 1;
        }
    }

    fprintf(stderr, "FASTA test... done: %u reads\n", writer.n);
    return 0;
}
