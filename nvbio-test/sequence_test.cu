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

// alignment_test.cu
//

#include <nvbio/basic/timer.h>
#include <nvbio/basic/console.h>
#include <nvbio/basic/packedstream.h>
#include <nvbio/basic/packedstream_loader.h>
#include <nvbio/basic/shared_pointer.h>
#include <nvbio/basic/console.h>
#include <nvbio/basic/dna.h>
#include <nvbio/io/sequence/sequence.h>
#include <nvbio/io/sequence/sequence_mmap.h>
#include <stdio.h>
#include <stdlib.h>

using namespace nvbio;

namespace nvbio {

int sequence_test(int argc, char* argv[])
{
    char* index_name = NULL;
    char* reads_name = NULL;

    for (int i = 0; i < argc; ++i)
    {
        if (strcmp( argv[i], "-map" ) == 0)
            index_name = argv[++i];
        else if (strcmp( argv[i], "-reads" ) == 0)
            reads_name = argv[++i];
    }

    log_info(stderr,"testing sequence-data... started\n");

    if (index_name != NULL)
    {
        io::SequenceDataMMAPServer server;
        if (server.load( DNA, index_name, "test", io::SequenceFlags( io::SEQUENCE_DATA | io::SEQUENCE_NAMES ) ) == false)
        {
            log_error(stderr,"  server mapping of file %s failed\n", index_name);
            return 0;
        }

        io::SequenceDataMMAP client;
        if (client.load( "test" ) == false)
        {
            log_error(stderr,"  client mapping of file %s failed\n", index_name);
            return 0;
        }

        log_verbose(stderr, "  sequences : %u\n", client.size() );
        log_verbose(stderr, "  bps       : %u\n", client.bps() );
    }
    if (reads_name != NULL)
    {
        SharedPointer<io::SequenceDataStream> read_file( io::open_sequence_file( reads_name ) );
        if (read_file == NULL || read_file->is_ok() == false)
        {
            log_error(stderr,"  failed opening reads file %s\n", reads_name);
            return 0;
        }

        io::SequenceDataHost read_data;

        io::next( DNA_N, &read_data, read_file.get(), 10000 );

        log_verbose(stderr, "  sequences : %u\n", read_data.size() );
        log_verbose(stderr, "  bps       : %u\n", read_data.bps() );
        log_verbose(stderr, "  avg bps   : %u (min: %u, max: %u)\n",
            read_data.avg_sequence_len(),
            read_data.min_sequence_len(),
            read_data.max_sequence_len() );
    }

    log_info(stderr,"testing sequence-data... done\n");
    return 1;
}

} // namespace nvbio
