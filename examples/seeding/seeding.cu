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

// seeding.cu
//

#include <stdio.h>
#include <stdlib.h>
#include <nvbio/basic/console.h>
#include <nvbio/basic/vector.h>
#include <nvbio/basic/shared_pointer.h>
#include <nvbio/strings/string_set.h>
#include <nvbio/strings/infix.h>
#include <nvbio/strings/seeds.h>
#include <nvbio/io/reads/reads.h>

using namespace nvbio;

// extract a set of uniformly spaced seeds from a string-set and return it as an InfixSet
//
template <typename system_tag, typename string_set_type>
InfixSet<string_set_type, const string_set_infix_coord_type*>
extract_seeds(
    const string_set_type                                   string_set,
    const uint32                                            seed_len,
    const uint32                                            seed_interval,
    nvbio::vector<system_tag,string_set_infix_coord_type>&  seed_coords)
{
    // enumerate all seeds
    const uint32 n_seeds = enumerate_string_set_seeds(
        string_set,
        uniform_seeds_functor<>( seed_len, seed_interval ),
        seed_coords );

    // and build the output infix-set
    return InfixSet<string_set_type, const string_set_infix_coord_type*>(
        n_seeds,
        string_set,
        nvbio::plain_view( seed_coords ) );
}

// main test entry point
//
int main(int argc, char* argv[])
{
    uint32 n_bps = 10000000;
    char*  reads = "./data/SRR493095_1.fastq.gz";

    for (int i = 0; i < argc; ++i)
    {
        if (strcmp( argv[i], "-bps" ) == 0)
            n_bps = uint32( atoi( argv[++i] ) )*1000u;
        else if (strcmp( argv[i], "-reads" ) == 0)
            reads = argv[++i];
    }

    log_info(stderr, "seeding... started\n");

    const io::QualityEncoding qencoding = io::Phred33;

    log_info(stderr, "  loading reads... started\n");

    SharedPointer<io::ReadDataStream> read_data_file(
        io::open_read_file(
            reads,
            qencoding,
            uint32(-1),
            uint32(-1) ) );

    if (read_data_file == NULL || read_data_file->is_ok() == false)
    {
        log_error(stderr, "    failed opening file \"%s\"\n", reads);
        return 1u;
    }

    const uint32 batch_size = uint32(-1);
    const uint32 batch_bps  = n_bps;

    // load a batch of reads
    SharedPointer<io::ReadData> h_read_data( read_data_file->next( batch_size, batch_bps ) );

    const io::ReadDataDevice d_read_data( *h_read_data );

    // fetch the actual string
    typedef io::ReadData::const_read_string_set_type                        string_set_type;
    typedef nvbio::vector<device_tag,string_set_infix_coord_type>           infix_vector_type;
    typedef InfixSet<string_set_type, const string_set_infix_coord_type*>   seed_set_type;

    const string_set_type d_read_string_set = d_read_data.read_string_set();

    // prepare enough storage for the seed coordinates
    infix_vector_type d_seed_coords;

    const seed_set_type d_seed_set = extract_seeds(
        d_read_string_set,
        20u,
        10u,
        d_seed_coords );

    log_info(stderr, "seeding... done\n");
    log_info(stderr, "  %u seeds\n", d_seed_set.size());

    return 0;
}
