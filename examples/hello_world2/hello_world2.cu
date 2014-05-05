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

// hello_world2.cu
//

#include <stdio.h>
#include <stdlib.h>
#include <nvbio/basic/packed_vector.h>
#include <nvbio/basic/dna.h>
#include <nvbio/strings/infix.h>
#include <nvbio/strings/seeds.h>


using namespace nvbio;

template <typename string_set_type>
struct print_strings
{
    NVBIO_HOST_DEVICE
    print_strings(const string_set_type _string_set) : string_set( _string_set ) {}

    NVBIO_HOST_DEVICE
    void operator() (const uint32 i) const
    {
        char seed[4];
        dna_to_string(
            string_set[i],
            length( string_set[i] ),
            seed );

        printf("seed[%u] = %s\n", i, seed);
    }

    const string_set_type string_set;
};

// main test entry point
//
int main(int argc, char* argv[])
{
    // our hello world ASCII string
    const char dna_string[] = "ACGTTGCA";
    const uint32 len = (uint32)strlen( dna_string );

    // our DNA alphabet size
    const uint32 ALPHABET_SIZE = 2u;

    // instantiate a packed host vector
    nvbio::PackedVector<host_tag,ALPHABET_SIZE> h_dna( len );

    // pack our ASCII string
    string_to_dna(
        dna_string,         // begin iterator of our ASCII string
        h_dna.begin() );    // begin iterator of our output string

    // instantiate a packed host vector
    nvbio::PackedVector<device_tag,ALPHABET_SIZE> d_dna( h_dna );

    // prepare a vector to store the coordinates of the resulting infixes
    nvbio::vector<device_tag,string_infix_coord_type> d_seed_coords;

    const uint32 n_seeds = enumerate_string_seeds(
		len,                                    // the input string length
        uniform_seeds_functor<>( 3u, 1u ),      // a seeding functor, specifying to extract all 3-mers offset by 1 base each
        d_seed_coords );                        // the output infix coordinates

    // define an infix-set to represent the resulting infixes
    typedef nvbio::PackedVector<device_tag,ALPHABET_SIZE>::iterator             packed_iterator_type;
    typedef nvbio::vector<device_tag,string_infix_coord_type>::const_iterator   infix_iterator_type;
    typedef InfixSet<packed_iterator_type, infix_iterator_type> infix_set_type;

    infix_set_type seeds(
        n_seeds,                                // the number of infixes in the set
        d_dna.begin(),                          // the underlying string
        d_seed_coords.begin() );                // the iterator to the infix coordinates

    thrust::for_each(
        device_tag(),
        thrust::make_counting_iterator(0u),
        thrust::make_counting_iterator(0u) + seeds.size(),
        print_strings<infix_set_type>( seeds ) );

    return 0;
}
