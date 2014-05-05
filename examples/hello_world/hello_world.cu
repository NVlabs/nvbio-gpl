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

// hello_world.cu
//

#include <stdio.h>
#include <stdlib.h>
#include <nvbio/basic/packed_vector.h>
#include <nvbio/basic/dna.h>
#include <nvbio/strings/infix.h>
#include <nvbio/strings/seeds.h>


using namespace nvbio;


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

    // copy the packed vector to the device
    nvbio::PackedVector<device_tag,ALPHABET_SIZE> d_dna( h_dna );

    return 0;
}
