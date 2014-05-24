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

// nvFM-server.cpp : Defines the entry point for the console application.
//

#include <nvbio/io/fmindex/fmindex.h>
#include <nvbio/io/sequence/sequence_mmap.h>
#include <nvbio/basic/mmap.h>
#include <string.h>
#include <string>

using namespace nvbio;

int main(int argc, char* argv[])
{
    if (argc == 1)
    {
        fprintf(stderr, "nvFM-server genome-prefix mapped-name\n");
        exit(1);
    }

    fprintf(stderr, "nvFM-server started\n");

    const char* file_name = argv[1];
    const char* mapped_name = argc == 3 ? argv[2] : argv[1];

    io::SequenceDataMMAPServer reference_driver;
    reference_driver.load( DNA, file_name, mapped_name );

    io::FMIndexDataMMAPServer fmindex_driver;
    fmindex_driver.load( file_name, mapped_name );

    getc(stdin);
    return 0;
}

