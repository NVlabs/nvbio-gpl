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

#include <nvbio/io/fmi.h>
#include <nvbio/basic/mmap.h>
#include <string.h>
#include <string>

using namespace nvbio;

struct Info
{
    uint32 sequence_length;
    uint32 sequence_words;
    uint32 occ_words;
    uint32 sa_words;
    uint32 primary;
    uint32 rprimary;
    uint32 L2[5];
    uint32 rL2[5];
};

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

    io::FMIndexDataMMAPServer driver;
    driver.load( file_name, mapped_name );

    getc(stdin);
    return 0;
}

