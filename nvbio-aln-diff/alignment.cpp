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

#include <nvbio-aln-diff/alignment.h>
#include <zlib/zlib.h>
#include <string.h>

namespace nvbio {
namespace alndiff {

AlignmentStream* open_dbg_file(const char* file_name);
AlignmentStream* open_bam_file(const char* file_name);

AlignmentStream* open_alignment_file(const char* file_name)
{
    if (strcmp( file_name + strlen(file_name) - 4u, ".dbg" ) == 0)
        return open_dbg_file( file_name );
    if (strcmp( file_name + strlen(file_name) - 4u, ".bam" ) == 0)
        return open_bam_file( file_name );

    return NULL;
}

} // alndiff namespace
} // nvbio namespace
