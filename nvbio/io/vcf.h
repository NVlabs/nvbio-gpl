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

#include <nvbio/basic/types.h>
#include <nvbio/basic/vector.h>
#include <nvbio/basic/packed_vector.h>

#include <vector>
#include <string>

#pragma once

namespace nvbio {
namespace io {

struct SNP_sequence_index
{
    // these indices are stored in base-pairs since variants are extremely short
    uint64 reference_start;
    uint32 reference_len;
    uint64 variant_start;
    uint32 variant_len;

    SNP_sequence_index()
        : reference_start(0), reference_len(0),
          variant_start(0), variant_len(0)
    { }

    SNP_sequence_index(uint64 reference_start, uint32 reference_len, uint64 variant_start, uint32 variant_len)
        : reference_start(reference_start), reference_len(reference_len),
          variant_start(variant_start), variant_len(variant_len)
    { }
};

struct SNPDatabase
{

    // the name of the reference sequence
    // note: VCF allows this to be an integer ID encoded in a string that references
    // a contig from an assembly referenced in the header; this is not supported yet
    std::vector<std::string> reference_sequence_names;

    // position of the variant in the reference sequence (first base in the sequence is position 1)
    nvbio::vector<host_tag, uint64> positions;

    // packed reference sequences
    nvbio::PackedVector<host_tag, 4> reference_sequences;
    // packed variant sequences
    nvbio::PackedVector<host_tag, 4> variants;
    // an index for both references and variants
    nvbio::vector<host_tag, SNP_sequence_index> ref_variant_index;

    // quality value assigned to each variant
    nvbio::vector<host_tag, uint8> variant_qualities;

    SNPDatabase()
    {
        reference_sequences.clear();
        variants.clear();
        ref_variant_index.clear();
    }
};

// loads variant data from file_name and appends to output
bool loadVCF(SNPDatabase& output, const char *file_name);

} // namespace io
} // namespace nvbio
