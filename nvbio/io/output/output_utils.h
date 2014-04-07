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

#pragma once

#include <nvbio/io/alignments.h>
#include <nvbio/io/fmi.h>

#include <nvbio/basic/dna.h>

namespace nvbio {
namespace io {

struct SeqFinder
{
    bool operator() (const io::BNTAnn& ann1, const io::BNTAnn& ann2) const
    {
        return ann1.offset < ann2.offset;
    }
    bool operator() (const io::BNTAnn& ann, const uint32 x) const
    {
        return ann.offset < x;
    }
    bool operator() (const uint32 x, const io::BNTAnn& ann) const
    {
        return x < ann.offset;
    }
};

// compute the CIGAR alignment position given the alignment base and the sink offset
inline uint32 compute_cigar_pos(const uint32 sink, const uint32 alignment)
{
    return alignment + (sink & 0xFFFFu);
}

// compute the reference mapped length from a CIGAR
template <typename vector_type>
uint32 reference_cigar_length(
    const vector_type cigar,
    const uint32      cigar_len)
{
    uint32 r = 0;
    for (uint32 i = 0; i < cigar_len; ++i)
    {
        const uint32 l  = cigar[ cigar_len - i - 1u ].m_len;
        const uint32 op = cigar[ cigar_len - i - 1u ].m_type;
        if (op == Cigar::SUBSTITUTION || op == Cigar::DELETION) r += l;
    }
    return r;
}

// count the symbols of a given type inside a CIGAR
template <typename vector_type>
uint32 count_symbols(
    const Cigar::Operation  type,
    const vector_type       cigar,
    const uint32            cigar_len)
{
    uint32 r = 0;
    for (uint32 i = 0; i < cigar_len; ++i)
    {
        const uint32 l  = cigar[ cigar_len - i - 1u ].m_len;
        const uint32 op = cigar[ cigar_len - i - 1u ].m_type;
        if (op == type) r += l;
    }
    return r;
}

// build the MD string from the internal representation
template <typename vector_type>
void analyze_md_string(const vector_type mds, uint32& n_mm, uint32& n_gapo, uint32& n_gape)
{
    const uint32 mds_len = uint32(mds[0]) | (uint32(mds[1]) << 8);

    n_mm   = 0;
    n_gapo = 0;
    n_gape = 0;

    for (uint32 i = 2; i < mds_len; )
    {
        const uint8 op = mds[i++];
        if (op == MDS_MATCH)
        {
            uint8 l = mds[i++];

            // prolong the MDS match if it spans multiple tokens
            while (i < mds_len && mds[i] == MDS_MATCH)
                l += mds[i++];
        }
        else if (op == MDS_MISMATCH)
        {
            n_mm++;

            ++i;
        }
        else if (op == MDS_INSERTION)
        {
            const uint8 l = mds[i++];

            n_gapo++;
            n_gape += l-1;

            i += l;
        }
        else if (op == MDS_DELETION)
        {
            const uint8 l = mds[i++];

            n_gapo++;
            n_gape += l-1;

            i += l;
        }
    }
}

} // namespace io
} // namespace nvbio
