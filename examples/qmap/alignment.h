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

// alignment.h
//

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <nvbio/alignment/alignment.h>
#include <nvbio/alignment/batched.h>
#include <nvbio/strings/string_set.h>
#include <nvbio/io/sequence/sequence.h>

using namespace nvbio;

// a functor to extract the read infixes from the hit diagonals
//
struct read_infixes
{
    // constructor
    NVBIO_HOST_DEVICE
    read_infixes(const io::ConstSequenceDataView reads) :
        m_reads( reads ) {}

    // functor operator
    NVBIO_HOST_DEVICE
    string_infix_coord_type operator() (const uint2 diagonal) const
    {
        const io::SequenceDataAccess<DNA_N> reads( m_reads );

        const uint32 read_id = diagonal.y;

        // fetch the read range
        return reads.get_range( read_id );
    }

    const io::ConstSequenceDataView m_reads;
};

// a functor to extract the genome infixes from the hit diagonals
//
template <uint32 BAND_LEN>
struct genome_infixes
{
    // constructor
    NVBIO_HOST_DEVICE
    genome_infixes(const uint32 genome_len, const io::ConstSequenceDataView reads) :
        m_genome_len( genome_len ),
        m_reads( reads ) {}

    // functor operator
    NVBIO_HOST_DEVICE
    string_infix_coord_type operator() (const uint2 diagonal) const
    {
        const io::SequenceDataAccess<DNA_N> reads( m_reads );

        const uint32 read_id  = diagonal.y;
        const uint32 text_pos = diagonal.x;

        // fetch the read range
        const uint2  read_range = reads.get_range( read_id );
        const uint32 read_len   = read_range.y - read_range.x;

        // compute the segment of text to align to
        const uint32 genome_begin = text_pos > BAND_LEN/2 ? text_pos - BAND_LEN/2 : 0u;
        const uint32 genome_end   = nvbio::min( genome_begin + read_len + BAND_LEN, m_genome_len );

        return make_uint2( genome_begin, genome_end );
    }

    const uint32                    m_genome_len;
    const io::ConstSequenceDataView m_reads;
};

// a functor to extract the score from a sink
//
struct sink_score
{
    typedef aln::BestSink<int16> argument_type;
    typedef int16                result_type;

    // functor operator
    NVBIO_HOST_DEVICE
    int16 operator() (const aln::BestSink<int16>& sink) const { return sink.score; }
};
