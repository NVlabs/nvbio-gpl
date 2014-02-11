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

#include <nvbio/basic/types.h>
#include <nvbio/alignment/utils.h>

namespace nvbio {
namespace aln {

namespace priv {

template <
    typename        aligner_type,
    typename        pattern_string,
    typename        qual_string,
    typename        text_string,
    typename        column_type>
struct alignment_score_dispatch {};

template <
    uint32          CHECKPOINTS,
    typename        aligner_type,
    typename        pattern_string,
    typename        qual_string,
    typename        text_string,
    typename        column_type>
struct alignment_checkpointed_dispatch {};

template <
    uint32          BAND_LEN,
    typename        aligner_type,
    typename        pattern_string,
    typename        qual_string,
    typename        text_string>
struct banded_alignment_score_dispatch {};

template <
    uint32          BAND_LEN,
    uint32          CHECKPOINTS,
    typename        aligner_type,
    typename        pattern_string,
    typename        qual_string,
    typename        text_string>
struct banded_alignment_checkpointed_dispatch {};

//
// Helper class for banded alignment
//
template <uint32 BAND_SIZE> struct Reference_cache
{
    static const uint32 BAND_WORDS = (BAND_SIZE-1+7) >> 3;
    typedef PackedStream<uint32*,uint8,2,false> type;
};
template <> struct Reference_cache<15u>
{
    static const uint32 BAND_WORDS = 14u;
    typedef uint32* type;
};
template <> struct Reference_cache<7u>
{
    static const uint32 BAND_WORDS = 6u;
    typedef uint32* type;
};
template <> struct Reference_cache<5u>
{
    static const uint32 BAND_WORDS = 4u;
    typedef uint32* type;
};
template <> struct Reference_cache<3u>
{
    static const uint32 BAND_WORDS = 3u;
    typedef uint32* type;
};

} // namespace priv

} // namespace alignment
} // namespace nvbio
