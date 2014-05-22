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

#pragma once

#include <nvbio/io/sequence/sequence_alphabet.h>

namespace nvbio {
namespace io {

///@addtogroup IO
///@{

///@addtogroup SequenceIO
///@{

template <SequenceAlphabet SEQUENCE_ALPHABET>
struct SequenceDataTraits
{
    static const uint32 SEQUENCE_BITS             = SequenceAlphabetTraits<SEQUENCE_ALPHABET>::SYMBOL_SIZE;     ///< symbol size for reads
    static const bool   SEQUENCE_BIG_ENDIAN       = false;                                                      ///< big endian?
    static const uint32 SEQUENCE_SYMBOLS_PER_WORD = (8*sizeof(uint32))/SEQUENCE_BITS;                           ///< symbols per word
};

///@} // SequenceIO
///@} // IO

} // namespace io
} // namespace nvbio
