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

#include <nvbio/basic/types.h>

namespace nvbio {

///
/// The supported sequence alphabet types
///
enum SequenceAlphabet
{
    DNA     = 0u,
    DNA_N   = 1u,
    PROTEIN = 2u
};

/// A traits class for SequenceAlphabet
///
template <SequenceAlphabet ALPHABET> struct SequenceAlphabetTraits {};

/// A traits class for DNA SequenceAlphabet
///
template <> struct SequenceAlphabetTraits<DNA>
{
    static const uint32 SYMBOL_SIZE  = 2;
    static const uint32 SYMBOL_COUNT = 4;
};
/// A traits class for DNA_N SequenceAlphabet
///
template <> struct SequenceAlphabetTraits<DNA_N>
{
    static const uint32 SYMBOL_SIZE  = 4;
    static const uint32 SYMBOL_COUNT = 5;
};
/// A traits class for Protein SequenceAlphabet
///
template <> struct SequenceAlphabetTraits<PROTEIN>
{
    static const uint32 SYMBOL_SIZE  = 8;
    static const uint32 SYMBOL_COUNT = 24;
};

NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
uint32 bits_per_symbol(const SequenceAlphabet alphabet)
{
    return alphabet == DNA     ? 2 :
           alphabet == DNA_N   ? 4 :
           alphabet == PROTEIN ? 8 :
           8u;
}

} // namespace nvbio
