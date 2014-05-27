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
#include <nvbio/basic/dna.h>

namespace nvbio {

/// convert a Protein symbol to its ASCII character
///
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE char protein_to_char(const uint8 c)
{
    return c ==  0 ? 'A' :
           c ==  1 ? 'C' :
           c ==  2 ? 'D' :
           c ==  3 ? 'E' :
           c ==  4 ? 'F' :
           c ==  5 ? 'G' :
           c ==  6 ? 'H' :
           c ==  7 ? 'I' :
           c ==  8 ? 'K' :
           c ==  9 ? 'L' :
           c == 10 ? 'M' :
           c == 11 ? 'N' :
           c == 12 ? 'P' :
           c == 13 ? 'Q' :
           c == 14 ? 'R' :
           c == 15 ? 'S' :
           c == 16 ? 'T' :
           c == 17 ? 'V' :
           c == 18 ? 'W' :
           c == 19 ? 'Y' :
           c == 20 ? 'B' :
           c == 21 ? 'Z' :
           c == 22 ? 'X' :
                     'N';
}

/// convert an ASCII character to a Protein symbol
///
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint8 char_to_protein(const char c)
{
    return c == 'A' ? 0u  :
           c == 'C' ? 1u  :
           c == 'D' ? 2u  :
           c == 'E' ? 3u  :
           c == 'F' ? 4u  :
           c == 'G' ? 5u  :
           c == 'H' ? 6u  :
           c == 'I' ? 7u  :
           c == 'K' ? 8u  :
           c == 'L' ? 9u  :
           c == 'M' ? 10u :
           c == 'N' ? 11u :
           c == 'P' ? 12u :
           c == 'Q' ? 13u :
           c == 'R' ? 14u :
           c == 'S' ? 15u :
           c == 'T' ? 16u :
           c == 'V' ? 17u :
           c == 'W' ? 18u :
           c == 'Y' ? 19u :
           c == 'B' ? 20u :
           c == 'Z' ? 21u :
           c == 'X' ? 22u :
                      11u;
}

// convert a given symbol to its ASCII character
//
template <Alphabet ALPHABET>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE char to_char(const uint8 c)
{
    if (ALPHABET == DNA)
        return dna_to_char( c );
    else if (ALPHABET == DNA_N)
        return dna_to_char( c );
    else if (ALPHABET == IUPAC16)
        return iupac16_to_char( c );
    else if (ALPHABET == PROTEIN) // TODO!
        return protein_to_char( c );
}

// convert a given symbol to its ASCII character
//
template <Alphabet ALPHABET>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint8 from_char(const char c)
{
    if (ALPHABET == DNA)
        return char_to_dna( c );
    else if (ALPHABET == DNA_N)
        return char_to_dna( c );
    else if (ALPHABET == IUPAC16)
        return char_to_iupac16( c );
    else if (ALPHABET == PROTEIN) // TODO!
        return char_to_protein( c );
}

// convert from the given alphabet to an ASCII string
//
template <Alphabet ALPHABET, typename SymbolIterator>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void to_string(
    const SymbolIterator begin,
    const uint32         n,
    char*                string)
{
    for (uint32 i = 0; i < n; ++i)
        string[i] = to_char<ALPHABET>( begin[i] );

    string[n] = '\0';
}

// convert from the given alphabet to an ASCII string
//
template <Alphabet ALPHABET, typename SymbolIterator>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void to_string(
    const SymbolIterator begin,
    const SymbolIterator end,
    char*                string)
{
    for (SymbolIterator it = begin; it != end; ++it)
        string[ (it - begin) % (end - begin) ] = to_char<ALPHABET>( *it );

    string[ end - begin ] = '\0';
}

// convert from an ASCII string to the given alphabet
//
template <Alphabet ALPHABET, typename SymbolIterator>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void from_string(
    const char*             begin,
    const char*             end,
    const SymbolIterator    symbols)
{
    for (const char* it = begin; it != end; ++it)
        symbols[ (it - begin) % (end - begin) ] = from_char<ALPHABET>( *it );
}

// convert from an ASCII string to the given alphabet
//
template <Alphabet ALPHABET, typename SymbolIterator>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void from_string(
    const char*             begin,
    const SymbolIterator    symbols)
{
    for (const char* it = begin; *it != '\0'; ++it)
        symbols[ it - begin ] = from_char<ALPHABET>( *it );
}

} // namespace nvbio

