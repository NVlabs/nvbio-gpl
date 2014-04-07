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

namespace nvbio {

/// convert a 2-bit DNA symbol to its ASCII character
///
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE char dna_to_char(const uint8 c)
{
    return c == 0 ? 'A' :
           c == 1 ? 'C' :
           c == 2 ? 'G' :
           c == 3 ? 'T' :
                    'N';
}
/// convert an ASCII DNA representation to its 2-bit symbol
///
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint8 char_to_dna(const char c)
{
    return c == 'A' ? 0u :
           c == 'C' ? 1u :
           c == 'G' ? 2u :
           c == 'T' ? 3u :
                      4u;
}

/// convert a 2-bit DNA string to an ASCII string
///
template <typename SymbolIterator>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void dna_to_string(
    const SymbolIterator begin,
    const uint32 n,
    char* string)
{
    for (uint32 i = 0; i < n; ++i)
        string[i] = dna_to_char( begin[i] );

    string[n] = '\0';
}

/// convert a 2-bit DNA string to an ASCII string
///
template <typename SymbolIterator>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void dna_to_string(
    const SymbolIterator begin,
    const SymbolIterator end,
    char* string)
{
    for (SymbolIterator it = begin; it != end; ++it)
        string[ (it - begin) % (end - begin) ] = dna_to_char( *it );

    string[ end - begin ] = '\0';
}
/// convert an ASCII DNA string to its 2-bit representation
///
template <typename SymbolIterator>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void string_to_dna(
    const char* begin,
    const char* end,
    SymbolIterator symbols)
{
    for (const char* it = begin; it != end; ++it)
        symbols[ (it - begin) % (end - begin) ] = char_to_dna( *it );
}

/// convert a NULL-terminated ASCII DNA string to its 2-bit representation
///
template <typename SymbolIterator>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void string_to_dna(
    const char* begin,
    SymbolIterator symbols)
{
    for (const char* it = begin; *it != '\0'; ++it)
        symbols[ it - begin ] = char_to_dna( *it );
}

} // namespace nvbio

