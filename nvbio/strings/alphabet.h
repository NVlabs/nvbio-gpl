/*
 * Copyright (c) 2012-14, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 *
 *
 *
 *
 *
 *
 *
 *
 */

#pragma once

#include <nvbio/basic/types.h>
#include <nvbio/basic/transform_iterator.h>

namespace nvbio {

///@addtogroup Strings
///@{

///\defgroup AlphabetsModule Alphabets
///
/// This module provides various operators to work with the following alphabets:
///
/// - DNA
/// - DNA_N
/// - IUPAC16
/// - PROTEIN
///
///@{

///
/// The supported sequence alphabet types
///
enum Alphabet
{
    DNA     = 0u,           ///< 4-letter DNA alphabet { A,C,G,T }
    DNA_N   = 1u,           ///< 5-letter DNA + N alphabet { A,C,G,T,N }
    PROTEIN = 2u,           ///< 23-letter Protein alphabet { A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y,B,Z,X }
    IUPAC16 = 3u            ///< 16-letter DNA IUPAC alphabet { =,A,C,M,G,R,S,V,T,W,Y,H,K,D,B,N }
};

/// A traits class for Alphabet
///
template <Alphabet ALPHABET> struct AlphabetTraits {};

/// A traits class for DNA Alphabet
///
template <> struct AlphabetTraits<DNA>
{
    static const uint32 SYMBOL_SIZE  = 2;
    static const uint32 SYMBOL_COUNT = 4;
};
/// A traits class for DNA_N Alphabet
///
template <> struct AlphabetTraits<DNA_N>
{
    static const uint32 SYMBOL_SIZE  = 4;
    static const uint32 SYMBOL_COUNT = 5;
};
/// A traits class for DNA_N Alphabet
///
template <> struct AlphabetTraits<IUPAC16>
{
    static const uint32 SYMBOL_SIZE  = 4;
    static const uint32 SYMBOL_COUNT = 16;
};
/// A traits class for Protein Alphabet
///
template <> struct AlphabetTraits<PROTEIN>
{
    static const uint32 SYMBOL_SIZE  = 8;
    static const uint32 SYMBOL_COUNT = 24;
};

NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
uint32 bits_per_symbol(const Alphabet alphabet)
{
    return alphabet == DNA     ? 2 :
           alphabet == DNA_N   ? 4 :
           alphabet == IUPAC16 ? 4 :
           alphabet == PROTEIN ? 8 :
           8u;
}

/// convert a given symbol to its ASCII character
///
template <Alphabet ALPHABET>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE char to_char(const uint8 c);

/// convert a given symbol to its ASCII character
///
template <Alphabet ALPHABET>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint8 from_char(const char c);

/// convert from the given alphabet to an ASCII string
///
template <Alphabet ALPHABET, typename SymbolIterator>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void to_string(
    const SymbolIterator begin,
    const uint32         n,
    char*                string);

/// convert from the given alphabet to an ASCII string
///
template <Alphabet ALPHABET, typename SymbolIterator>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void to_string(
    const SymbolIterator begin,
    const SymbolIterator end,
    char*                string);

/// convert from an ASCII string to the given alphabet
///
template <Alphabet ALPHABET, typename SymbolIterator>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void from_string(
    const char*             begin,
    const char*             end,
    const SymbolIterator    symbols);

/// convert from an ASCII string to the given alphabet
///
template <Alphabet ALPHABET, typename SymbolIterator>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE void from_string(
    const char*             begin,
    const SymbolIterator    symbols);

/// conversion functor from a given alphabet to ASCII char
///
template <Alphabet ALPHABET>
struct to_char_functor
{
    typedef uint8 argument_type;
    typedef char  result_type;

    /// functor operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE char operator() (const uint8 c) { return to_char<ALPHABET>( c ); }
};

/// conversion functor from a given alphabet to ASCII char
///
template <Alphabet ALPHABET>
struct from_char_functor
{
    typedef char  argument_type;
    typedef uint8 result_type;

    /// functor operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE uint8 operator() (const char c) { return from_char<ALPHABET>( c ); }
};

/// convert a string iterator from a given alphabet to an ASCII string iterator
///
template <Alphabet ALPHABET, typename Iterator>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
transform_iterator< Iterator, to_char_functor<ALPHABET> > 
to_string(Iterator it)
{
    return make_transform_iterator( it, to_char_functor<ALPHABET>() );
}

/// convert an ASCII string iterator from to a given alphabet string iterator
///
template <Alphabet ALPHABET, typename Iterator>
NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
transform_iterator< Iterator, from_char_functor<ALPHABET> >
from_string(Iterator it)
{
    return make_transform_iterator( it, from_char_functor<ALPHABET>() );
}

///@} AlphabetsModule
///@} Strings

} // namespace nvbio

#include <nvbio/strings/alphabet_inl.h>