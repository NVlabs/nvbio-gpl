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

#include <nvbio/basic/cuda/arch.h>
#include <nvbio/basic/packedstream.h>
#include <nvbio/basic/packedstream_loader.h>
#include <nvbio/basic/vector_wrapper.h>

#pragma once

namespace nvbio {

///@addtogroup IO
///@{

///@addtogroup Utils
///@{

enum ReadType { STANDARD = 0u, COMPLEMENT = 1u };
enum DirType  { FORWARD  = 0u, REVERSE    = 1u };

///
/// A utility functor to reverse a given stream
///
template<typename IndexType>
struct ReverseXform
{
    typedef IndexType index_type;
    const index_type pos;

    /// empty constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    ReverseXform() : pos(0) { }

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    ReverseXform(const index_type n) : pos(n-1) { }

    /// functor operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    index_type operator() (const index_type i) const { return pos-i; }
};

///
/// A utility functor to apply an offset to a given a stream
///
template<typename IndexType>
struct OffsetXform
{
    typedef IndexType index_type;
    const index_type pos;

    /// empty constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    OffsetXform() : pos(0) { }

    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    OffsetXform(const index_type n) : pos(n) { }

    /// functor operator
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    index_type operator() (const index_type i) const { return pos+i; }
};

struct quality_nop {};

///
/// Helper class to extract qualities from a read
///
template <typename ReadStreamType>
struct ReadStreamQualities
{
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    ReadStreamQualities() : m_read( NULL ) {}

    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    ReadStreamQualities(const ReadStreamType& read) : m_read( &read ) {}

    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    uint8 operator[] (const uint32 pos) const { return m_read->quality(pos); }

    const ReadStreamType* m_read;
};

///
/// Helper class to represent a read or its reverse-complement - useful to avoid warp-divergence.
///
template< typename StreamType, typename QualType = quality_nop >
struct ReadStream
{
    typedef typename StreamType::symbol_type value_type;
    typedef ReadStream<StreamType,QualType>  this_type;
    typedef ReadStreamQualities<this_type>   qual_string_type; 

    /// default constructor
    ///
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    ReadStream() {}

    /// constructor
    ///
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    ReadStream(const StreamType& s, const uint2 range) 
      : stream(s), first(range.y-1), last(range.x) { }

    /// constructor
    ///
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    ReadStream(const StreamType& s, const QualType q, const uint2 range) 
      : stream(s), qual(q), first(range.y-1), last(range.x) { }

    /// set the read flags
    ///
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    void set_flags(ReadType t) { rev_comp = (t == COMPLEMENT); }

    /// return string length
    ///
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    uint32 length() const { return 1+first-last; }

    /// return a given base
    ///
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    value_type operator[] (uint32 pos) const
    {
        if (rev_comp) {
            value_type c = stream[last+pos];
            return c<4 ? 3-c : c;
        }
        return stream[first-pos];
    }

    /// return a given base quality
    ///
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    uint8 quality(const uint32 pos) const
    {
        return rev_comp ? qual[last+pos] : qual[first-pos];
    }

    /// return qualities
    ///
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    qual_string_type qualities() const { return qual_string_type(*this); }

    bool        rev_comp;           ///< rev.comp. flag
    uint32      first, last;        ///< offset of first and last elements
    StreamType  stream;             ///< read stream
    QualType    qual;               ///< quality stream
};

///
/// Utility class to load a read with a StringLoader.
/// The Tag type allows to specify the caching policy.
///
template <typename BatchType, typename Tag>
struct ReadLoader
{
    typedef typename BatchType::read_iterator                                                      read_storage;
    typedef typename BatchType::qual_iterator                                                      qual_iterator;
    typedef PackedStringLoader<read_storage, io::ReadData::READ_BITS, io::ReadData::HI_BITS,Tag>   loader_type;
    typedef typename loader_type::iterator                                                         read_iterator;
    typedef ReadStream<read_iterator,qual_iterator>                                                string_type;

    /// load a read
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    string_type load(const BatchType& batch, const uint2 range, bool rc)
    {
        const qual_iterator quals( batch.qual_stream() + range.x );

        string_type read(
            loader.load( batch.read_stream(), range.x, range.y - range.x ),
            quals,
            make_uint2( 0, range.y - range.x ) );

        read.set_flags( ReadType(rc) );
        return read;
    }
    /// load a read substring
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    string_type load(const BatchType& batch, const uint2 range, bool rc, const uint2 subrange)
    {
        const qual_iterator quals( batch.qual_stream() + range.x );

        string_type read(
            loader.load( batch.read_stream(), range.x, range.y - range.x, subrange, !rc ), // FIXME: negate the RC flag here as
            quals,                                                                         // reads are stored in reverse fashion...
            make_uint2( 0, range.y - range.x ) );

        read.set_flags( ReadType(rc) );
        return read;
    }

    loader_type loader;
};

///
/// Utility class to load a genome substring with a StringLoader.
/// The Tag type allows to specify the caching policy.
///
template <typename GenomeStorage, typename Tag>
struct GenomeLoader
{
    typedef PackedStringLoader<GenomeStorage,2,true,Tag>    loader_type;
    typedef typename loader_type::iterator                  iterator;
    typedef vector_wrapper<iterator>                        string_type;

    /// load a genome segment
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    string_type load(const GenomeStorage ptr, const uint32 offset, const uint32 len)
    {
        return string_type( len, loader.load( ptr, offset, len ) );
    }
    /// load a substring of a genome segment
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    string_type load(const GenomeStorage ptr, const uint32 offset, const uint32 len, const uint2 subrange)
    {
        return string_type( len, loader.load( ptr, offset, len, subrange, STANDARD ) );
    }

    loader_type loader;
};

///@} // Utils
///@} // IO

} // namespace nvbio
