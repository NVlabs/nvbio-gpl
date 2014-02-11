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

///
///\file seed_hit.h
///

#pragma once

#include <nvBowtie/bowtie2/cuda/defs.h>
#include <nvbio/basic/cuda/arch.h>
#include <nvbio/basic/pod.h>
#include <nvbio/io/utils.h>
#include <algorithm>

namespace nvbio {
namespace bowtie2 {
namespace cuda {

///@addtogroup nvBowtie
///@{

///@addtogroup SeedHits
///@{

///
/// Represents a seed match in the FM-index as an SA range, tracking the direction of the read
/// and that of the FM-index.
///
struct SeedHit
{
    struct Flags { uint32 m_pos:10, m_rc:1, m_indexdir:1; };

    /// build the compressed seed flags for a hit
    ///
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    static Flags build_flags(
        const ReadType readtype, 
        const DirType  indexdir,
        const uint32   pos)
    {
        Flags flags;
        flags.m_pos      = pos;
        flags.m_rc       = readtype;
        flags.m_indexdir = indexdir;
        return flags;
    }

    /// constructor
    ///
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    SeedHit() {}

    /// constructor
    ///
    /// \param readtype     select between forward and reverse-complemented read
    /// \param indexdir     direction in the FM-index
    /// \param pos          seed position within the read
    /// \param range        SA range
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    SeedHit(const ReadType  readtype, 
            const DirType   indexdir,
            const uint32    pos,
            const uint2&    range)
            : m_range_begin( range.x ),
              m_range_delta( range.y - range.x ),
              m_pos( pos ),
              m_rc( readtype ),
              m_indexdir( indexdir )
    {
        NVBIO_CUDA_ASSERT( range.x <= range.y );
        NVBIO_CUDA_ASSERT( range.y - range.x < (1u << 20) );
    }

    /// constructor
    ///
    /// \param flags        compressed seed flags
    /// \param range        SA range
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    SeedHit(const Flags  flags,
            const uint2& range)
            : m_range_begin( range.x ),
              m_range_delta( range.y - range.x ),
              m_pos( flags.m_pos ),
              m_rc( flags.m_rc ),
              m_indexdir( flags.m_indexdir )
    {
        NVBIO_CUDA_ASSERT( range.x <= range.y );
        NVBIO_CUDA_ASSERT( range.y - range.x < (1u << 20) );
    }

    /// set the SA range
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    void set_range(const uint2 range)
    {
        NVBIO_CUDA_ASSERT( range.x <= range.y );
        NVBIO_CUDA_ASSERT( range.y - range.x < (1u << 20) );
        m_range_begin = range.x;
        m_range_delta = range.y - range.x;
    }

    /// get the SA range
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint2 get_range() const
    { return make_uint2( m_range_begin, m_range_begin + m_range_delta ); }

    /// get the size of the SA range
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 get_range_size() const
    { return m_range_delta; }

    /// return the front of the SA range
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 front() const { return m_range_begin; }

    /// return the back of the SA range
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 back() const { return m_range_begin + m_range_delta - 1u; }

    /// pop the front of the SA range
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 pop_front()
    {
        const uint32 r = m_range_begin++;
        --m_range_delta;
        return r;
    }

    /// pop n-elements from the front of the SA range
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 pop_front(const uint32 n)
    {
        const uint32 r = m_range_begin;
        m_range_begin += n;
        m_range_delta -= n;
        return r;
    }

    /// pop the back of the SA range
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 pop_back()
    {
        --m_range_delta;
        return m_range_begin + m_range_delta;
    }

    /// pop n-elements from the back of the SA range
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 pop_back(const uint32 n)
    {
        m_range_delta -= n;
        return m_range_begin + m_range_delta;
    }

    /// set the seed position in the read
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    void set_posinread(const uint32 p)
    { m_pos = p; }
    
    /// set the read flags
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    void set_readtype(const ReadType d)
    { m_rc = d; }
    
    /// set the FM-index orientation
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    void set_indexdir(const DirType d)
    { m_indexdir = d; }

    /// get the seed position in the read
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    uint32 get_posinread() const
    { return m_pos; }
    
    /// get the read flags
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    ReadType get_readtype() const
    { return ReadType( m_rc ); }
        
    /// get the FM-index orientation
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    DirType get_indexdir() const
    { return DirType( m_indexdir ); }

private:
    uint32 m_range_begin;
    uint32 m_range_delta:20, m_pos:10, m_rc:1, m_indexdir:1;
};

struct hit_compare
{
    NVBIO_HOST_DEVICE NVBIO_FORCEINLINE
    bool operator() (const SeedHit& f, const SeedHit& s)
    {
        const uint32 size_f = f.get_range_size();
        const uint32 size_s = s.get_range_size();
        return (size_f > size_s);
    }
};

///@}  // group SeedHits
///@}  // group nvBowtie

} // namespace cuda
} // namespace bowtie2
} // namespace nvbio
