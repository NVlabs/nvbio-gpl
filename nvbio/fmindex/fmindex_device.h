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

#include <nvbio/fmindex/fmindex.h>

namespace nvbio {

///
/// helper class to hold fm index device data
///
template <uint32 OCC_INT>
struct fm_index_device_data
{
    /// constructor
    ///
    fm_index_device_data(
        const uint32  len,
        const uint32* bwt,
        const uint32* occ,
        const uint32* L2)
    {
        const uint32 WORDS = (len+16)/16;

        cudaMalloc( &m_L2,   sizeof(uint32)*5 );
        cudaMalloc( &m_bwt,  sizeof(uint32)*WORDS );
        cudaMalloc( &m_occ,  uint64(sizeof(uint32))*4*uint64(len+OCC_INT-1)/OCC_INT );

        cudaMemcpy( m_L2,   L2,   sizeof(uint32)*5,                                         cudaMemcpyHostToDevice );
        cudaMemcpy( m_occ,  occ,  uint64(sizeof(uint32))*4*uint64(len+OCC_INT-1)/OCC_INT,   cudaMemcpyHostToDevice );
        cudaMemcpy( m_bwt,  bwt,  sizeof(uint32)*WORDS,                                     cudaMemcpyHostToDevice );
    }
    /// destructor
    ////
    ~fm_index_device_data()
    {
        cudaFree( m_L2 );
        cudaFree( m_bwt );
        cudaFree( m_occ );
    }

    uint32* m_bwt;
    uint32* m_occ;
    uint32* m_L2;
};

} // namespace nvbio
