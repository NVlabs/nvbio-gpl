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
#include <nvbio/basic/numbers.h>
#include <thrust/device_vector.h>

namespace nvbio {
namespace cuda {

/// implements an inter-CTA synchronization primitive which can be called
/// multiple times from the same grid, or even across multiple kernel
/// launches, as long as all kernel launches have the same size.
///
/// ACHTUNG!
/// this primitive is NOT SAFE, and only works if all CTAs are resident!
///
struct syncblocks
{
    /// constructor
    ///
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    syncblocks(int32* counter = NULL);

    /// enact the syncblocks
    ///
    /// \return     true on successful completion, false otherwise
    ///
    NVBIO_FORCEINLINE NVBIO_DEVICE
    bool enact(const uint32 max_iter = uint32(-1));

    int32* m_counter;
};

/// a class to build a syncblocks primitive from the host
///
struct syncblocks_storage
{
    /// constructor
    ///
    syncblocks_storage();

    /// return a syncblocks object
    ///
    syncblocks get();

    /// clear the syncblocks, useful if one wants to reuse it
    /// across differently sized kernel launches.
    ///
    void clear();

private:
    thrust::device_vector<int32> m_counter;
};

} // namespace cuda
} // namespace nvbio

#include <nvbio/basic/cuda/syncblocks_inl.h>
