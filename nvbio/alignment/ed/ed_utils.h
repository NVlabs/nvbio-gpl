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

namespace nvbio {
namespace aln {

namespace priv {

///@addtogroup private
///@{

///
/// An adapter implementation of the \ref SmithWatermanScoringScheme model
/// producing the edit distance scoring
///
struct EditDistanceSWScheme
{
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE int32 match(const uint8 q = 0)      const { return  0; };
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE int32 mismatch(const uint8 q = 0)   const { return -1; };
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE int32 mismatch(const uint8 a, const uint8 b, const uint8 q = 0)   const { return -1; };
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE int32 deletion()                    const { return -1; };
    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE int32 insertion()                   const { return -1; };
};

/// @} // end of private group

} // namespace priv

} // namespace aln
} // namespace nvbio

