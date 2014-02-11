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

namespace nvbio {
namespace io {

// check whether two alignments are distinct
//
NVBIO_FORCEINLINE NVBIO_DEVICE bool distinct_alignments(
    const uint32 pos1,
    const bool   rc1,
    const uint32 pos2,
    const bool   rc2,
    const uint32 dist)
{
    return rc1 != rc2 ? 
        true :
            pos1 >= pos2 - nvbio::min( pos2, dist ) &&
            pos1 <= pos2 + dist ? false :
                                  true;
}

// check whether two paired-end alignments are distinct
//
NVBIO_FORCEINLINE NVBIO_DEVICE bool distinct_alignments(
    const uint32 apos1,
    const uint32 opos1,
    const bool   arc1,
    const bool   orc1,
    const uint32 apos2,
    const uint32 opos2,
    const bool   arc2,
    const bool   orc2)
{
    return (arc1 != arc2) || (orc1 != orc2) || (apos1 != apos2) || (opos1 != opos2);
}

// check whether two paired-end alignments are distinct
//
NVBIO_FORCEINLINE NVBIO_DEVICE bool distinct_alignments(
    const uint32 apos1,
    const uint32 opos1,
    const bool   arc1,
    const bool   orc1,
    const uint32 apos2,
    const uint32 opos2,
    const bool   arc2,
    const bool   orc2,
    const uint32 dist)
{
    return arc1 != arc2 || orc1 != orc2 ? 
        true :
            (apos1 >= apos2 - nvbio::min( apos2, dist ) &&  apos1 <= apos2 + dist) &&
            (opos1 >= opos2 - nvbio::min( opos2, dist ) &&  opos1 <= opos2 + dist) ?
                false :
                true;
}

// check whether two paired-end alignments are distinct
//
NVBIO_FORCEINLINE NVBIO_DEVICE bool distinct_alignments(
    const PairedAlignments& p1,
    const PairedAlignments& p2)
{
    return distinct_alignments(
        p1.mate(0).alignment() + p1.mate(0).sink(),
        p1.mate(1).alignment() + p1.mate(1).sink(),
        p1.mate(0).is_rc(),
        p1.mate(1).is_rc(),
        p2.mate(0).alignment() + p2.mate(0).sink(),
        p2.mate(1).alignment() + p2.mate(1).sink(),
        p2.mate(0).is_rc(),
        p2.mate(1).is_rc() );
}
// check whether two paired-end alignments are distinct
//
NVBIO_FORCEINLINE NVBIO_DEVICE bool distinct_alignments(
    const PairedAlignments& p1,
    const PairedAlignments& p2,
    const uint32            dist)
{
    return distinct_alignments(
        p1.mate(0).alignment() + p1.mate(0).sink(),
        p1.mate(1).alignment() + p1.mate(1).sink(),
        p1.mate(0).is_rc(),
        p1.mate(1).is_rc(),
        p2.mate(0).alignment() + p2.mate(0).sink(),
        p2.mate(1).alignment() + p2.mate(1).sink(),
        p2.mate(0).is_rc(),
        p2.mate(1).is_rc(),
        dist );
}

} // namespace io
} // namespace nvbio
