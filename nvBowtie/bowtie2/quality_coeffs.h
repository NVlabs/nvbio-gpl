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
namespace bowtie2 {

/// map Phred quality scores into their maq equivalents
///
extern unsigned char s_phred_to_maq[];

/// map Solexa quality values to Phred scale
///
extern unsigned char s_solexa_to_phred[];

NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
char phred_to_maq(int q)
{
    return
        q < 5 ?
            0 :
        (q < 15) ?
            10 :
        (q < 25) ?
            20 :
            30;
}

} // namespace bowtie2
} // namespace nvbio
