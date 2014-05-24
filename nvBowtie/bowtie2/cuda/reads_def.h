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

#include <nvBowtie/bowtie2/cuda/defs.h>
#include <nvbio/io/sequence/sequence.h>
#include <nvbio/basic/cuda/ldg.h>

namespace nvbio {
namespace bowtie2 {
namespace cuda {

struct ReadsDef
{
    typedef typename binary_switch<uint32,uint4,USE_UINT4_PACKING>::type                                                     read_storage_type;
    typedef typename binary_switch<const read_storage_type*,nvbio::cuda::ldg_pointer<read_storage_type>,USE_TEX_READS>::type read_base_type;
    typedef typename binary_switch<const char*,             nvbio::cuda::ldg_pointer<char>,             USE_TEX_READS>::type read_qual_type;
    typedef io::SequenceDataViewCore<const uint32*,read_base_type,read_qual_type,const char*>                                read_view_type;
    typedef io::SequenceDataAccess<DNA_N,read_view_type>                                                                     type;
};

} // namespace cuda
} // namespace bowtie2
} // namespace nvbio
