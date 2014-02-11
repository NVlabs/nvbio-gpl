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
#include <nvbio/io/fmi.h>

namespace nvbio {
namespace bowtie2 {
namespace cuda {

struct FMIndexDef
{
    typedef TEX_SELECTOR( io::FMIndexIterators, io::FMIndexLdgIterators )   driver_data_iterators;
    typedef driver_data_iterators::count_table_type                         count_table_type;
    typedef driver_data_iterators::occ_type                                 occ_type;
    typedef driver_data_iterators::bwt_type                                 bwt_type;
    typedef driver_data_iterators::ssa_type                                 ssa_type;
    typedef driver_data_iterators::rank_dict_type                           rank_dict_type;

    typedef fm_index<rank_dict_type,ssa_type>                               type;
};

} // namespace cuda
} // namespace bowtie2
} // namespace nvbio
