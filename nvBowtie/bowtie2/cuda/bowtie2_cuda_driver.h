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

#include <nvbio/io/fmi.h>
#include <nvbio/io/sequence/sequence.h>
#include <map>
#include <string>

namespace nvbio {
namespace bowtie2 {
namespace cuda {

int driver(const char*                              output_name,
           const io::FMIndexData&                   driver_data,
                 io::SequenceDataStream&            read_data_stream,
           const std::map<std::string,std::string>& options);

int driver(const char*                              output_name,
           const io::FMIndexData&                   driver_data,
           const io::PairedEndPolicy                pe_policy,
                 io::SequenceDataStream&            read_data_stream1,
                 io::SequenceDataStream&            read_data_stream2,
           const std::map<std::string,std::string>& options);

} // namespace cuda
} // namespace bowtie2
} // namespace nvbio
