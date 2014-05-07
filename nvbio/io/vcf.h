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

#include <nvbio/basic/types.h>

#include <vector>
#include <string>

#pragma once

namespace nvbio {
namespace io {

struct SNPDatabase
{
    std::vector<std::string> chromosome;
    std::vector<uint64> position;
    std::vector<std::string> reference;
    std::vector<std::string> variant;
    std::vector<uint8> variant_quality;
};

// loads variant data from file_name and appends to output
bool loadVCF(SNPDatabase& output, const char *file_name);

} // namespace io
} // namespace nvbio
