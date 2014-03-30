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

#include <nvbio/sufsort/sufsort_utils.h>

namespace nvbio {

///@addtogroup Sufsort
///@{

/// open a string-set BWT file, returning a handler that can be used by the string-set BWT
/// construction functions.
///
/// The file type is specified by the extension of the output name; the following extensions
/// are supported:
///
/// .txt        ASCII
/// .txt.gz     ASCII, gzip compressed
/// .txt.bgz    ASCII, block-gzip compressed
/// .bwt        2-bit packed binary
/// .bwt.gz     2-bit packed binary, gzip compressed
/// .bwt.bgz    2-bit packed binary, block-gzip compressed
/// .bwt4       4-bit packed binary
/// .bwt4.gz    4-bit packed binary, gzip compressed
/// .bwt4.bgz   4-bit packed binary, block-gzip compressed
///
/// Alongside with the main BWT file, a file containing the mapping between the primary
/// dollar tokens and their position in the BWT will be generated. This (.pri|.pri.gz|.pri.bgz)
/// file is a plain list of (position,string-id) pairs, either in ASCII or binary form.
/// The ASCII file has the form:
///\verbatim
///#PRI
///position[1] string[1]
///...
///position[n] string[n]
///\endverbatim
///
/// The binary file has the form:
///\verbatim
///char[4] header = "PRIB";
///struct { uint64 position; uint32 string_id; } pairs[n];
///\endverbatim
///
/// \param output_name      output name
/// \param params           additional compression parameters (e.g. "1R", "9", etc)
/// \return     a handler that can be used by the string-set BWT construction functions
///
BaseBWTHandler* open_bwt_file(const char* output_name, const char* params);

///@}

} // namespace nvbio
