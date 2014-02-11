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
#include <string>

namespace nvbio {

template <typename options_type>
uint32 uint_option(const options_type& options, const char* name, const uint32 val)
{
    return (options.find( std::string(name) ) != options.end()) ?
        atoi( options.find(std::string(name))->second.c_str() ) :
        val;
}

template <typename options_type>
int32 int_option(const options_type& options, const char* name, const int32 val)
{
    return (options.find( std::string(name) ) != options.end()) ?
        atoi( options.find(std::string(name))->second.c_str() ) :
        val;
}

template <typename options_type>
int64 int64_option(const options_type& options, const char* name, const int64 val)
{
    return (options.find( std::string(name) ) != options.end()) ?
        atoi( options.find(std::string(name))->second.c_str() ) :
        val;
}

template <typename options_type>
float float_option(const options_type& options, const char* name, const float val)
{
    return (options.find( std::string(name) ) != options.end()) ?
        (float)atof( options.find(std::string(name))->second.c_str() ) :
        val;
}

template <typename options_type>
std::string string_option(const options_type& options, const char* name, const char* val)
{
    return (options.find( std::string(name) ) != options.end()) ?
        options.find(std::string(name))->second :
        std::string( val );
}

} // namespace nvbio
