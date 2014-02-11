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

/*! \file cached_iterator.h
 *   \brief CUDA-compatible iterator wrappers allowing to cache the dereferenced
 *   value of generic iterators
 */

#pragma once

#include <nvbio/basic/types.h>

namespace nvbio
{

struct cuda_error
{
    cuda_error(const char* format, ...);

    const char* what() const { return s_error; }

private:
    static char s_error[4096];
};

struct bad_alloc
{
    bad_alloc(const char* format, ...);

    const char* what() const { return s_error; }

private:
    static char s_error[4096];
};
struct runtime_error
{
    runtime_error(const char* format, ...);

    const char* what() const { return s_error; }

private:
    static char s_error[4096];
};
struct logic_error
{
    logic_error(const char* format, ...);

    const char* what() const { return s_error; }

private:
    static char s_error[4096];
};

} // namespace nvbio
