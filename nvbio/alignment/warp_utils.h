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
namespace aln {
namespace priv {

// helper structure to hold a score + sink and allow for comparisons during a warp-scan
template<typename score_type>
struct alignment_result
{
    score_type score;
    uint2 sink;

    __device__ alignment_result(score_type score, uint2 sink)
    {
        this->score = score;
        this->sink = sink;
    };

    __device__ alignment_result()
    {
    };

    __device__ alignment_result(const volatile alignment_result<score_type>& other)
    {
        *this = other;
    };

    __device__ alignment_result(const alignment_result<score_type>& other)
    {
        *this = other;
    };

    __device__ alignment_result<score_type>& operator=(const alignment_result<score_type>& other)
    {
        score = other.score;
        sink.x = other.sink.x;
        sink.y = other.sink.y;
        return *this;
    };

    __device__ volatile alignment_result<score_type>& operator=(const volatile alignment_result<score_type>& other) volatile
    {
        score = other.score;
        sink.x = other.sink.x;
        sink.y = other.sink.y;
        return *this;
    };

    __device__ static alignment_result<score_type> minimum_value()
    {
        alignment_result<score_type> ret;
        ret.score = Field_traits<score_type>::min();
        return ret;
    };

    // comparison functor
    struct max_operator
    {
        __device__ const volatile alignment_result<score_type>& operator() (const alignment_result<score_type>& s1,
                                                                            const volatile alignment_result<score_type>& s2) const
        {
            if (s1.score > s2.score)
            {
                return s1;
            } else {
                return s2;
            }
        }
    };
};

} // namespace priv
} // namespace aln
} // namespace nvbio
