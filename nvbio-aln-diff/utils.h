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
#include <nvbio/basic/numbers.h>
#include <nvbio/basic/console.h>
#include <string>

namespace nvbio {
namespace alndiff {

struct BooleanStats
{
    BooleanStats() : L(0), R(0), L_not_R(0), R_not_L(0), L_and_R(0), n(0) {}

    void push(const bool l, const bool r)
    {
        L       += (l == true) ? 1u : 0u;
        R       += (r == true) ? 1u : 0u;
        L_not_R += (l == true) && (r == false) ? 1u : 0u;
        R_not_L += (r == true) && (l == false) ? 1u : 0u;
        L_and_R += (l == true) && (r == true)  ? 1u : 0u;
        ++n;
    }

    float avg_L() const { return n ? float(L) / float(n) : 0.0f; }
    float avg_R() const { return n ? float(R) / float(n) : 0.0f; }
    float avg_L_not_R() const { return n ? float(L_not_R) / float(n) : 0.0f; }
    float avg_R_not_L() const { return n ? float(R_not_L) / float(n) : 0.0f; }
    float avg_L_and_R() const { return n ? float(L_and_R) / float(n) : 0.0f; }

    uint32 L;
    uint32 R;
    uint32 L_not_R;
    uint32 R_not_L;
    uint32 L_and_R;
    uint32 n;
};

template <uint32 X>
struct Histogram
{
    Histogram() : count(0)
    {
        for (uint32 i = 0; i < 2*X; ++i)
            bins[i] = 0;
    }

    uint32 all_but(const uint32 i) const { return count - bins[i+X]; }

    uint32 operator[] (const int32 i) const { return bins[i + X]; }

    void push(const int32 i)
    {
        const int32 bin = nvbio::min(nvbio::max(int32(i + X),0),int32(2*X-1));
        ++bins[bin];
        ++count;
    }

    uint32  count;
    uint32  bins[2*X];
};

// accumulate histogram values bottom-up
template <uint32 X>
Histogram<X> cumulative(const Histogram<X>& I)
{
    Histogram<X> H;
    H.count   = I.count;
    H.bins[0] = I.bins[0];
    for (int32 i = 1; i < 2*X; ++i)
        H.bins[i] = I.bins[i] + H.bins[i-1];
    return H;
}
// accumulate histogram values top-down
template <uint32 X>
Histogram<X> reverse_cumulative(const Histogram<X>& I)
{
    Histogram<X> H;
    H.count       = I.count;
    H.bins[2*X-1] = I.bins[2*X-1];
    for (int32 i = 2*X-2; i >= 0; --i)
        H.bins[i] = I.bins[i] + H.bins[i+1];
    return H;
}

template <uint32 X, uint32 Y>
struct Histogram2d
{
    Histogram2d() : count(0)
    {
        for (uint32 i = 0; i < 2*X; ++i)
            for (uint32 j = 0; j < 2*Y; ++j)
                bins[i][j] = 0;
    }

    void push(const int32 x, const int32 y)
    {
        const int32 bin_x = nvbio::min(nvbio::max(int32(x + X),0),int32(2*X-1));
        const int32 bin_y = nvbio::min(nvbio::max(int32(y + Y),0),int32(2*Y-1));
        ++bins[bin_x][bin_y];
        ++count;
    }
    uint32 operator() (const int32 i, const int32 j) const { return bins[i + X][j + Y]; }

    uint32  count;
    uint32  bins[2*X][2*Y];
};

inline
uint32 read_length_bin_range(const uint32 bin)
{
    switch (bin)
    {
    case 0:
        return 16;
    case 1:
        return 36;
    case 2:
        return 100;
    case 3:
        return 150;
    case 4:
        return 200;
    case 5:
        return 250;
    case 6:
        return 300;
    case 7:
        return 350;
    case 8:
        return 400;
    case 9:
        return 450;
    case 10:
        return 500;
    default:
        return 1000;
    }
}

inline
uint32 read_length_bin(const uint32 read_len)
{
    if (read_len <= 16)
        return 0;
    else if (read_len <= 36)
        return 1;
    else if (read_len <= 100)
        return 2;
    else if (read_len <= 150)
        return 3;
    else if (read_len <= 200)
        return 4;
    else if (read_len <= 250)
        return 5;
    else if (read_len <= 300)
        return 6;
    else if (read_len <= 350)
        return 7;
    else if (read_len <= 400)
        return 8;
    else if (read_len <= 450)
        return 9;
    else if (read_len <= 500)
        return 10;
    else
        return 11;
}
inline int32 log_bin(const int32 x)
{
    return x == 0 ? 0u :
           x  < 0 ? -int32(1u + nvbio::log2(-x)) :
                     int32(1u + nvbio::log2(x));
}
inline int32 log_bin_range(const int32 bin)
{
    return bin == 0 ? 0 :
           bin  < 0 ? -int32(1u << (-bin+1)) :
                       int32(1u <<  (bin-1));
}

// return the local file name from a path
//
inline const char* local_file(const std::string& file_name)
{
  #if WIN32
    const size_t pos = file_name.find_last_of("/\\");
  #else
    const size_t pos = file_name.rfind('/');
  #endif

    if (pos == std::string::npos)
        return file_name.c_str();
    else
        return file_name.c_str() + pos + 1u;
}

} // namespace alndiff
} // namespace nvbio
