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
#include <string.h>
#include <string>

namespace nvbio {
namespace bowtie2 {
namespace cuda {

///@addtogroup nvBowtie
///@{

enum MappingMode {
    BestMappingApprox = 0,
    BestMappingExact  = 1,
    AllMapping        = 2
};

enum ScoringMode {
    EditDistanceMode  = 0,
    SmithWatermanMode = 1,
};

enum AlignmentType {
    EndToEndAlignment = 0,
    LocalAlignment    = 1,
};

static const char* s_mapping_mode[] = {
    "best",
    "best-exact",
    "all"
};
inline const char* mapping_mode(const uint32 mode)
{
    return s_mapping_mode[ mode ];
}

inline uint32 mapping_mode(const char* str)
{
    if (strcmp( str, "best" ) == 0)
        return BestMappingApprox;
    else if (strcmp( str, "best-exact" ) == 0)
        return BestMappingExact;
    else
        return AllMapping;
}

static const char* s_scoring_mode[] = {
    "ed",
    "sw"
};
inline const char* scoring_mode(const uint32 mode)
{
    return s_scoring_mode[ mode ];
}

inline uint32 scoring_mode(const char* str)
{
    if (strcmp( str, "ed" ) == 0)
        return EditDistanceMode;
    else if (strcmp( str, "sw" ) == 0)
        return SmithWatermanMode;
    else
        return EditDistanceMode;
}

struct SimpleFunc
{
    enum Type { LinearFunc = 0, LogFunc = 1, SqrtFunc = 2 };

    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    SimpleFunc(const Type _type = LinearFunc, const float _k = 0.0f, const float _m = 1.0f) : type(_type), k(_k), m(_m) {}

    NVBIO_FORCEINLINE NVBIO_HOST_DEVICE
    int32 operator() (const int32 x) const
    {
        return int32( k + m * (type == LogFunc  ?  logf(float(x)) :
                               type == SqrtFunc ? sqrtf(float(x)) :
                                                        float(x)) );
    }

    const char* type_string() const
    {
        return type == LogFunc  ? "log" :
               type == SqrtFunc ? "sqrt" :
                                  "linear";
    }
    const char* type_symbol() const
    {
        return type == LogFunc  ? "G" :
               type == SqrtFunc ? "S" :
                                  "L";
    }

    Type  type;
    float k;
    float m;
};

///
/// A POD structure holding all of nvBowtie's parameters
///
struct ParamsPOD
{
    bool          keep_stats;
    bool          randomized;
    uint32        mode;
    uint32        scoring_mode;
    uint32        alignment_type;
    uint32        top_seed;
    uint32        seed_len;
    SimpleFunc    seed_freq;
    uint32        max_hits;
    uint32        max_dist;
    uint32        max_effort_init;
    uint32        max_effort;
    uint32        min_ext;
    uint32        max_ext;
    uint32        max_reseed;
    uint32        rep_seeds;
    uint32        allow_sub;
    uint32        subseed_len;
    uint32        mapq_filter;
    uint32        min_read_len;

    // paired-end options
    uint32        pe_policy;
    bool          pe_overlap;
    bool          pe_dovetail;
    bool          pe_unpaired;
    uint32        min_frag_len;
    uint32        max_frag_len;

    // Internal fields
    uint32        scoring_window;
    DebugState    debug;
};

///
/// A non-POD structure holding all of nvBowtie's parameters plus a few extra string options
///
struct Params : public ParamsPOD
{
    std::string   report;
    std::string   scoring_file;

    int32         persist_batch;
    int32         persist_seeding;
    int32         persist_extension;
    std::string   persist_file;
};

///@}  // group nvBowtie

} // namespace cuda
} // namespace bowtie2
} // namespace nvbio
