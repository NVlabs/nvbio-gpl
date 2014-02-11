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

//
// NOTE: the code below is a derivative of bntseq.h, originally distributed
// under the MIT License, Copyright (c) 2008 Genome Research Ltd (GRL).
//

#pragma once

#include <nvbio/basic/types.h>
#include <string>
#include <vector>
#include <cstdio>

namespace nvbio {

struct BNTAnnData
{
    BNTAnnData() : offset(0), len(0), n_ambs(0), gi(0) {}

    int64   offset;
	int32   len;
	int32   n_ambs;
	uint32  gi;
};

struct BNTAnnInfo
{
    std::string name;
    std::string anno;
};

struct BNTAmb
{
	int64   offset;
	int32   len;
	char    amb;
};

struct BNTSeq
{
    BNTSeq() : l_pac(0), n_seqs(0), seed(0), n_holes(0) {}

    int64   l_pac;
	int32   n_seqs;
	uint32  seed;
	int32   n_holes;
    std::vector<BNTAnnInfo> anns_info;  // n_seqs elements
    std::vector<BNTAnnData> anns_data;  // n_seqs elements
    std::vector<BNTAmb>     ambs;       // n_holes elements
};

struct BNTInfo
{
    int64   l_pac;
	int32   n_seqs;
	uint32  seed;
	int32   n_holes;
};

struct BNTSeqLoader
{
    virtual void set_info(const BNTInfo info) {}
    virtual void read_ann(const BNTAnnInfo& info, BNTAnnData& data) {}
    virtual void read_amb(const BNTAmb& amb) {}
};

struct bns_fopen_failure {};
struct bns_files_mismatch {};

void save_bns(const BNTSeq& bns, const char *prefix);
void load_bns(BNTSeq& bns, const char *prefix);

void load_bns_info(BNTInfo& bns, const char *prefix);
void load_bns(BNTSeqLoader* bns, const char *prefix);

} // namespace nvbio
