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

#include <nvbio/io/sequence/sequence_priv.h>
#include <nvbio/io/sequence/sequence_encoder.h>

#include <nvbio/io/sequence/sequence_fastq.h>
#include <nvbio/io/sequence/sequence_txt.h>
#include <nvbio/io/sequence/sequence_sam.h>
#include <nvbio/io/sequence/sequence_bam.h>

namespace nvbio {
namespace io {

// grab the next batch of reads into a host memory buffer
int SequenceDataFile::next(SequenceDataEncoder* encoder, const uint32 batch_size, const uint32 batch_bps)
{
    const uint32 reads_to_load = std::min(m_max_reads - m_loaded, batch_size);

    if (!is_ok() || reads_to_load == 0)
        return NULL;

    // a default average read length used to reserve enough space
    const uint32 AVG_READ_LENGTH = 100;

    encoder->begin_batch();
    encoder->reserve(
        batch_size,
        batch_bps == uint32(-1) ? batch_size * AVG_READ_LENGTH : batch_bps ); // try to use a default read length

    // fetch the sequence info
    SequenceDataInfo* info = encoder->info();

    while (info->size() < reads_to_load &&
           info->bps()  < batch_bps)
    {
        // load 100 at a time if possible
        const uint32 chunk_reads = nvbio::min(reads_to_load - info->size(), uint32(100));
        const uint32 chunk_bps   = batch_bps - info->bps();

        const int n = nextChunk( encoder , chunk_reads, chunk_bps );
        assert(n <= (int) chunk_reads);
        if (n == 0)
            break;

        assert(info->size() <= reads_to_load);
    }

    if (info->size() == 0)
        return 0;

    m_loaded += info->size();

    encoder->end_batch();

    return info->size();
}


// factory method to open a read file, tries to detect file type based on file name
SequenceDataStream *open_sequence_file(
    const SequenceAlphabet   alphabet,
    const char *             sequence_file_name,
    const QualityEncoding    qualities,
    const uint32             max_seqs,
    const uint32             max_sequence_len,
    const SequenceEncoding   flags)
{
    // parse out file extension; look for .fastq.gz, .fastq suffixes
    uint32 len = uint32( strlen(sequence_file_name) );
    bool is_gzipped = false;

    // do we have a .gz suffix?
    if (len >= strlen(".gz"))
    {
        if (strcmp(&sequence_file_name[len - strlen(".gz")], ".gz") == 0)
        {
            is_gzipped = true;
            len = uint32(len - strlen(".gz"));
        }
    }

    // check for fastq suffix
    if (len >= strlen(".fastq"))
    {
        if (strncmp(&sequence_file_name[len - strlen(".fastq")], ".fastq", strlen(".fastq")) == 0)
        {
            return new SequenceDataFile_FASTQ_gz(
                sequence_file_name,
                qualities,
                max_seqs,
                max_sequence_len,
                flags);
        }
    }

    // check for fastq suffix
    if (len >= strlen(".fq"))
    {
        if (strncmp(&sequence_file_name[len - strlen(".fq")], ".fq", strlen(".fq")) == 0)
        {
            return new SequenceDataFile_FASTQ_gz(
                sequence_file_name,
                qualities,
                max_seqs,
                max_sequence_len,
                flags);
        }
    }

    // check for txt suffix
    if (len >= strlen(".txt"))
    {
        if (strncmp(&sequence_file_name[len - strlen(".txt")], ".txt", strlen(".txt")) == 0)
        {
            return new SequenceDataFile_TXT_gz(
                sequence_file_name,
                qualities,
                max_seqs,
                max_sequence_len,
                flags);
        }
    }

    // check for sam suffix
    if (len >= strlen(".sam"))
    {
        if (strncmp(&sequence_file_name[len - strlen(".sam")], ".sam", strlen(".sam")) == 0)
        {
            SequenceDataFile_SAM *ret;

            ret = new SequenceDataFile_SAM(
                sequence_file_name,
                max_seqs,
                max_sequence_len,
                flags);

            if (ret->init() == false)
            {
                delete ret;
                return NULL;
            }

            return ret;
        }
    }

    // check for bam suffix
    if (len >= strlen(".bam"))
    {
        if (strncmp(&sequence_file_name[len - strlen(".bam")], ".bam", strlen(".bam")) == 0)
        {
            SequenceDataFile_BAM *ret;

            ret = new SequenceDataFile_BAM(
                sequence_file_name,
                max_seqs,
                max_sequence_len,
                flags);

            if (ret->init() == false)
            {
                delete ret;
                return NULL;
            }

            return ret;
        }
    }

    // we don't actually know what this is; guess fastq
    log_warning(stderr, "could not determine file type for %s; guessing %sfastq\n", sequence_file_name, is_gzipped ? "compressed " : "");
    return new SequenceDataFile_FASTQ_gz(
        sequence_file_name,
        qualities,
        max_seqs,
        max_sequence_len,
        flags);
}

} // namespace io
} // namespace nvbio
