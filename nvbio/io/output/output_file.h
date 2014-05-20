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

#include <nvbio/io/output/output_types.h>
#include <nvbio/io/output/output_stats.h>
#include <nvbio/io/fmi.h>
#include <nvbio/io/sequence/sequence.h>

#include <stdio.h>

namespace nvbio {
namespace io {

/**
   @addtogroup IO
   @{
   @addtogroup Output
   @{
*/

/**
   The output file interface.

   This class is the base class for all file output classes. By itself, it
   implements a "null" output, which discards all data sent into it. Other
   classes derive from this to implement various file output formats.

   After each alignment pass in the pipeline, the aligner should call
   OutputFile::process. This call takes a handle to the alignment buffers in
   GPU memory (a GPUOutputBatch structure). The implementation ensures that
   any data required is copied out before the call returns, allowing the
   caller to invalidate the GPU memory immediately.

   Sequence alignment is generally done in batches of reads; for each batch multiple
   OutputFile::process calls can be made. Two extra enum parameters serve to
   identify which alignment pass we're handling at each call. Each batch
   should be preceded by a OutputFile::start_batch call; at the end of each
   batch, OutputFile::end_batch should be called. In most cases, data is only
   written to disk after OutputFile::end_batch is called.

   The factory method OutputFile::open is used to create OutputFile
   objects. It parses the file name extension to determine the file format for
   the output.
*/
struct OutputFile
{

protected:
    OutputFile(const char *file_name, AlignmentType alignment_type, BNT bnt);

public:
    virtual ~OutputFile();

    /// Configure the MapQ evaluator. Must be called prior to any batch processing.
    virtual void configure_mapq_evaluator(const io::MapQEvaluator *mapq,
                                          int mapq_filter);

    /// Begin a new batch of alignment results
    /// \param read_data_1 The (host-side) read data pointer for the first mate
    /// \param read_data_2 The (host-side) read data pointer for the second mate, if any (can be NULL for single-end alignment)
    virtual void start_batch(const io::SequenceDataHost *read_data_1,
                             const io::SequenceDataHost *read_data_2 = NULL);

    /// Process a set of alignment results for the current batch.
    /// \param gpu_batch Handle to the GPU buffers containing the alignment results
    /// \param alignment_mate Identifies the mate for this pass
    /// \param alignment_score Identifies the score type being calculated in this pass (currently either best or second-best)
    virtual void process(struct GPUOutputBatch& gpu_batch,
                         const AlignmentMate alignment_mate,
                         const AlignmentScore alignment_score);

    /// Mark a batch of alignment results as complete
    virtual void end_batch(void);

    /// Flush and close the output file
    virtual void close(void);

    /// Returns aggregate I/O statistics for this object
    virtual IOStats& get_aggregate_statistics(void);

protected:
    /// Read back batch data into the host
    /// \param [out] cpu_batch The CPUOutputBatch struct which will receive the data
    /// \param [in] gpu_batch The GPU memory handle to read from
    /// \param [in] alignment_mate Identifies the mate that the data in gpu_batch refers to
    /// \param [in] alignment_score The type of score data in gpu_batch
    void readback(struct CPUOutputBatch& cpu_batch,
                  const struct GPUOutputBatch& gpu_batch,
                  const AlignmentMate alignment_mate,
                  const AlignmentScore alignment_score);

    /// Name of the file we're writing
    const char *file_name;
    /// The type of alignment we're running (single or paired-end)
    AlignmentType alignment_type;
    /// Reference genome data handle
    BNT bnt;

    /// The MapQ evaluator struct used to compute mapping qualities during output
    const io::MapQEvaluator *mapq_evaluator;
    /// The current mapping quality filter: reads with a mapq below this value will be marked as not aligned
    int mapq_filter;

    /// Host-side copies of the read data for the current batch.
    /// These are set by start_batch and invalidated by end_batch.
    const io::SequenceDataHost *read_data_1;
    const io::SequenceDataHost *read_data_2;

    /// I/O statistics
    IOStats iostats;

public:
    /// Factory method to create OutputFile objects
    /// \param [in] file_name The name of the file to create (will be silently overwritten if it already exists).
    ///             This method parses out the extension from the file name to determine what kind of file format to write.
    /// \param [in] aln_type The type of alignment (single or paired-end)
    /// \param [in] bnt A handle to the reference genome
    /// \return A pointer to an OutputFile object, or NULL if an error occurs.
    static OutputFile *open(const char *file_name, AlignmentType aln_type, BNT bnt);
};

/**
   @} // Output
   @} // IO
*/

} // namespace io
} // namespace nvbio
