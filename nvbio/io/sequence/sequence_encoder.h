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

#include <nvbio/io/sequence/sequence.h>

namespace nvbio {
namespace io {

///
/// Base class to encode a host-side SequenceData object.
/// The alphabet is provided to this class as a run-time parameter.
///
struct SequenceDataEncoder
{
    /// a set of flags describing the operators to apply to a given strand
    ///
    enum StrandOp
    {
        NO_OP                 = 0x0000,     ///< default, no operator applied
        REVERSE_OP            = 0x0001,     ///< reverse operator
        COMPLEMENT_OP         = 0x0002,     ///< complement operator
        REVERSE_COMPLEMENT_OP = 0x0003,     ///< convenience definition, same as StrandOp( REVERSE_OP | COMPLEMENT_OP )
    };

    /// constructor
    ///
    SequenceDataEncoder(const SequenceAlphabet alphabet) : m_alphabet( alphabet ) {}

    /// destructor
    ///
    virtual ~SequenceDataEncoder() {}

    /// reserve enough storage for a given number of reads and bps
    ///
    virtual void reserve(const uint32 n_reads, const uint32 n_bps) {}

    /// add a read to the end of this batch
    ///
    /// \param sequence_len                 input read length
    /// \param name                         read name
    /// \param base_pairs                   list of base pairs
    /// \param quality                      list of base qualities
    /// \param quality_encoding             quality encoding scheme
    /// \param truncate_sequence_len        truncate the read if longer than this
    /// \param conversion_flags             conversion operators applied to the strand
    ///
    virtual void push_back(
        const uint32            sequence_len,
        const char*             name,
        const uint8*            base_pairs,
        const uint8*            quality,
        const QualityEncoding   quality_encoding,
        const uint32            truncate_sequence_len,
        const StrandOp          conversion_flags) {}

    /// signals that the batch is complete
    ///
    virtual void end_batch(void) {}

    /// fetch the actual SequenceData object
    ///
    virtual void* get_data() const { return NULL; }

    /// return the sequence data info
    ///
    virtual SequenceDataInfo* info() const { return NULL; }

    /// get the alphabet
    ///
    SequenceAlphabet alphabet() const { return m_alphabet; }

    /// fetch the actual SequenceData object of the specified alphabet
    /// NOTE: this only works for the proper alphabet
    ///
    template <SequenceAlphabet SEQUENCE_ALPHABET>
    SequenceData<SEQUENCE_ALPHABET>* get() const
    {
        return reinterpret_cast<SequenceData<U_ALPHABET>*>( get_data(); );
    }

private:
    SequenceAlphabet m_alphabet;
};

/// create a sequence encoder
///
SequenceDataEncoder* create_encoder(SequenceDataHost<DNA>* data);

/// create a sequence encoder
///
SequenceDataEncoder* create_encoder(SequenceDataHost<DNA_N>* data);

/// create a sequence encoder
///
SequenceDataEncoder* create_encoder(SequenceDataHost<PROTEIN>* data);

} // namespace io
} // namespace nvbio
