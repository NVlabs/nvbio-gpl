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

// loader for variant call format files, version 4.2

#include <nvbio/basic/console.h>
#include <nvbio/io/vcf.h>
#include <nvbio/io/bufferedtextfile.h>

#include <stdlib.h>

namespace nvbio {
namespace io {

// loads a VCF 4.2 file, appending the data to output
bool loadVCF(SNPDatabase& output, const char *file_name)
{
    BufferedTextFile file(file_name);
    char *line, *end;
    uint32 line_counter = 0;

    while((line = file.next_record(&end)))
    {
        line_counter++;
        *end = '\0';

        // strip out comments
        char *comment = strchr(line, '#');
        if (comment)
            *comment = '\0';

        // skip all leading whitespace
        while (*line == ' ' || *line == '\t' || *line == '\r')
        {
            line++;
        }

        if (*line == '\0')
        {
            // empty line, skip
            continue;
        }

        // parse the entries in each record
        char *chrom, *pos, *id, *ref, *alt, *qual, *filter;

// ugly macro to tokenize the string based on strchr
#define NEXT(prev, next)                        \
    {                                           \
        next = strchr(prev, '\t');              \
        if (!next) {                                                    \
            log_error(stderr, "Error parsing VCF file (line %d): incomplete variant\n", line_counter); \
            return false;                                               \
        }                                                               \
        *next = '\0';                                                   \
        next++;                                                         \
    }

        chrom = line;
        NEXT(chrom, pos);
        NEXT(pos, id);
        NEXT(id, ref);
        NEXT(ref, alt);
        NEXT(alt, qual);
        NEXT(qual, filter);

#undef NEXT

        // convert position and quality
        char *endptr = NULL;
        uint64 position = strtoll(pos, &endptr, 10);
        if (!endptr || endptr == pos || *endptr != '\0')
        {
            log_error(stderr, "VCF file error (line %d): invalid position\n", line_counter);
            return false;
        }

        uint8 quality;
        if (*qual == '.')
        {
            quality = 0xff;
        } else {
            quality = (uint8) strtol(qual, &endptr, 10);
            if (!endptr || endptr == qual || *endptr != '\0')
            {
                log_warning(stderr, "VCF file error (line %d): invalid quality\n", line_counter);
                quality = 0xff;
            }
        }

        // add an entry for each possible variant listed in this record
        do {
            char *next_base = strchr(alt, ',');
            if (next_base)
                *next_base = '\0';

            output.chromosome.push_back(std::string(chrom));
            output.position.push_back(position);
            output.reference.push_back(std::string(ref));

            // if this is a called monomorphic variant (i.e., a site which has been identified as always having the same allele)
            // we push the reference string as the variant
            if (strcmp(alt, ".") == 0)
                output.variant.push_back(std::string(ref));
            else
                output.variant.push_back(std::string(alt));

            output.variant_quality.push_back(quality);

            if (next_base)
                alt = next_base + 1;
            else
                alt = NULL;
        } while (alt && *alt != '\0');
    }

    return true;
}

} // namespace io
} // namespace nvbio
