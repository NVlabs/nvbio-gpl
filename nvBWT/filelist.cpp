/*
 * nvbio
 * Copyright (C) 2011-2013, NVIDIA Corporation
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

#include <string>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#ifdef _WIN32
#include <io.h>
#else
#include <glob.h>
#endif
#include <string.h>
#include <nvbio/basic/types.h>


/* Appends sorted list of files matching pattern to out_list.
 * Should work on linux or windows */
void list_files(const char* pattern, std::vector<std::string>& out_list)
{
#ifdef _WIN32
    const char dirsep = '\\';
#else
    const char dirsep = '/';
#endif

    std::string base_input_name(pattern);
    size_t last_slash = base_input_name.rfind(dirsep);
    std::string base_path = base_input_name.substr( 0, last_slash+1 );
    fprintf(stderr, "directory  : \"%s\"\n", base_path.c_str());

    typedef std::pair<nvbio::uint32, std::string> sortkey;
    std::vector<sortkey> files;
    sortkey temp;

#ifdef _WIN32

    _finddata_t file_info;
    intptr_t find_handle = _findfirst( base_input_name.c_str(), &file_info );
    if (find_handle == -1)
    {
        fprintf(stderr, "unable to locate \"%s\"", base_input_name.c_str());
        exit(1);
    }

    do {
      temp.second = base_path + std::string( file_info.name );
      files.push_back(temp);
    } while (_findnext( find_handle, &file_info) != -1);

#else

    glob_t info;
    int stat = glob( base_input_name.c_str(), 0, 0, &info );
    if( stat != 0 )
    {
        fprintf(stderr, "unable to locate \"%s\"", base_input_name.c_str());
        exit(1);
    }
    for(nvbio::uint32 i=0; i<info.gl_pathc; ++i) {
        temp.second = std::string( info.gl_pathv[i] );
        files.push_back(temp);
    }
    globfree(&info);

#endif

    // Figure out what the sort number should be for each file
    // We sort by the last block of numbers before the last dot
    std::vector<sortkey>::iterator iter;
    for(iter=files.begin(); iter!=files.end(); ++iter) {
        size_t end, begin;
        std::string numstring;
        end = iter->second.find_last_of(".");
        end = iter->second.find_last_of("0123456789", end);
        if( end == std::string::npos ) {
            // no numbers in file
            iter->first = -1; // sort will be random but before numbers
        } else {
            begin = 1 + iter->second.find_last_not_of("0123456789", end);
            numstring = iter->second.substr(begin, end-begin+1);
            iter->first = atoi( numstring.c_str() );
        }
    }
    std::sort(files.begin(), files.end());
    
    for(iter=files.begin(); iter!=files.end(); ++iter) {
        out_list.push_back(iter->second);
    }
}

