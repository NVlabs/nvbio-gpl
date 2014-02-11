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
#include <stdio.h>
#include <stdlib.h>

namespace nvbio {
namespace html {

enum Formatting { FORMATTED };

void html_begin(FILE* output);
void html_end(FILE* output);
void header(FILE* output, const char* title, const char* css, const char* meta = NULL);
void body_begin(FILE* output);
void body_end(FILE* output);
void table_begin(FILE* output, const char* id, const char* cls, const char* caption);
void table_end(FILE* output);
void tr_begin(FILE* output, ...);
void tr_end(FILE* output);
void th(FILE* output, const char* name, ...);
void td(FILE* output, const char* name, ...);

const char* style();

/// html document object
///
struct html_object
{
    /// html document constructor
    ///
    html_object(FILE* output) : m_output( output ) { html_begin(m_output); }

    /// destructor
    ///
    ~html_object()                                  { html_end(m_output); }

    FILE* m_output;
};
/// header object
///
struct header_object
{
    /// header constructor
    ///
    header_object(FILE* output, const char* title, const char* css, const char* meta = NULL) : m_output( output ) { header(m_output, title, css, meta); }

    /// destructor
    ///
    ~header_object() {}

    FILE* m_output;
};
/// body object
///
struct body_object
{
    /// body constructor
    ///
    body_object(FILE* output) : m_output( output ) { body_begin(m_output); }

    /// destructor
    ///
    ~body_object()                                  { body_end(m_output); }

    FILE* m_output;
};
/// table object
///
struct table_object
{
    /// table constructor
    ///
    table_object(FILE* output, const char* id, const char* cls, const char* caption) : m_output( output ) { table_begin(m_output, id, cls, caption); }

    /// destructor
    ///
    ~table_object()                                                                                        { table_end(m_output); }

    FILE* m_output;
};
/// tr element
///
struct tr_object
{
    /// constructor
    ///
    /// \param ...      NULL-terminated list of key/value string pairs
    tr_object(FILE* output, ...);

    /// destructor
    ///
    ~tr_object() { tr_end(m_output); }

    FILE* m_output;
};
/// th element
///
struct th_object
{
    /// constructor
    ///
    /// \param name     element name
    /// \param ...      NULL-terminated list of key/value string tags
    th_object(FILE* output, const char* name, ...);

    /// formatted constructor: allows to specify the element name with
    /// a printf-like formatted string
    ///
    /// \param formatted    formatting flag
    /// \param ...          a NULL-terminated list of key/value string tags
    ///                     followed by a printf-like formatted string
    th_object(FILE* output, const Formatting formatted, ...);

    /// destructor
    ///
    ~th_object() {}

    FILE* m_output;
};
/// td element
///
struct td_object
{
    /// constructor
    ///
    /// \param name     element name
    /// \param ...      NULL-terminated list of key/value string tags
    td_object(FILE* output, const char* name, ...);

    /// formatted constructor: allows to specify the element name with
    /// a printf-like formatted string
    ///
    /// \param formatted    formatting flag
    /// \param ...          a NULL-terminated list of key/value string tags
    ///                     followed by a printf-like formatted string
    td_object(FILE* output, const Formatting formatted, ...);

    /// destructor
    ///
    ~td_object() {}

    FILE* m_output;
};

} // namespace html
} // namespace nvbio
