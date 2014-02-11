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

#include <stdio.h>

#if WIN32
extern unsigned int TEXT_BLUE;
extern unsigned int TEXT_RED;
extern unsigned int TEXT_GREEN;
extern unsigned int TEXT_BRIGHT;
#endif

void textcolor(unsigned int color);

void log_visible(FILE* file, const char* string, ...);
void log_info(FILE* file, const char* string, ...);
void log_stats(FILE* file, const char* string, ...);
void log_verbose(FILE* file, const char* string, ...);
void log_debug(FILE* file, const char* string, ...);
void log_warning(FILE* file, const char* string, ...);
void log_error(FILE* file, const char* string, ...);

void log_visible_cont(FILE* file, const char* string, ...);
void log_info_cont(FILE* file, const char* string, ...);
void log_stats_cont(FILE* file, const char* string, ...);
void log_verbose_cont(FILE* file, const char* string, ...);
void log_debug_cont(FILE* file, const char* string, ...);
void log_warning_cont(FILE* file, const char* string, ...);
void log_error_cont(FILE* file, const char* string, ...);

void log_visible_nl(FILE* file);
void log_info_nl(FILE* file);
void log_stats_nl(FILE* file);
void log_verbose_nl(FILE* file);
void log_debug_nl(FILE* file);
void log_warning_nl(FILE* file);
void log_error_nl(FILE* file);

enum Verbosity
{
    V_ERROR   = 0,
    V_WARNING = 1,
    V_VISIBLE = 2,
    V_INFO    = 3,
    V_STATS   = 4,
    V_VERBOSE = 5,
    V_DEBUG   = 6,
};

void set_verbosity(Verbosity);
