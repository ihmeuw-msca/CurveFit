#! /bin/python3
# vim: set expandtab:
# ----------------------------------------------------------------------------
# Original *.md files are written in this directory (must start with docs/)
output_dir = 'docs/extract_md'
# list of files that contain markdown sections in them
file_list = [
    'example/get_started.py',
    'example/covariate.py',
    'example/sizes_to_indices.py',

    'src/curvefit/core/utils.py',
    'docs/extract_md.py',
]
# ----------------------------------------------------------------------------
'''[begin_markdown extract_md]
# Extracting Markdown Documentation From Soure Code

## Syntax
`python docs/extract_md.py`

## output_dir
The variable *output_dir* at top of `docs/extract_md.py`
determines the directory where the markdown files will be written.
Any files names that end in `.md` in that directory will be
removed at the beginning (so that the extracted files can be easily reconized).

## file_list
The variable *file_list* at top of `docs/extract_md.py`
is a list of file names, relative to the top git repository directory,
that the markdown files will be extracted from.

## Start Section
The start of a markdown section of the input file is indicated by the following
text:
<p style="margin-left:10%">
[begin_markdown <i>section_name</i>]
<p/>
Here *section_name* is the name of output file corresponding to this section.
This name does not inlucde the *output_dir* or the .md extension.

## End Section
The end of a markdown section of the input file is indicated by the following
text:
<p style="margin-left:10%">
[end_markdown <i>section_name</i>]
<p/>
Here *section_name* must be the same as in the start of this markdown section.

## Code Blocks
A code block within a markdown section begins and ends with three back quotes.
Thuse there must be an even number of occurances of three back quotes.
The other characters on the same line as the three back quotes are not
included in the markdown output. (This enables one to begin or end a comment
block without having those characters in the markdown output.)
There is one exception to this rule: if a language name directly follows
the three bakc quotes that start a code block, the language name is included
in the output.

## Indentation
If all of the extracted markdown documentation for a section is indented
by the same number of space characters, that may space characters
are not included in the markdown output. This enables one to indent the
markdown so it is grouped with the proper code block in the soruce.

[end_markdown extract_md]'''
# ----------------------------------------------------------------------------
import sys
import re
import os
import pdb
#
# add program name to system error call
def sys_exit(msg) :
    sys.exit( 'docs/extract_md.py: ' + msg )
#
# check working directory
if not os.path.isdir('.git') :
    msg = 'must be executed from to git directory\n'
    sys_exit(msg)
#
# remove all *.md files from output directory (so only new ones remain)
assert output_dir.startswith('docs/')
if os.path.isdir(output_dir) :
    for file_name in os.listdir(output_dir) :
        if file_name.endswith('.md') :
            file_path = output_dir + "/" + file_name
            os.remove(file_path)
else :
    os.mkdir(output_dir)
#
# initialize list of section names and corresponding file names
section_list       = list()
corresponding_file = list()
#
# pattern for start of markdown section
pattern_begin_markdown = re.compile( '\\[begin_markdown\\s*(\\w*)\\]' )
pattern_end_markdown   = re.compile( '\\[end_markdown\\s*(\\w*)\\]' )
pattern_begin_3quote   = re.compile( '[^\\n]*(```\\s*\\w*)[^\\n]*' )
pattern_end_3quote     = re.compile( '[^\\n]*(```)[^\\n]*' )
pattern_newline        = re.compile( '\\n')
# -----------------------------------------------------------------------------
# process each file in the list
for file_in in file_list :
    #
    # file_data
    file_ptr   = open(file_in, 'r')
    file_data  = file_ptr.read()
    file_ptr.close()
    #
    # data_index is where to start search for next pattern
    data_index  = 0
    while data_index < len(file_data) :
        #
        # match_begin
        data_rest   = file_data[data_index : ]
        match_begin = pattern_begin_markdown.search(data_rest)
        #
        if match_begin == None :
            if data_index == 0 :
                # There is no match for pattern_begin_markdown in this file.
                msg  = 'can not find: [begin_markdown section_name\]\n'
                msg += 'in ' + file_in + '\n'
                sys_exit(msg)
            data_index = len(file_data)
        else :
            # section_name
            section_name = match_begin.group(1)
            if section_name == '' :
                msg  = 'section_name after begin_markdown is empty; see file\n'
                msg += file_in
                sys_exit(msg)
            #
            if section_name in section_list :
                # this section appears multiple times
                index = section_list.index(section_name)
                msg  = 'begin_markdown ' + section_name
                msg += ' appears twice; see files\n' + file_in + ' and '
                msg += corresponding_file[index]
                sys_exit(msg)
            section_list.append( section_name )
            corresponding_file.append( file_in )
            #
            # data_index
            data_index += match_begin.end()
            #
            # match_end
            data_rest = file_data[data_index : ]
            match_end = pattern_end_markdown.search(data_rest)
            #
            if match_end == None :
                msg  = 'can not find: [end\_markdown section_name]\n'
                msg += 'in ' + file_in + ', section ' + section_name + '\n'
                sys_exit(msg)
            #
            # output_data
            output_start = data_index
            output_end   = data_index + match_end.start()
            output_data  = file_data[ output_start : output_end ]
            #
            # remove characters on same line as triple back quote
            data_index  = 0
            match_begin = pattern_begin_3quote.search(output_data)
            while match_begin != None :
                begin_start = match_begin.start() + data_index
                begin_end   = match_begin.end() + data_index
                output_rest = output_data[ begin_end : ]
                match_end   = pattern_end_3quote.search( output_rest )
                if match_end == None :
                    msg  = 'number of triple backquotes is not even in '
                    msg += file_in + '\n'
                    sys_exit(msg)
                end_start = match_end.start() + begin_end
                end_end   = match_end.end()   + begin_end
                #
                data_left   = output_data[: begin_start ]
                data_left  += match_begin.group(1)
                data_left  += output_data[ begin_end : end_start ]
                data_left  += match_end.group(1)
                data_right  = output_data[ end_end : ]
                #
                output_data = data_left + data_right
                data_index  = len(data_left)
                match_begin = pattern_begin_3quote.search(data_right)
            #
            # num_remove
            len_output   = len(output_data)
            num_remove   = len(output_data)
            newline_itr  = pattern_newline.finditer(output_data)
            newline_list = list()
            for itr in newline_itr :
                start = itr.start()
                newline_list.append( start )
                next_ = start + 1
                if next_ < len_output and num_remove != 0 :
                    ch = output_data[next_]
                    while ch == ' ' and next_ + 1 < len_output :
                        next_ += 1
                        ch = output_data[next_]
                    if ch == '\t' :
                        msg  = 'tab in white space at begining of a line\n'
                        msg += 'in ' + file_in
                        msg +=+ ', section ' + section_name + '\n'
                        sys_exit(msg)
                    if ch != '\n' and ch != ' ' :
                        num_remove = min(num_remove, next_ - start - 1)
            #
            # write file for this section
            file_out   = output_dir + '/' + section_name + '.md'
            file_ptr   = open(file_out, 'w')
            start_line = num_remove
            for newline in newline_list :
                if start_line <= newline :
                    file_ptr.write( output_data[start_line : newline + 1] )
                else :
                    file_ptr.write( "\n" )
                start_line = newline + 1 + num_remove
            file_ptr.close()
            #
            # data_index
            data_index += match_end.end()
# -----------------------------------------------------------------------------
# read mkdocs.yml
file_in   = 'mkdocs.yml'
file_ptr  = open(file_in, 'r')
file_data = file_ptr.read()
file_ptr.close()
#
# match_nav_start
match_nav_start   = re.search('\nnav:', file_data)
if match_nav_start == None :
    msg  = 'can not find: nav: at beginning of line in ' + file_in + '\n'
    sys_exit(msg)
#
# match_extract
data_index    = match_nav_start.end() + 1
data_rest     = file_data[data_index : ]
match_extract = re.search('\\n  - Extracted Doc:', data_rest)
#
# match_nav_end
match_nav_end  = re.search('\n[^ ]', data_rest)
if match_nav_end == None :
    msg  = 'can not find: end of nav: before end of file in ' + file_in + '\n'
    sys_exit(msg)
#
# open mkdocs.yml for writing
file_out = file_in
file_ptr  = open(file_in, 'w')
#
file_ptr.write( file_data[ : data_index ] )
file_ptr.flush()
if match_extract == None :
    # write up to end of old nav section
    file_ptr.write( data_rest[: match_nav_end.start() + 1] )
else :
    # write up to beginning of old Extract Documentation
    file_ptr.write( data_rest[: match_extract.start() + 1] )
file_ptr.flush()
#
# write out extracted section
file_ptr.write( '  - Extracted Doc:\n' )
for section_name in section_list :
    section_path = output_dir[5 :] + '/' + section_name + '.md'
    line      = '    - ' + section_name + ": '" + section_path + "'\n"
    file_ptr.write(line)
file_ptr.flush()
#
# write out rest of mkdocs.yml
file_ptr.write( data_rest[ match_nav_end.start() + 1 :] )
file_ptr.close()
#
print('docs/extract.py: OK')
sys.exit(0)
