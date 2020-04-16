#! /usr/bin/env python3
# ----------------------------------------------------------------------------
# Original *.md files are written in this sub-directory of the docs directory
extract_dir = 'extract_md'
# list of files that contain markdown sections in them
file_list = [
    'example/get_started.py',
    'example/covariate.py',
    'example/random_effect.py',
    'example/sizes_to_indices.py',
    'example/param_time_fun.py',
    'example/unzip_x.py',

    'src/curvefit/core/utils.py',
    'src/curvefit/core/functions.py',
    'src/curvefit/core/effects2params.py',
    'src/curvefit/core/objective_fun.py',
    'bin/extract_md.py',
    'bin/get_cppad_py.py',
]
# list of extra words that the spell checker will consider correct
extra_special_words = [
    'covariates',
    'covariate',
    'curvefit',
    'curvemodel',
    'dict',
    'initialized',
    'initialize',
    'numpy',
    'py',
    'sandbox',
    'scipy',
    'xam',

    r'\begin',
    r'\cdot',
    r'\circ',
    r'\ell',
    r'\end',
    r'\exp',
    r'\frac',
    r'\int',
    r'\ldots',
    r'\log',
    r'\left',
    r'\mbox',
    r'\partial',
    r'\right',
    r'\sum',
]
# ----------------------------------------------------------------------------
'''{begin_markdown extract_md.py}
{spell_markdown
    markdown
    md
    underbar
    mkdocs.yml
    nbsp
}

# Extracting Markdown Documentation From Source Code

## Syntax
`bin/extract_md.py`

## extract_dir
The variable *extract_dir* at top of `bin/extract_md.py`
determines the sub-directory, below the `docs` directory,
where the markdown files will be written.
Any files names that end in `.md` in that directory will be
removed at the beginning so that all the files in this directory
have been extracted from the current version of the source code.

## file_list
The variable *file_list* at top of `bin/extract_md.py`
is a list of file names, relative to the top git repository directory,
that the markdown files will be extracted from.

## extra_special_words
The variable *extra_special_words* is a list of extra words that
the spell checker will consider correct; see
[spell checking](#spell-checking) below.

## Start Section
The start of a markdown section of the input file is indicated by the following
text:
<p style="margin-left:10%">
{begin_markdown <i>section_name</i>}
</p>
Here *section_name* is the name of output file corresponding to this section.
The possible characters in *section_name* are A-Z, a-z, 0-9, underbar `_`,
and dot `.`

### mkdocs.yml
For each *section_name* in the documentation there must be a line in the
`mkdocs.yml` file fo the following form:

&nbsp;&nbsp;&nbsp;&nbsp;- *section_name* : '*extract_dir*/*section_name*.md'

where there can be any number of spaces around the dash character (-)
and the colon character (:).

## End Section
The end of a markdown section of the input file is indicated by the following
text:
<p style="margin-left:10%">
{end_markdown <i>section_name</i>}
</p>
Here *section_name* must be the same as in the start of this markdown section.

## Spell Checking
Special words can be added to the correct spelling list for a particular
section as follows:
<p style="margin-left:10%">
{spell_markdown
    <i>special_1 ...  special_n</i>
}
</p>
Here *special_1*, ..., *special_n* are special words
that are to be considered valid for this section.
In the syntax above they are all on the same line,
but they could be on different lines.
Each word starts with an upper case letter,
a lower case letter, or a back slash.
The rest of the characters in a word are lower case letters.
The case of the first letter does not matter when checking for special words;
e.g., if `abcd` is *special_1* then `Abcd` will be considered a valid word.
The back slash is included at the beginning of a word
so that latex commands are considered words.
The latex commands corresponding to the letters in the greek alphabet
are automatically included.
Any latex commands in the
[extra_special_words](#extra_special_words)
are also automatically included.

## Code Blocks
A code block within a markdown section begins and ends with three back quotes.
Thus there must be an even number of occurrences of three back quotes.
The first three back quotes must have a language name directly after it.
The language name must be a sequence of letters; e.g., `python`.
The other characters on the same line as the three back quotes
are not included in the markdown output. This enables one to begin or end
a comment block without having those characters in the markdown output.

## Indentation
If all of the extracted markdown documentation for a section is indented
by the same number of space characters, those space characters
are not included in the markdown output. This enables one to indent the
markdown so it is grouped with the proper code block in the source.

{end_markdown extract_md.py}'''
# ----------------------------------------------------------------------------
import sys
import re
import os
import pdb
import spellchecker
# ---------------------------------------------------------------------------
# spell_checker
bad_words_in_spellchecker = [
    'thier',
]
greek_alphabet_latex_command = [
    r'\alpha',
    r'\beta',
    r'\gamma',
    r'\delta',
    r'\epsilon',
    r'\zeta',
    r'\eta',
    r'\theta',
    r'\iota',
    r'\kappa',
    r'\lamda',
    r'\mu',
    r'\nu',
    r'\xi',
    r'\omicron',
    r'\pi',
    r'\rho',
    r'\sigma',
    r'\tau',
    r'\upsilon',
    r'\phi',
    r'\chi',
    r'\psi',
    r'\omega',
]
#
spell_checker = spellchecker.SpellChecker(distance=1)
spell_checker.word_frequency.remove_words(bad_words_in_spellchecker)
spell_checker.word_frequency.load_words(greek_alphabet_latex_command)
spell_checker.word_frequency.load_words(extra_special_words)
# ---------------------------------------------------------------------------
#
# add program name to system error call
def sys_exit(msg) :
    sys.exit( 'bin/extract_md.py: ' + msg )
#
# check working directory
if not os.path.isdir('.git') :
    msg = 'must be executed from to git directory\n'
    sys_exit(msg)
#
# remove all *.md files from output directory (so only new ones remain)
output_dir = 'docs/' + extract_dir
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
pattern_begin_markdown = re.compile( r'{begin_markdown \s*([A-Za-z0-9_.]*)}' )
pattern_end_markdown   = re.compile( r'{end_markdown \s*([A-Za-z0-9_.]*)}' )
pattern_spell_markdown = re.compile( r'{spell_markdown([^}]*)}' )
pattern_begin_3quote   = re.compile( r'[^\n]*(```([a-zA-Z]*))[^\n]*' )
pattern_end_3quote     = re.compile( r'[^\n]*(```)[^\n]*' )
pattern_newline        = re.compile( r'\n')
pattern_word           = re.compile( r'[\\A-Za-z][a-z]*' )
# -----------------------------------------------------------------------------
# process each file in the list
for file_in in file_list :
    #
    # file_data
    file_ptr   = open(file_in, 'r')
    file_data  = file_ptr.read()
    file_ptr.close()
    #
    # file_index is where to start search for next pattern in file_data
    file_index  = 0
    while file_index < len(file_data) :
        #
        # match_begin_markdown
        data_rest   = file_data[file_index : ]
        match_begin_markdown = pattern_begin_markdown.search(data_rest)
        #
        if match_begin_markdown == None :
            if file_index == 0 :
                # Use @ so does not match pattern_begin_markdown in this file.
                msg  = 'can not find: @begin_markdown section_name}\n'
                msg  = msg.replace('@', '{')
                msg += 'in ' + file_in + '\n'
                sys_exit(msg)
            file_index = len(file_data)
        else :
            # section_name
            section_name = match_begin_markdown.group(1)
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
            # file_index
            file_index += match_begin_markdown.end()
            #
            # match_end_markdown
            data_rest = file_data[file_index : ]
            match_end_markdown = pattern_end_markdown.search(data_rest)
            #
            if match_end_markdown == None :
                msg  = 'can not find: "{end_markdown section_name}\n'
                msg += 'in ' + file_in + ', section ' + section_name + '\n'
                sys_exit(msg)
            if match_end_markdown.group(1) != section_name :
                msg = 'in file ' + file_in + '\nsection names do not match\n'
                msg += 'begin_markdown section name = '+section_name + '\n'
                msg += 'end_markdown section name   = '
                msg += match_end_markdown.group(1) + '\n'
                sys_exit(msg)
            #
            # output_data
            output_start = file_index
            output_end   = file_index + match_end_markdown.start()
            output_data  = file_data[ output_start : output_end ]
            #
            # process spell command
            match_spell = pattern_spell_markdown.search(output_data)
            spell_list  = list()
            if match_spell != None :
                for itr in pattern_word.finditer( match_spell.group(1) ) :
                    spell_list.append( itr.group(0).lower() )
                start       = match_spell.start()
                end         = match_spell.end()
                output_data = output_data[: start] + output_data[end :]
            #
            # remove characters on same line as triple back quote
            output_index  = 0
            match_begin_3quote = pattern_begin_3quote.search(output_data)
            while match_begin_3quote != None :
                if match_begin_3quote.group(2) == '' :
                    msg  = 'language missing directly after first'
                    msg += ' ``` for a code block\n'
                    msg += 'in ' + file_in
                    msg += ', section ' + section_name + '\n'
                    sys.exit(msg)
                begin_start = match_begin_3quote.start() + output_index
                begin_end   = match_begin_3quote.end()   + output_index
                output_rest = output_data[ begin_end : ]
                match_end_3quote   = pattern_end_3quote.search( output_rest )
                if match_end_3quote == None :
                    msg  = 'number of triple backquotes is not even in '
                    msg += file_in + ', section ' + section_name + '\n'
                    sys_exit(msg)
                end_start = match_end_3quote.start() + begin_end
                end_end   = match_end_3quote.end()   + begin_end
                #
                data_left   = output_data[: begin_start ]
                data_left  += match_begin_3quote.group(1)
                data_left  += output_data[ begin_end : end_start ]
                data_left  += match_end_3quote.group(1)
                data_right  = output_data[ end_end : ]
                #
                output_data  = data_left + data_right
                output_index = len(data_left)
                match_begin_3quote  = pattern_begin_3quote.search(data_right)
            #
            # num_remove (for indented documentation)
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
                    tripple_back_quote = output_data[next_:].startswith('```')
                    if ch != '\n' and ch != ' ' and not tripple_back_quote :
                        num_remove = min(num_remove, next_ - start - 1)
            #
            # write file for this section
            file_out          = output_dir + '/' + section_name + '.md'
            file_ptr          = open(file_out, 'w')
            start_line        = 0
            first_spell_error = True
            for newline in newline_list :
                tripple_back_quote = output_data[start_line:].startswith('```')
                if not tripple_back_quote :
                    start_line += num_remove
                if start_line <= newline :
                    line = output_data[start_line : newline + 1]
                    # ------------------------------------------------------
                    # check spelling
                    word_list = list()
                    for itr in pattern_word.finditer( line ) :
                        word = itr.group(0)
                        if len( spell_checker.unknown( [word] ) ) > 0 :
                            if not word.lower() in spell_list :
                                if first_spell_error :
                                    msg  = 'warning: file = ' + file_in
                                    msg += ', section = ' + section_name
                                    print(msg)
                                    first_spell_error = False
                                msg  = 'spelling = ' + word
                                suggest = spell_checker.correction(word)
                                if suggest != word :
                                    msg += ', suggest = ' + suggest
                                print(msg)
                                spell_list.append(word.lower())
                    # ------------------------------------------------------
                    file_ptr.write( line )
                else :
                    file_ptr.write( "\n" )
                start_line = newline + 1
            file_ptr.close()
            #
            # file_index
            file_index += match_end_markdown.end()
# -----------------------------------------------------------------------------
# read mkdocs.yml
file_in   = 'mkdocs.yml'
file_ptr  = open(file_in, 'r')
file_data = file_ptr.read()
file_ptr.close()
#
for section_name in section_list :
    # There should be an line in mkdocs.yml with the following contents:
    # - section_name : 'extract_dir/section_name.md'
    # where the spaces are optional
    pattern  = r'\n[ \t]*-[ \t]*'
    pattern += section_name.replace('.', '[.]')
    pattern += r"[ \t]*:[ \t]*'"
    pattern += extract_dir.replace('.', '[.]')
    pattern += r'/'
    pattern += section_name.replace('.', '[.]')
    pattern += r"[.]md'[ \t]*\n"
    #
    match_line = re.search(pattern, file_data)
    if match_line == None :
        msg   = 'Can not find following line in ' + file_in + ':\n'
        line  = ' - ' + section_name
        line += " : '" + extract_dir
        line += '/' + section_name
        line += ".md'"
        msg  += '    ' + line + '\n'
        msg  += 'Spaces above are optional and can be multiple spaces\n'
        sys.exit(msg)
#
print('docs/extract.py: OK')
sys.exit(0)
