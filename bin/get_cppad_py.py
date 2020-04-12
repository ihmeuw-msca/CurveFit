#! /usr/bin/env python3
'''{begin_markdown get_cppad_py.py}
{spell_markdown
    cppad
    debug
}

# Download and Install cppad_py

## Syntax
`bin/get_cppad_py.py [test] [debug] [prefix]

## Command Line Arguments
The order of the command line arguments *test*, *debug*, and *prefix*
does not matter

## test
If this argument is present it must be `test=True`.
In this case cppad_py will be tested before the install.

## debug
If this argument is present it must be `debug=True`.
In this case the debug version of cppad_py will be built
(possible tested) and installed.

## prefix
If this argument is present it must be `prefix=`*path*.
In this case the *path* will be the prefix where cppad_py is installed.
Otherwise the standard `setup.py` path is used.

{end_markdown get_cppad_py.py}'''
#
import sys
import os
import subprocess
import distutils.dir_util
# ---------------------------------------------------------------------------
def sys_exit(msg) :
    sys.exit( 'bin/get_cppad_py.py: ' + msg )
#
def system_command(cmd_list) :
    print( ' '.join(cmd_list) )
    result = subprocess.run(cmd_list, capture_output=True)
    returncode = result.returncode
    stdout     = result.stdout.decode('utf-8')
    stderr     = result.stderr.decode('utf-8')
    if len(stderr) != 0 :
        assert stderr[-1] == '\n'
        print(stderr[:-1])
    if returncode != 0 :
        msg  = 'system command failed:\n' + ' '.join(cmd_list) + '\n'
        msg += stderr
        sys.exit(msg)
    return stdout
# ---------------------------------------------------------------------------
usage = '''usage: bin/get_cppad_py.py [test] [debug] [prefix]
        where the order of the arguments does not matter and
test:   if present must be test=True (test cppad_py)
debug:  if present must be debug=True (debug version of cppad_py)
prefix: if not present, must be prefix=path
        where path is the prefix used for the cppad_py install.
'''
test   = None
debug  = None
prefix = None
argv_index = 0
while argv_index + 1 < len(sys.argv) :
    argv_index += 1
    if sys.argv[argv_index] == 'test=True' :
        test = True
    elif sys.argv[argv_index] == 'debug=True' :
        debug = True
    elif sys.argv[argv_index].startswith('prefix=') :
        prefix = sys.argv[argv_index][7:]
    else :
        usage += '\n"' + sys.argv[argv_index] + '" is not a valid choice'
        sys.exit(usage)
#
# install_prefix
if len(sys.argv) == 3 :
    install_prefix = sys.argv[2]
else :
    install_prefix = None
#
# check topsrcdir
if not os.path.isdir('.git') :
    msg = 'must be executed from to git directory\n'
    sys_exit(msg)
#
# change into build/external directory
distutils.dir_util.mkpath('build/external')
os.chdir('build/external')
#
# cppad_py.git
if not os.path.isdir('cppad_py.git') :
    system_command( [
        'git', 'clone', 'https://github.com/bradbell/cppad_py.git',
        'cppad_py.git'
    ] )
os.chdir('cppad_py.git')
system_command( [ 'git', 'reset', '--hard' ] )
system_command( [ 'git', 'pull' ] )
#
# get_cppad.sh
system_command( [
    'sed', '-i', 'setup.py', '-e' , 's|\(^text_cppad *=\).*|\1 "false"|'
] )
stdout = system_command( [ 'bin/get_cppad.sh' ] )
print(stdout[:-1])
#
# test
if test :
    # build
    build_cmd = [ 'python3', 'setup.py', 'build_ext', '--inplace' ]
    if debug :
        build_cmd += [ '--debug', '--undef', 'NDEBUG' ]
    system_command( build_cmd )
    #
    # test
    os.chdir('lib/example/python')
    stdout = system_command( [ 'python3', 'check_all.py' ] )
    print(stdout[:-1])
    os.chdir('../../..')
#
# install
install_cmd = [ 'python3', 'setup.py', 'build_ext' ]
if debug :
        install_cmd += [ '--debug', '--undef', 'NDEBUG' ]
install_cmd.append( 'install' )
if prefix != None :
    install_cmd.append( '--prefix=' + prefix )
system_command( install_cmd )
# ----------------------------------------------------------------------------
print('bin/get_cppad_py.py: OK')
sys.exit(0)
