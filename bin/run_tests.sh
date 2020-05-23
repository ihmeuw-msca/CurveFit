#! /bin/bash -e
# This is a test wrapper used by .travis.yml to set LD_LIBRARY_PATH
# -----------------------------------------------------------------------------
get_cppad_sh='build/external/cppad_py.git/bin/get_cppad.sh'
eval $(sed -n '/^cppad_prefix=/p'  $get_cppad_sh)
echo "export LD_LIBRARY_PATH=$cppad_prefix/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH="$cppad_prefix/lib:$LD_LIBRARY_PATH"
make tests
make examples
echo 'run_test.sh: OK'
exit 0
