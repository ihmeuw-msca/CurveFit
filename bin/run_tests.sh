#! /bin/bash -e
# This is a test wrapper used by .travis.yml to set LD_LIBRARY_PATH
# -----------------------------------------------------------------------------
eval $(sed -n '/^cppad_prefix=/p'  bin/get_cppad.sh)
export LD_LIBRARY_PATH="$cppad_prefix/lib:$LD_LIBRARY_PATH"
make tests
make examples
echo 'run_test.sh: OK'
exit 0
