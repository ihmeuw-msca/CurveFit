#! /bin/bash -e
# This is a simple install used by .travis.yml
# -----------------------------------------------------------------------------
# bash function that echos and executes a command
echo_eval() {
	echo $*
	eval $*
}
# -----------------------------------------------------------------------------
eval $(sed -n '/^cppad_prefix=/p'  bin/get_cppad.sh)
export LD_LIBRARY_PATH="$cppad_prefix/lib:$LD_LIBRARY_PATH"
if [ ! -e build/external ]
then
    echo_eval mkdir -p build/external
fi
echo_eval cd build/external
# -----------------------------------------------------------------------------
if [ ! -e cppad_py.git ]
then
    echo_eval git clone https://github.com/bradbell/cppad_py.git cppad_py.git
fi
echo_eval cd cppad_py.git
echo_eval git reset --hard
echo_eval git pull
echo_eval pip3 install . --user
# -----------------------------------------------------------------------------
echo 'get_cppad_py.sh: OK'
exit 0
