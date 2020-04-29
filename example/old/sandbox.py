import os
import sys
import pathlib
def path() :
	#
	# check .git is in working directory
	git_path  = pathlib.Path('.git')
	if not git_path.is_dir() :
		msg = 'sandbox.path: must be run from top git directory\n'
		sys.exit(msg)
	#
	# import sandbox version of curvefit
	src_dir     = os.getcwd() + '/src'
	module_path = pathlib.Path( src_dir + '/curvefit' )
	if module_path.is_dir() :
		sys.path.insert(0, src_dir)
	else :
		msg = 'sandbox.path: cannot file src/curvefit directory'
		sys.exit(msg)
