#! /usr/bin/env python3
"""
{begin_markdown unpack_param_xam}
{spell_markdown utils param}

# Example and Test of unpack_param

## Function Documentation
[unpack_param](unpack_param.md)

## Example Source Code
```python"""
import sys
import numpy
import sandbox
sandbox.path()
import curvefit
#
t      = numpy.array( [1, 2, 3, 4] )
#
# param a vector
param  = [5, 6]
a, b   = curvefit.core.utils.unpack_param(t, param)
assert a == 5
assert b == 6
#
# param a matrix
param  = numpy.array( [
    [5, 6, 7, 8]   ,
    [9, 10, 11, 12],
] )
alpha, beta  = curvefit.core.utils.unpack_param(t, param)
assert all( alpha == param[0, :] )
assert all( beta  == param[1, :] )
#
print('unpack_param.py: OK')
sys.exit(0)
"""```
{end_markdown unpack_param_xam}
"""
