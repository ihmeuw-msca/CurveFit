#! /usr/bin/env python3
"""
{begin_markdown numpy_ufunc_xam}
{spell_markdown ufunc finfo erf}

# Example and Test of numpy_ufunc

## Function Documentation
[numpy_ufunc](numpy_ufunc.md)

## Example Source Code
```python"""
import sys
import numpy
import scipy
import sandbox
sandbox.path()
import curvefit
from cppad_py import a_double
#
eps99     = 99.0 * numpy.finfo('float').eps
#
array     = numpy.array( [-0.5, 0.0, +0.5] )
result    = curvefit.core.numpy_ufunc.erf(array)
for i in range(len(array)) :
    abs_error = result[i] - scipy.special.erf(array[i])
    assert abs(abs_error) < eps99
#
array     = numpy.array( [a_double(-0.5), a_double(0.0), a_double(+0.5)] )
result    = curvefit.core.numpy_ufunc.erf(array)
for i in range(len(array)) :
    abs_error = result[i].value() - scipy.special.erf(array[i].value())
    assert abs(abs_error) < eps99
#
print('numpy_ufunc.py: OK')
sys.exit(0)
"""```
{end_markdown numpy_ufunc_xam}
"""
