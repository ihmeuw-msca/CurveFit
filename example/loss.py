#! /usr/bin/env python3
'''{begin_markdown loss_xam}
{spell_markdown finfo param aloss}

# Example and Test of Loss Functions

## Function Documentation
[st_loss](st_loss.md),
[normal_loss](normal_loss.md)

## Example Source Code
```python'''
import sys
import numpy
import cppad_py
import sandbox
sandbox.path()
import curvefit
# ----------------------------------------------------------------------------
# Loss Functions
# ----------------------------------------------------------------------------
eps99  = 99.0 * numpy.finfo(float).eps
#
# test values for t, param
r  = numpy.array( [ 1, 2, 3], dtype=float )
nu = numpy.array( [ 3, 2, 1], dtype=float )
# -----------------------------------------------------------------------
# f(t) = st_loss
ar     = cppad_py.independent(r)
aloss  = curvefit.core.functions.st_loss(ar, nu)
ay     = numpy.array( [ aloss ] )
f      = cppad_py.d_fun(ar, ay)
#
y          = f.forward(0, r)
check      = curvefit.core.functions.st_loss(r, nu)
rel_error  = y[0] / check - 1.0
assert abs( rel_error ) < eps99
# -----------------------------------------------------------------------
# f(t) = normal_loss
ar     = cppad_py.independent(r)
aloss  = curvefit.core.functions.normal_loss(ar)
ay     = numpy.array( [ aloss ] )
f      = cppad_py.d_fun(ar, ay)
#
y          = f.forward(0, r)
check      = curvefit.core.functions.normal_loss(r)
rel_error  = y[0] / check - 1.0
assert abs( rel_error ) < eps99
# -----------------------------------------------------------------------
print('loss.py: OK')
sys.exit(0)
'''```
{end_markdown loss_xam}'''
