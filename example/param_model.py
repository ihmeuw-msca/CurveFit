#! /usr/bin/env python3
"""
{begin_markdown param_model_xam}
{spell_markdown param ufunc erf finfo arange vstack}

# Example and Test of param_model

## Function Documentation
[param_model](param_model.md)

## Example Source Code
```python"""
import sys
import scipy
import numpy
import sandbox
sandbox.path()
import curvefit
from cppad_py import a_double
from curvefit.core.numpy_ufunc import erf
#
def f(t, alpha, beta, p) :
    return 0.5 * p * ( 1.0 + erf( alpha * ( t - beta ) ) )
#
eps99 = 99.0 * numpy.finfo(float).eps
#
# param a vector of float
alpha  = 0.1
beta   = 0.2
p      = 0.3
t      = numpy.array( [1.0, 2.0, 3.0, 4.0] )
n      = len(t)
param  = [alpha, beta, p]
result = curvefit.core.param_model.gaussian_cdf(t, param)
for i in range(n) :
    rel_error = result[i] / f(t[i], alpha, beta, p) - 1.0
    assert abs(rel_error) < eps99
#
# param a matrix of a_double
alpha  = a_double(0.1) + numpy.arange(n, dtype = float)
beta   = a_double(0.2) + numpy.arange(n, dtype = float)
p      = a_double(0.3) + numpy.arange(n, dtype = float)
param  = numpy.vstack( (alpha, beta, p ) )
result = curvefit.core.param_model.gaussian_cdf(t, param)
for i in range(n) :
    assert type(t[i]) == numpy.double
    assert type(result[i]) == a_double
    check      = f(t[i], alpha[i], beta[i], p[i])
    rel_error  = result[i] / check - 1.0
    assert abs(rel_error.value()) < eps99
#
#
print('param_model.py: OK')
sys.exit(0)
"""```
{end_markdown param_model_xam}
"""
