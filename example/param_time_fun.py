#! /usr/bin/env python3
'''{begin_markdown param_time_fun_xam}
{spell_markdown finfo sqrt eval expit erf param params vstack derf dderf}

# Example and Test of Predefined Parametric Functions of Time

## Function Documentation
[param_time_fun](param_time_fun.md)

## Example Source Code
```python'''
import sys
import numpy
import scipy
import sandbox
sandbox.path()
import curvefit
#
eps99       = 99.0 * numpy.finfo(float).eps
sqrt_eps    = numpy.sqrt( numpy.finfo(float).eps )
quad_eps    = numpy.sqrt( sqrt_eps )
d_tolerance = 1e-6
#
def eval_expit(t, alpha, beta, p) :
    return p / ( 1.0 + numpy.exp( - alpha * (t - beta) ) )
#
def eval_erf(t, alpha, beta, p) :
    z   = alpha * (t - beta)
    return p * ( 1.0 + scipy.special.erf(z) ) / 2.0
#
# test values for t, alpha, beta, p
t      = numpy.array( [ 5.0 , 10.0 ] )
beta   = numpy.array( [ 30.0 , 20.0 ] )
alpha  = 2.0 / beta
p      = numpy.array( [ 0.1, 0.2 ] )
params = numpy.vstack( (alpha, beta, p) )
#
# check expit
value     = curvefit.core.functions.expit(t, params)
check     = eval_expit(t, alpha, beta, p)
rel_error = value / check - 1.0
assert all( abs( rel_error ) < eps99 )
#
# check log_expit
value     = curvefit.core.functions.log_expit(t, params)
check     = numpy.log(check)
rel_error = value / check - 1.0
assert all( abs( rel_error ) < eps99 )
#
# check erf
value     = curvefit.core.functions.erf(t, params)
check     = eval_erf(t, alpha, beta, p)
rel_error = value / check - 1.0
assert all( abs( rel_error ) < eps99 )
#
# check log_erf
value     = curvefit.core.functions.log_erf(t, params)
check     = numpy.log(check)
rel_error = value / check - 1.0
assert all( abs( rel_error ) < eps99 )
#
# check derf
step      = sqrt_eps * beta
value     = curvefit.core.functions.derf(t, params)
check_m   = eval_erf(t - step, alpha, beta, p)
check_p   = eval_erf(t + step, alpha, beta, p)
check     = (check_p - check_m) / (2.0 * step)
rel_error = value / check - 1.0
assert all( abs( rel_error ) < d_tolerance )
#
# check log_derf
value     = curvefit.core.functions.log_derf(t, params)
check     = numpy.log(check)
rel_error = value / check - 1.0
assert all( abs( rel_error ) < d_tolerance )
#
# check_dderf
step      = quad_eps * beta
value     = curvefit.core.functions.dderf(t, params)
check_m   = eval_erf(t - step, alpha, beta, p)
check_0   = eval_erf(t,        alpha, beta, p)
check_p   = eval_erf(t + step, alpha, beta, p)
check     = (check_p - 2.0 * check_0 + check_m) / step**2
rel_error = value / check - 1.0
assert all( abs( rel_error ) < d_tolerance )
#
print('param_time_fun.py: OK')
sys.exit(0)
'''```
{end_markdown param_time_fun_xam}'''
