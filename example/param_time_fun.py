#! /usr/bin/env python3
'''{begin_markdown param_time_fun_xam}
{spell_markdown finfo sqrt eval expit erf param params vstack dgaussian}

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
def eval_gaussian_cdf(t, alpha, beta, p) :
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
# check ln_expit
value     = curvefit.core.functions.ln_expit(t, params)
check     = numpy.log(check)
rel_error = value / check - 1.0
assert all( abs( rel_error ) < eps99 )
#
# check gaussian_cdf
value     = curvefit.core.functions.gaussian_cdf(t, params)
check     = eval_gaussian_cdf(t, alpha, beta, p)
rel_error = value / check - 1.0
assert all( abs( rel_error ) < eps99 )
#
# check ln_gaussian_cdf
value     = curvefit.core.functions.ln_gaussian_cdf(t, params)
check     = numpy.log(check)
rel_error = value / check - 1.0
assert all( abs( rel_error ) < eps99 )
#
# check gaussian_pdf
step      = sqrt_eps * beta
value     = curvefit.core.functions.gaussian_pdf(t, params)
check_m   = eval_gaussian_cdf(t - step, alpha, beta, p)
check_p   = eval_gaussian_cdf(t + step, alpha, beta, p)
check     = (check_p - check_m) / (2.0 * step)
rel_error = value / check - 1.0
assert all( abs( rel_error ) < d_tolerance )
#
# check ln_gaussian_pdf
value     = curvefit.core.functions.ln_gaussian_pdf(t, params)
check     = numpy.log(check)
rel_error = value / check - 1.0
assert all( abs( rel_error ) < d_tolerance )
#
# check_dgaussian_pdf
step      = quad_eps * beta
value     = curvefit.core.functions.dgaussian_pdf(t, params)
check_m   = eval_gaussian_cdf(t - step, alpha, beta, p)
check_0   = eval_gaussian_cdf(t,        alpha, beta, p)
check_p   = eval_gaussian_cdf(t + step, alpha, beta, p)
check     = (check_p - 2.0 * check_0 + check_m) / step**2
rel_error = value / check - 1.0
assert all( abs( rel_error ) < d_tolerance )
#
print('param_time_fun.py: OK')
sys.exit(0)
'''```
{end_markdown param_time_fun_xam}'''
