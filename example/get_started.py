#! /bin/python3
# vim: set expandtab:
'''
[begin_markdown get_started_xam]

# Getting Started Using CurveFit

## Data Mean
The model for the mean of the data for this example is:
\[
    f(t; \alpha, \beta, p)  = \frac{p}{1 + \exp [ -\alpha(t  - \beta) ]}
\]
where \( \alpha \), \( \beta \), and \( p \) are unknown parameters.

## Problem Settings
The following settings are used to simulate the data and check
that the solution is correct:
```python '''
n_data       = 21    # number simulated measurements to generate
alpha_true   = 2.0   # values of alpha, beta, p, used to simulate data
beta_true    = 3.0
p_true       = 4.0
rel_tol      = 1e-6  # relative tolerance used to check optimal solution
'''```

## Simulated data

### Time Grid
A grid of *n_data* points in time, \( t_i \), where
\[
    t_i = \beta_T / ( n_D - 1 )
\]
where the subscript \( T \) to denotes the true value
of the currespondng parameter
and \( n_D \) is the number of data points.
The minimum value is zero for this grid is zero and its maximum is \( \beta \).

### Measurement values
We simulate data, \( y_i \), with no noise at each of the time points.
To be specific, for \( i = 0 , \ldots , n_D - 1 \)
\[
    y_i = f( t_i , \alpha_T , \beta_T , p_T )
\]
Note that when we do the fitting, we model each data point as having
noise.

## Fixed Effects
We use the notation \( a \), \( b \) and \( \phi \)
for the fixed effect corresponding to the parameters
\( \alpha \), \( \beta \), \( p \) respectively.
For this example, the link functions, that map from the fixed
effects to the parameters, are
\[
\begin{aligned}
    \alpha & = \exp( a ) \\
    \beta  & =  b \\
    p      & = \exp( \phi  )
\end{aligned}
\]


## Source Code
```python '''
# -------------------------------------------------------------------------
import sys
import pandas
import numpy
import sandbox
sandbox.path()
import curvefit
#
# number of parameters in this model
num_params   = 3
#
# model for the mean of the data
def generalized_logistic(t, params) :
    alpha = params[0]
    beta  = params[1]
    p     = params[2]
    return p / ( 1.0 + numpy.exp( - alpha * ( t - beta ) ) )
#
# link function used for beta
def identity_fun(x) :
    return x
#
# link function used for alpha, p
def exp_fun(x) :
    return numpy.exp(x)
#
# inverse of function used for alpha, p
def log_fun(x) :
    return numpy.log(x)
#
# params_true
params_true       = numpy.array( [ alpha_true, beta_true, p_true ] )
#
# data_frame
independent_var   = numpy.array(range(n_data)) * beta_true / (n_data-1)
measurement_value = generalized_logistic(independent_var, params_true)
measurement_std   = n_data * [ 0.1 ]
constant_one      = n_data * [ 1.0 ]
data_group        = n_data * [ 'world' ]
data_dict         = {
    'independent_var'   : independent_var   ,
    'measurement_value' : measurement_value ,
    'measurement_std'   : measurement_std   ,
    'constant_one'      : constant_one      ,
    'data_group'        : data_group        ,
}
data_frame        = pandas.DataFrame(data_dict)
#
# curve_model
col_t        = 'independent_var'
col_obs      = 'measurement_value'
col_covs     = num_params *[ [ 'constant_one' ] ]
col_group    = 'data_group'
param_names  = [ 'alpha', 'beta',       'p'     ]
link_fun     = [ exp_fun, identity_fun, exp_fun ]
var_link_fun = link_fun
fun          = generalized_logistic
col_obs_se   = 'measurement_std'
#
curve_model = curvefit.core.model.CurveModel(
    data_frame,
    col_t,
    col_obs,
    col_covs,
    col_group,
    param_names,
    link_fun,
    var_link_fun,
    fun,
    col_obs_se
)
#
# fit_params
inv_link_fun = [ log_fun, identity_fun, log_fun ]
fe_init      = numpy.zeros( num_params )
for i in range(num_params) :
    fe_init[i]   = inv_link_fun[i](params_true[i] / 3.0)
curve_model.fit_params(fe_init)
params_estimate = curve_model.params
#
for i in range(num_params) :
    rel_error = params_estimate[i] / params_true[i] - 1.0
    assert abs(rel_error) < rel_tol
#
print('get_started.py: OK')
sys.exit(0)
''' ```
[end_markdown get_started_xam]
'''
