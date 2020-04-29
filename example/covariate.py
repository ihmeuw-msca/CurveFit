#! /usr/bin/env python3
"""
{begin_markdown covariate_xam}
{spell_markdown
    params
    param
    covs
    init
    ftol
    gtol
    optimizer
    allclose
    rtol
    erf
}

# Using Covariates

## Generalized Gaussian Cumulative Distribution Function
The model for the mean of the data for this example is:
\[
    f(t; \alpha, \beta, p) =
        \frac{p}{2} \left( 1 + \frac{2}{\pi} \int_0^{\alpha ( t - \beta )}
            \exp( - \tau^2 ) d \tau \right)
\]
where \( \alpha \), \( \beta \), and \( p \) are unknown parameters.
In addition, the value of \( \beta \) depends on covariate.


## Fixed Effects
We use the notation \( a \), \( b \), \( c \) and \( \phi \)
for the fixed effects corresponding to the parameters
\( \alpha \), \( \beta \), and \( p \).
For this example, the link functions, that map from the fixed
effects to the parameters, are
\[
\begin{aligned}
    \alpha & = \exp( a ) \\
    \beta  & =  b + c \cdot s \\
    p      & = \exp( \phi  )
\end{aligned}
\]
where \( s \) is the social distance covariate.

## Random effects
For this example the random effects are constrained to be zero.

## Social Distance
For this simulation, the social distance covariate has two values:
\[
    s_i = \left\{ \begin{array}{ll}
        0 & \mbox{if} \; i < n_D / 2 \\
        1 & \mbox{otherwise}
    \end{array} \right.
\]

## Simulated data

### Problem Settings
The following settings are used to simulate the data and check
that the solution is correct:
```python """
import math

n_data = 21  # number simulated measurements to generate
b_true = 20.0  # b used to simulate data
a_true = math.log(2.0 / b_true)  # a used to simulate data
c_true = 1.0 / b_true  # c used to simulate data
phi_true = math.log(0.1)  # phi used to simulate data
rel_tol = 1e-5  # relative tolerance used to check optimal solution
"""```
The fixed effects
\( a \), \( b \), \( c \), and \( \phi \)
are initialized so that they correspond to
the true fixed effects divided by three.

### Time Grid
A grid of *n_data* points in time, \( t_i \), where
\[
    t_i = b_T / ( n_D - 1 )
\]
where the subscript \( T \) denotes the true value
of the corresponding parameter and \( n_D \) is the number of data points.
The minimum value for this grid is zero and its maximum is \( b_T \).

### Measurement Values
We simulate data, \( y_i \), with no noise at each of the time points.
To be specific, for \( i = 0 , \ldots , n_D - 1 \)
\[
    y_i = f( t_i , \alpha_T , b_T + c_T \cdot s_i , p_T )
\]
Note that when we do the fitting, we model each data point as having noise.


## Example Source Code
```python """
# -------------------------------------------------------------------------
import sys
import pandas
import numpy
import scipy

from curvefit.core.functions import gaussian_cdf, normal_loss
from curvefit.core.data import Data
from curvefit.core.parameter import Variable, Parameter, ParameterSet
from curvefit.models.core_model import CoreModel
from curvefit.solvers.solvers import ScipyOpt


# link function used for beta
def identity_fun(x):
    return x


# link function used for alpha, p
def exp_fun(x):
    return numpy.exp(x)


# inverse of function used for alpha, p
def log_fun(x):
    return numpy.log(x)


# true value for fixed effects
fe_true = numpy.array([a_true, b_true, c_true, phi_true])
num_fe = len(fe_true)

# -----------------------------------------------------------------------
# data_frame

independent_var = numpy.array(range(n_data)) * b_true / (n_data - 1)
social_distance = numpy.zeros(n_data, dtype=float)
params_value = numpy.zeros((n_data, 3), dtype=float)
alpha_true = numpy.exp(a_true)
p_true = numpy.exp(phi_true)
for i in range(n_data):
    social_distance[i] = 0 if i < n_data / 2.0 else 1
    beta_true = b_true + c_true * social_distance[i]
    params_value[i] = [alpha_true, beta_true, p_true]
params_value = numpy.transpose(params_value)
measurement_value = gaussian_cdf(independent_var, params_value)
measurement_std = n_data * [0.1]
cov_one = n_data * [1.0]
data_group = n_data * ['world']
data_dict = {
    'independent_var': independent_var,
    'measurement_value': measurement_value,
    'measurement_std': measurement_std,
    'cov_one': cov_one,
    'social_distance': social_distance,
    'data_group': data_group,
}
data_frame = pandas.DataFrame(data_dict)

# ------------------------------------------------------------------------
# curve_model

data = Data(
    df=data_frame,
    col_t='independent_var',
    col_obs='measurement_value',
    col_covs=['cov_one', 'social_distance'],
    col_group='data_group',
    obs_space=gaussian_cdf,
    col_obs_se='measurement_std'
)

a_intercept = Variable(
    covariate='cov_one',
    var_link_fun=identity_fun,
    fe_init=a_true / 3,
    re_init=0.0,
    fe_bounds=[-numpy.inf, numpy.inf],
    re_bounds=[0.0, 0.0]
)

b_intercept = Variable(
    covariate='cov_one',
    var_link_fun=identity_fun,
    fe_init=b_true / 3,
    re_init=0.0,
    fe_bounds=[-numpy.inf, numpy.inf],
    re_bounds=[0.0, 0.0]
)

b_social_distance = Variable(
    covariate='social_distance',
    var_link_fun=identity_fun,
    fe_init=b_true / 3,
    re_init=0.0,
    fe_bounds=[-numpy.inf, numpy.inf],
    re_bounds=[0.0, 0.0]
)


phi_intercept = Variable(
    covariate='cov_one',
    var_link_fun=identity_fun,
    fe_init=p_true / 3,
    re_init=0.0,
    fe_bounds=[-numpy.inf, numpy.inf],
    re_bounds=[0.0, 0.0]
)

alpha = Parameter(param_name='alpha', link_fun=exp_fun, variables=[a_intercept])
beta = Parameter(param_name='beta', link_fun=identity_fun, variables=[b_intercept, b_social_distance])
p = Parameter(param_name='p', link_fun=exp_fun, variables=[phi_intercept])

parameters = ParameterSet([alpha, beta, p])

optimizer_options = {
    'ftol': 1e-12,
    'gtol': 1e-12,
}

model = CoreModel(
    param_set=parameters,
    curve_fun=gaussian_cdf,
    loss_fun=normal_loss
)
solver = ScipyOpt(model)
solver.fit(data=data._get_df(copy=True, return_specs=True), options=optimizer_options)
params_estimate = model.get_params(solver.x_opt, expand=True)

# -------------------------------------------------------------------------
# check result

for i in range(3):
    assert numpy.allclose(params_estimate[i], params_value[i], rtol=rel_tol)

print('covariate.py: OK')

""" ```
{end_markdown covariate_xam}
"""
