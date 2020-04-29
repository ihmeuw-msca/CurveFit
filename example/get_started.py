#! /usr/bin/env python3
"""
{begin_markdown get_started_xam}
{spell_markdown
    params
    covs
    param
    inv
    init
    finfo
    py
    ftol
    gtol
    optimizer
    expit
}

# Getting Started Using CurveFit

## Generalized Logistic Model
The model for the mean of the data for this example is one of the following:
\[
    f(t; \alpha, \beta, p)  = \frac{p}{1 + \exp [ -\alpha(t  - \beta) ]}
\]
where \( \alpha \), \( \beta \), and \( p \) are unknown parameters.


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
The fixed effects are initialized to be their true values divided by three.

## Random effects
For this example the random effects are constrained to be zero.

## Covariates
This example data set has two covariates,
the constant one and a social distance measure.
While the social distance is in the data set, it is not used.

## Simulated data

### Problem Settings
The following settings are used to simulate the data and check
that the solution is correct:
```python """
n_data = 21  # number simulated measurements to generate
beta_true = 20.0  # max death rate at 20 days
alpha_true = 2.0 / beta_true  # alpha_true * beta_true = 2.0
p_true = 0.1  # maximum cumulative death fraction
rel_tol = 1e-5  # relative tolerance used to check optimal solution
"""```

### Time Grid
A grid of *n_data* points in time, \( t_i \), where
\[
    t_i = \beta_T / ( n_D - 1 )
\]
where the subscript \( T \) denotes the true value
of the corresponding parameter and \( n_D \) is the number of data points.
The minimum value for this grid is zero and its maximum is \( \beta \).

### Measurement values
We simulate data, \( y_i \), with no noise at each of the time points.
To be specific, for \( i = 0 , \ldots , n_D - 1 \)
\[
    y_i = f( t_i , \alpha_T , \beta_T , p_T )
\]
Note that when we do the fitting, we model each data point as having
noise.

## Example Source Code
```python """
# -------------------------------------------------------------------------
import pandas
import numpy

from curvefit.core.functions import expit, normal_loss
from curvefit.core.data import Data
from curvefit.core.parameter import Variable, Parameter, ParameterSet
from curvefit.models.core_model import CoreModel
from curvefit.solvers.solvers import ScipyOpt


# for this model number of parameters is same as number of fixed effects
num_params = 3
num_fe = 3

params_true = numpy.array([alpha_true, beta_true, p_true])

# -----------------------------------------------------------------------
# data_frame

independent_var = numpy.array(range(n_data)) * beta_true / (n_data - 1)
measurement_value = expit(independent_var, params_true)
measurement_std = n_data * [0.1]
constant_one = n_data * [1.0]
social_distance = [0.0 if i < n_data / 2 else 1.0 for i in range(n_data)]
data_group = n_data * ['world']

data_dict = {
    'independent_var': independent_var,
    'measurement_value': measurement_value,
    'measurement_std': measurement_std,
    'constant_one': constant_one,
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
    col_covs=num_params * ['constant_one'],
    col_group='data_group',
    obs_space=expit,
    col_obs_se='measurement_std'
)

alpha_intercept = Variable(
    covariate='constant_one',
    var_link_fun=lambda x: x,
    fe_init=numpy.log(alpha_true) / 3,
    re_init=0.0,
    fe_bounds=[-numpy.inf, numpy.inf],
    re_bounds=[0.0, 0.0]
)

beta_intercept = Variable(
    covariate='constant_one',
    var_link_fun=lambda x: x,
    fe_init=beta_true / 3,
    re_init=0.0,
    fe_bounds=[-numpy.inf, numpy.inf],
    re_bounds=[0.0, 0.0]
)

p_intercept = Variable(
    covariate='constant_one',
    var_link_fun=lambda x: x,
    fe_init=numpy.log(p_true) / 3,
    re_init=0.0,
    fe_bounds=[-numpy.inf, numpy.inf],
    re_bounds=[0.0, 0.0]
)

alpha = Parameter(param_name='alpha', link_fun=numpy.exp, variables=[alpha_intercept])
beta = Parameter(param_name='beta', link_fun=lambda x: x, variables=[beta_intercept])
p = Parameter(param_name='p', link_fun=numpy.exp, variables=[p_intercept])

parameters = ParameterSet([alpha, beta, p])

optimizer_options = {
    'ftol': 1e-12,
    'gtol': 1e-12,
}

model = CoreModel(
    param_set=parameters,
    curve_fun=expit,
    loss_fun=normal_loss
)
solver = ScipyOpt(model)
solver.fit(data=data._get_df(copy=True, return_specs=True), options=optimizer_options)
params_estimate = model.get_params(solver.x_opt, expand=False)

# -------------------------------------------------------------------------
# check optimal parameters

for i in range(num_params):
    rel_error = params_estimate[i] / params_true[i] - 1.0
    assert abs(rel_error) < rel_tol

print('get_started_old.py: OK')

""" ```
{end_markdown get_started_xam}
"""
