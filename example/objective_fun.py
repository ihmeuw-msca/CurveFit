#! /usr/bin/env python3
"""
{begin_markdown objective_fun_xam}
{spell_markdown
    param
    arange
    concatenate
    covs
    gprior
    params
    finfo
    obj
}

# Example and Test of `objective_fun`

## Function Documentation
[objective_fun][objective_fun.md]

## Example Source Code
```python """
import sys
import numpy
from curvefit.core.functions import gaussian_cdf
from curvefit.core.objective_fun import objective_fun
from curvefit.core.effects2params import effects2params, unzip_x

# -----------------------------------------------------------------------
# Test parameters
num_param = 3
num_group = 2

# -----------------------------------------------------------------------
# arguments to objective_fun
def identity_fun(x):
    return x


def gaussian_loss(x):
    return numpy.sum(x * x) / 2.0


# set parameters for the objective function
num_fe = num_param
num_re = num_group * num_fe
fe     = numpy.array(range(num_fe), dtype=float) / num_fe
re     = numpy.array(range(num_re), dtype=float) / num_re
group_sizes = (numpy.arange(num_group) + 1) * 2

# observations
x       = numpy.concatenate((fe, re))
num_obs = sum(group_sizes)
t       = numpy.arange(num_obs, dtype=float)
obs     = numpy.arange(num_obs, dtype=float) / num_obs
obs_se  = (obs + 1.0) / 10.0

# covariates
covs = list()
for k in range(num_param):
    covs.append(numpy.ones((num_obs, 1), dtype=float))

# fit, loss and link functions
model_fun = gaussian_cdf
loss_fun = gaussian_loss
link_fun = [numpy.exp, identity_fun, numpy.exp]
var_link_fun = num_param * [identity_fun]

# fixed effects Gaussian prior
fe_gprior = numpy.empty((num_fe, 2), dtype=float)
for j in range(num_fe):
    fe_gprior[j, 0] = j / (2.0 * num_fe)
fe_gprior[:, 1] = 1.0 + fe_gprior[:, 0] * 1.2

# random effects Gaussian prior
re_gprior = numpy.empty((num_fe, num_group, 2), dtype=float)

# parameter functional Gaussian prior
param_gprior_mean = numpy.empty((num_param, num_group), dtype=float)
for i in range(num_group):
    for j in range(num_fe):
        # the matrix re_gprior[:,:,0] is the transposed from the order in re
        re_gprior[j, i, 0] = (i + j) / (2.0 * (num_fe + num_re))
        k = j
        param_gprior_mean[k, i] = (i + k) / (3.0 * (num_fe + num_re))

re_gprior[:, :, 1] = (1.0 + re_gprior[:, :, 0] / 3.0)
param_gprior_std = (1.0 + param_gprior_mean / 2.0)
param_gprior_fun = identity_fun
param_gprior = [param_gprior_fun, param_gprior_mean, param_gprior_std]
re_zero_sum_std = numpy.array( num_fe * [numpy.inf] )
# -----------------------------------------------------------------------
# call to objective_fun
obj_val = objective_fun(
    x,
    t,
    obs,
    obs_se,
    covs,
    group_sizes,
    model_fun,
    loss_fun,
    link_fun,
    var_link_fun,
    fe_gprior,
    re_gprior,
    param_gprior,
    re_zero_sum_std,
)
# -----------------------------------------------------------------------
# check objective_fun return value
expand = True
param = effects2params(x, group_sizes, covs, link_fun, var_link_fun, expand)
fe, re = unzip_x(x, num_group, num_fe)
obs_res = (obs - model_fun(t, param)) / obs_se
fe_res = (fe.T - fe_gprior[:, 0]) / fe_gprior[:, 1]
re_res = (re.T - re_gprior[:, :, 0]) / re_gprior[:, :, 1]

expand = False
param = effects2params(x, group_sizes, covs, link_fun, var_link_fun, expand)
range_gprior = param_gprior[0](param)
param_res = (range_gprior - param_gprior[1][0]) / param_gprior[1][1]

check = loss_fun(obs_res) + gaussian_loss(fe_res)
check += gaussian_loss(re_res) + gaussian_loss(param_res)

rel_error = obj_val / check - 1.0
eps99 = 99.0 * numpy.finfo(float).eps
assert abs(rel_error) < eps99

print('objective_fun.py: OK')

""" ```
{end_markdown objective_fun_xam}
"""
