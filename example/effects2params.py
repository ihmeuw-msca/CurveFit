#! /usr/bin/env python3
"""
{begin_markdown effects2params_xam}
{spell_markdown params param dtype arange covs finfo}

# Example and Test of effects2params}

## Function Documentation
[effects2params](effects2params.md)

## Example Source Code
```python
"""
import numpy
from curvefit.core.effects2params import effects2params
# -----------------------------------------------------------------------
# Test parameters
num_param = 3
num_group = 2


# -----------------------------------------------------------------------
def identity_fun(x):
    return x


num_fe = num_param
num_x = (num_group + 1) * num_fe
x = numpy.array(range(num_x), dtype=float) / num_x
group_sizes = numpy.arange(num_group) * 2 + 1
num_obs = sum(group_sizes)
covs = list()
for k in range(num_param):
    covs.append(numpy.ones((num_obs, 1), dtype=float))
link_fun = [numpy.exp, identity_fun, numpy.exp]
var_link_fun = num_param * [identity_fun]
expand = False
param = effects2params(
    x, group_sizes, covs, link_fun, var_link_fun, expand
)
# ----------------------------------------------------------------------
# check result
eps99 = 99.0 * numpy.finfo(float).eps
fe = x[0: num_fe]
re = x[num_fe:].reshape((num_group, num_fe), order='C')
fe_re = fe + re
var = numpy.empty((num_group, num_fe), dtype=float)
for j in range(num_fe):
    var[:, j] = var_link_fun[j](fe_re[:, j])
check = numpy.empty((num_param, num_group), dtype=float)
for k in range(num_param):
    check[k, :] = link_fun[k](var[:, k] * covs[k][0])
#
rel_error = param / check - 1.0
assert ((abs(rel_error) < eps99).all())
print('effects2params.py: OK')

"""```
{end_markdown effects2params_xam}
"""
