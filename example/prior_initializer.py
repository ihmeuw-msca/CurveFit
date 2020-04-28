#! /usr/bin/env/python3
"""
{begin_markdown prior_initializer_xam}
{spell_markdown arange, gprior, concat, covs, func, init, initialization, Initializer}

# Example of Prior Initializer

## Function Documentation
See [PriorInitializer](PriorInitializer.md) for function documentation.

## Example

In this example, we will simulate data from the `curvefit.core.functions.ln_gaussian_pdf` with `n_groups` different
groups random effects, and fit a `PriorInitializer` to these groups to get
a new `ParameterSet` to be used for fitting other groups. The idea is that if you have more complete time series'
for some groups, then you could first fit a model(s) to these groups alone, extract information about
fixed effects means, random effects variances, etc., and then use that information to set better priors for
the groups that have less complete time series'.

This example shows how to generate a **new** `ParameterSet` using `PriorInitializer` that could then be used
to fit other models.

First, we need to import all of the packages and modules that we will use:

```python """
import numpy as np
import pandas as pd

from curvefit.core.functions import ln_gaussian_pdf, normal_loss
from curvefit.core.data import Data
from curvefit.core.parameter import ParameterSet, Parameter, Variable
from curvefit.initializer.initializer import PriorInitializer
from curvefit.initializer.initializer_component import BetaPrior
from curvefit.models.core_model import CoreModel
from curvefit.solvers.solvers import ScipyOpt
""" ```
### Simulate Data
Now we will set the simulation parameters. We will simulate 10 groups with 10 time points each, and their
parameters are in the order `alpha`, `beta`, and `p` for the `ln_gaussian_pdf` functional form. The 
random effects variance will be used to simulate parameters from the mean `alpha`, `beta`, and `p` for each group.
```python """
# simulation parameters
np.random.seed(10)

time = np.arange(10)
n_groups = 10

fe_mean = np.array([1, 3., 1.])
re_var = np.array([0.01, 0.1, 0.1])

# fe_gprior and re_gprior for the solver
fe_gprior = [0., np.inf]
re_gprior = [0., np.inf]

# how much noise to add to the data
data_noise = 1.
""" ```
Define a function to simulate data based on these parameters, and create simulated data.
```python """
def simulate_data():
    re_mean = np.random.normal(fe_mean, re_var, size=(n_groups, len(fe_mean)))
    group_names = [f"group_{i}" for i in range(n_groups)]

    groups = []
    for i in range(n_groups):
        group_sim = ln_gaussian_pdf(time, re_mean[i, :]) + np.random.normal(0, data_noise, size=len(time))
        groups.append(pd.DataFrame({
            'group_name': group_names[i],
            'obs': group_sim,
            'time': time
        }))
    group_data = pd.concat(groups)
    group_data['intercept'] = 1
    return group_data

df = simulate_data()

# Use the `Data` object to store information about what the columns represent
data = Data(
    df=df, col_t='time', col_obs='obs', col_covs=['intercept'],
    obs_space=ln_gaussian_pdf, col_group='group_name', obs_se_func=lambda x: 1
)
""" ```
### Create a Parameter Set
We need to define variables, parameters that use those
variables (in this case there are only intercepts -- and one covariate per parameter -- so the variables
are effectively the same as the parameters), and a parameter set that collects all of that information into one
object.

```python """
alpha_fe = Variable(
    covariate='intercept',
    var_link_fun=lambda x: x,
    fe_init=fe_mean[0], re_init=0.,
    fe_gprior=fe_gprior,
    re_gprior=re_gprior
)
beta_fe = Variable(
    covariate='intercept',
    var_link_fun=lambda x: x,
    fe_init=fe_mean[1], re_init=0.,
    fe_gprior=fe_gprior,
    re_gprior=re_gprior
)
p_fe = Variable(
    covariate='intercept',
    var_link_fun=lambda x: x,
    fe_init=fe_mean[2], re_init=0.,
    fe_gprior=fe_gprior,
    re_gprior=re_gprior
)
alpha = Parameter('alpha', link_fun=np.exp, variables=[alpha_fe])
beta = Parameter('beta', link_fun=lambda x: x, variables=[beta_fe])
p = Parameter('p', link_fun=lambda x: x, variables=[p_fe])

params = ParameterSet([alpha, beta, p])
""" ```

### Models and Solvers
We now need a `Model` that contains our parameter set and also which curve function we want to fit, and the loss
function for the optimization. We also need to define a `Solver` that will actually perform the optimization.
Finally, we will fit the solver to our simulated data.

```python """
model = CoreModel(
    param_set=params,
    curve_fun=ln_gaussian_pdf,
    loss_fun=normal_loss
)
solver = ScipyOpt()
""" ```

### Use the Prior Initializer
The purpose of using the prior initializer is that it gets smarter priors for later on. Therefore, there are many
different things that one could do to get good priors based on subsets of the data. Each of these types
is implemented in a separate [`PriorInitializerComponent`](PriorInitializerComponent.md). Here we will use
the `BetaPrior` prior initializer component; the goal is to estimate the mean and variance of the random effects
and then set that as the fixed effects prior variance for a **new** parameter set that can be used in later model fits.

Even though we only have 10 groups, our random effect variance should get pretty close to our simulated value.
The `BetaPrior()` will update the fixed effects Gaussian prior for the `beta` parameter to what is estimated based on
a joint model fit with random effects.

```python """
# Instantiate a PriorInitializer with one component
pi = PriorInitializer(prior_initializer_components=[BetaPrior()])

# Create a new parameter set from running the initialization
new_param_set = pi.initialize(
    data=data,
    model_prototype=model,
    solver_prototype=solver
)

# Make sure that the estimated variance is close to the simulated variance
np.testing.assert_almost_equal(
    new_param_set.fe_gprior[1], [fe_mean[1], re_var[1]], decimal=1
)
print('prior_initializer.py: OK')
""" ```
{end_markdown prior_initializer_xam}
"""