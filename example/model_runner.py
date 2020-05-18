#! /usr/bin/env/python3
"""
{begin_markdown model_runner_xam}
{spell_markdown modeling initializer arange gprior concat covs func init}

# Example and Test of `ModelRunner`

`ModelRunner` is the main class that connects all of the individual components of a `curvefit`
model into a larger modeling pipeline. Here we will create examples of the building
blocks that are necessary for a ModelRunner.

The examples begin with an identical setup to the example in [prior initializer](prior_initializer_xam.md).
We will not use the prior initializer in this overall example, but it is an optional argument
to the `ModelRunner`.

## Function Documentation
Please see function documentation here: [`ModelRunner`][ModelRunner.md].

## Importing Packages
```python """
import numpy as np
import pandas as pd

from curvefit.core.functions import ln_gaussian_pdf, normal_loss
from curvefit.core.data import Data
from curvefit.core.parameter import ParameterSet, Parameter, Variable
from curvefit.models.core_model import CoreModel
from curvefit.solvers.solvers import ScipyOpt
from curvefit.uncertainty.predictive_validity import PredictiveValidity
from curvefit.uncertainty.residual_model import SmoothResidualModel
from curvefit.uncertainty.draws import Draws
from curvefit.run.model_run import ModelRunner
""" ```
### Simulate Data
Now we will set the simulation parameters. We will simulate 10 groups with 10 time points each, and their
parameters are in the order `alpha`, `beta`, and `p` for the `ln_gaussian_pdf` functional form. The 
random effects variance will be used to simulate parameters from the mean `alpha`, `beta`, and `p` for each group.
```python """
# simulation parameters
np.random.seed(10)

time = np.arange(20)
n_groups = 10

fe_mean = np.array([1., 10., 1.])
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
        truth = ln_gaussian_pdf(time, re_mean[i, :])
        group_sim = truth + np.random.normal(0, data_noise, size=len(time))
        groups.append(pd.DataFrame({
            'group_name': group_names[i],
            'obs': group_sim,
            'time': time,
            'truth': truth
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

### Uncertainty

For the uncertainty analysis, we need to specify three things: (1) a predictive validity object, (2) a residual model, 
and (3) a draws object.

#### Predictive Validity

The predictive validity object estimates out of sample residuals for every single data point in the dataset.
For example, if you have 10 data points, the predictive validity object will start with the first data point
in the time series, predict out the full time series, 
and then repeat by adding in the next time point and predicting out, etc. By the end, there is a square 
prediction matrix that can be converted to a residual matrix by subtracting off the observed data,
for each group of size the number of observations for that group. The upper triangular indices (excluding the diagonal)
are all of the **out of sample residuals**. Predictive validity provides a model agnostic way to get
uncertainty in the predictions.

To create a predictive validity object, you need to decide what space you want to evaluate the predictions.
This space does *not* need to be the space that you're fitting the model in.

```python """
pv = PredictiveValidity(evaluation_space=ln_gaussian_pdf)
""" ```

#### Residual Model

The residual model is a predictive model for the residuals that are obtained from doing predictive validity.
Right now there is only one type of residual model implemented. A `SmoothResidualModel`, with a description
of the methods in the [function documentation](SmoothResidualModel.md). The purpose is to understand how
the coefficient of variation of the out of sample predictions from predictive validity changes with key factors.

You need to pass
bounds on the coefficient of variation of the predictions (can be `[-np.inf, np.inf]` to not pass bounds),
a dictionary of covariates to use in predicting the residuals which for this model needs to be `"far_out"`
and `"num_data"`. You also need a smoothing radius that defines a "neighborhood" based on those covariates,
and how many times to run a local smoother over the residuals. 

```python """
rm = SmoothResidualModel(
    cv_bounds=[1e-6, np.inf],
    covariates={'far_out': None, 'num_data': None},
    num_smooth_iterations=1,
    smooth_radius=[2, 2],
    robust=True
)
""" ```

#### Draws

The predictive validity object and residual model come together to create uncertainty in the predictions
in the `Draws` class. These are the final predictions. You can specify how many draws of the time series you want
and for which time series: `num_draws` and `prediction_times`.

```python """
draws = Draws(
    num_draws=1000,
    prediction_times=np.arange(0., 25.)
)
""" ```

### Model Runner

Using all of the objects we've created so far we can instantiate a model runner.

```python """
mr = ModelRunner(
    data=data,
    model=model,
    solver=solver,
    residual_model=rm,
    predictive_validity=pv,
    draws=draws
)

mr.run()
""" ```

We can now inspect the results for each group.

```python """
# The residual matrix for group_9 from predictive validity analysis
mr.predictive_validity.group_residuals['group_9'].residual_matrix

# The draws for group_9
mr.draws.get_draws('group_9')

# The mean estimates for group_9 (the first element is the mean)
mr.draws.get_draws_summary('group_9')[0]

# Example check that the mean estimate for this group are approximately the same as the truth
for group in df.group_name.unique():
    np.testing.assert_array_almost_equal(
        df.loc[df.group_name == group].truth.values,
        mr.draws.get_draws_summary(group)[0][:20],
        decimal=0
    )

""" ```

```python """

print('model_runner.py: OK')
""" ```
{end_markdown model_runner_xam}
"""