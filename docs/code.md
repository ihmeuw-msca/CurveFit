# Map of the Code

We first start by walking through the [core curve fitting model](#core-model), 
and then the extensions that make
it possible for `CurveFit` to be used for forecasting 
over time including [pipelines](#pipelines) and [predictive validity](#predictive-validity).

## Core Model
**`curevefit.core`**

Here we will walk through how to use `CurveModel`.

First, here is an example that you can copy
and paste into your Python interpreter to run start to finish.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from curvefit.core.model import CurveModel
from curvefit.core.functions import ln_gaussian_cdf

np.random.seed(1234)

# Create example data -- both death rate and log death rate
df = pd.DataFrame()
df['time'] = np.arange(100)

df['death_rate'] = np.exp(.1 * (df.time - 20)) / (1 + np.exp(.1 * (df.time - 20))) + \
                   np.random.normal(0, 0.1, size=100).cumsum()
df['ln_death_rate'] = np.log(df['death_rate'])

df['group'] = 'all'
df['intercept'] = 1.0

# Set up the CurveModel
model = CurveModel(
    df=df,
    col_t='time',
    col_obs='ln_death_rate',
    col_group='group',
    col_covs=[['intercept'], ['intercept'], ['intercept']],
    param_names=['alpha', 'beta', 'p'],
    link_fun=[lambda x: x, lambda x: x, lambda x: x],
    var_link_fun=[lambda x: x, lambda x: x, lambda x: x],
    fun=ln_gaussian_cdf
)

# Fit the model to estimate parameters
model.fit_params(fe_init=[0, 0, 1.],
                 fe_gprior=[[0, np.inf], [0, np.inf], [1., np.inf]])

# Get predictions
y_pred = model.predict(
    t=df.time,
    group_name=df.group.unique()
)

# Plot results
plt.plot(df.time, y_pred, '-')
plt.plot(df.time, df.ln_death_rate, '.')
```

Now we will walk through each of the steps from above and explain how to use them in detail.

### Setting Up a Model

The code for the core curve fitting model is `curvefit.core.model.CurveModel`.
To initialize a `CurveModel`, you need a `pandas` data frame and information
about what type of model you want to fit. It needs to know which columns
represent what and some model parameters.

- `df (pd.DataFrame)`: data frame with all available information for the model
- `col_t (str)`: the column that indicates the independent variable
- `col_obs (str)`: the column that indicates the dependent variable
- `col_covs (list{list{str}})`: list of lists of strings that indicate the covariates to use
    for each covariate
- `col_group (str)`: the column that indicates the group variable (even if you only have one group
    you must pass a column that indicates group membership)
- `param_names (list{str})`: names of the parameters for your specific functional form (more in [functions](#functions))
- `link_fun (list{function})`: list of link functions for each of the parameters
- `var_link_fun (list{function})`: list of functions for the variables including fixed and random effects

First, we create sample data frame where `time` is the independent variable,
`death_rate` is the dependent variable, and `group` is a variable indicating which group an observation belongs to.
In this example, we want to fit to the log erf (also referred to as log Gaussian CDF) functional form (see [functions](#functions)) with
identity link functions for each parameter and identity variable link functions for each parameter.
In this example, no parameters have covariates besides an intercept column of 1's.

```python
model = CurveModel(
    df=df,
    col_t='time',
    col_obs='ln_death_rate',
    col_group='group',
    col_covs=[['intercept'], ['intercept'], ['intercept']],
    param_names=['alpha', 'beta', 'p'],
    link_fun=[lambda x: x, lambda x: x, lambda x: x],
    var_link_fun=[lambda x: x, lambda x: x, lambda x: x],
    fun=ln_gaussian_cdf
)
```

### Functions

The `curvefit` package has some built-in functions for curves to fit. However, this list is not exhaustive,
and you may pass any callable function that takes in `t` (an independent variable) and `params` (a list of parameters)
to the function to the `CurveModel` class for the `fun` argument. What you pass in for `param_names` in the 
`CurveModel` needs to match what the `fun` callable expects.

The available built-in functions in `curvefit.core.functions` are:

**The Error Function**

- `gaussian_cdf`: Gaussian cumulative distribution function
- `gaussian_pdf`: Gaussian probability distribution function
- `ln_gaussian_cdf`: log Gaussian cumulative distribution function
- `ln_gaussian_pdf`: log Gaussian probability distribution function

**The Expit Function** (inverse of the logit function)

- `expit`: expit function
- `log_expit`: log expit function

Please see the [functions](methods.md#covid-19-functional-forms) for information
about the parametrization of these functions and how they relate to COVID-19 modeling.

### Fitting a Model

Once you have a model defined, the method `fit_params` fits the model. At minimum,
the only information that `model.fit_params` needs is initial values for the fixed effects.
But there are many optional arguments that you can pass to `model.fit_params` to inform
the optimization process. Below we describe each of these optional arguments.
The result of `fit_params` is stored in `CurveModel.result` and the
parameter estimates in `CurveModel.params`.

#### Gaussian Priors
`fe_gprior` and `re_gprior`

Each parameter may have an associated Gaussian prior. This is optional and can be
passed in as a list of lists. This specification, referring to [our example](#setting-up-a-model)
will put Gaussian priors with mean 0 and standard deviation 1. on the `alpha` parameter,
mean 0 and standard deviation 1e-3 on the `beta` parameter and mean 5 and standard
deviation 10. on the `p` parameter.

```python
model.fit_params(fe_gprior=[[0, 1.], [0, 1e-3], [5, 10.]])
```

Likewise, you may have random effects Gaussian priors using the argument `re_gprior`,
which has the same shape as `fe_gprior`, but refers to the random effects. For the
specifications of fixed and random effects, please see [the methods](methods.md#statistical-model).

#### Constraints
`fe_bounds` and `re_bounds`

You can also include parameter constraints for each of the fixed effects and the
random effects. They are included as a list of lists. This specification,
referring to [our example](#setting-up-a-model), will bound all of the fixed effects
between 0 and 100. and the random effects between -1 and 1.

```python
model.fit_params(
    fe_bounds=[[0., 100.], [0., 100.], [0., 100.]],
    re_bounds=[[-1., 1.], [-1., 1.], [-1., 1.]]
)
```

If you do not want to include random effects, set the bounds to be exactly 0. Please
see more information on constraints in [the methods](methods.md#constraints).

#### Initialization

The optimization routine will perform better with smart starting values for the
parameters. Initial values for the fixed effects, `fe_init`, are required and is passed in as
a `numpy` array of the same length as your parameters. The initial values for the random
effects, `re_init`, are passed in as a `numpy` array ordered by the group name and parameters.

For example, if you had two groups in the model, the following would initialize
the fixed effects at 1., 1., 1., and the random effects at -0.5, -0.5, -0.5, for the first group and
0.5, 0.5, 0.5, for the second group.

```python
import numpy as np

model.fit_params(
    fe_init=np.array([1., 1., 1.]),
    re_init=np.array([-0.5, -0.5, -0.5, 0.5, 0.5, 0.5])
)
```

There is an optional flag, `smart_initialize` that if `True` will run a model
individually for each group in the model and use their fixed effects estimates
to inform the initial values for both fixed and random effects of the mixed
model that you want to fit.

#### Optimization

The optimization uses `scipy.optimize.minimize` and the `"L-BFGS-B"` which has a list of options
that you can pass to it. These keyword options can be passed to the `minimize` function
using the `options` argument. For example, the following would perform a maximum
of 500 iterations and require an objective function tolerance of 1e-10.

```python
model.fit_params(
    options={
        'ftol': 1e-10,
        'maxiter': 500
    }
)
```

If you have indicated that you want the model to do smart initialization with
`smart_initialize = True`, then you can optionally pass a dictionary of `smart_init_options`
to override the `options` just for the group-specific initial fits. Otherwise
it will use all of the `options` in both the group-specific and overall fits.

Please see [`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)
for more options. Please see [the methods](methods.md#optimization-procedure) for more
information about the optimization procedure.

### Obtaining Predictions from a Model

To obtain predictions from a model that has been fit, use the method `CurveModel.predict`.
The `predict` function needs to know which values of the independent variable you want
to predict for, which group you want to predict for, and optionally, which space
you want to predict in. For example, you might want to predict in `ln_gaussian_cdf` space but
make predictions in `ln_gaussian_pdf` space. This is only possible for functions that are
related to one another (see the [functions](#functions) section).

Continuing with [our example](#setting-up-a-model), the following call would
make predictions at all of the times in the original data frame, for group `"all"`.

```python
y_pred = model.predict(
    t=df.time,
    group_name=df.group.unique()
)
```

## Model Pipelines
**`curvefit.pipelines`**

To customize the modeling process for a specific problem, and integrate the core model with predictive
validity and uncertainty, there is a class `curvefit.pipelines._pipeline.ModelPipeline` that sets up the structure.
Each file in `curvefit.piplelines` subclasses this `ModelPipeline` to have different types of modeling processes.

A `ModelPipeline` needs to get much of the same information that is passed to `CurveModel`. The additional
arguments that it needs are

- `predict_space (callable)`: a `curvefit.core.functions` function that matches what space
    you want to do predictive validity in
- `all_cov_names (list{str})`: a list of all the covariate names that will be used
- `obs_se_func (callable)`: in place of `col_obs_se` we now need to define a function that
    produces the standard error as a function of the independent variable

The overall `run()` method that will be used in `ModelPipeline` does the following things:

- `ModelPipeline.run_init_model()`: runs aspects of the model that will not be re-run during
    predictive validity and/or stores information for use later
- `ModelPipeline.run_predictive_validity()`: runs predictive validity, described [here](#predictive-validity)
- `ModelPipeline.fit_residuals()`: fits residuals from predictive validity
- `ModelPipeline.create_draws()`: creates random realizations of the mean function that for the uncertainty intervals
    
Each subclass of `ModelPipeline` has different requirements, each of which are described in their
respective docstrings. Available classes and a brief description of what they do are below:

- `BasicModel`: Runs one model jointly with all groups.
- `BasicModelWithInit`: Runs all models separately to do a [smart initialization](#optimization) of the
    fixed and random effects, and then runs a joint model with all groups.
- `TightLooseModel`: Runs four models with different combinations of settings (one setting should be "tight",
    meaning that it follows the prior closely and one "loose" meaning that it follows the prior less closely)
    and covariate models (can place the covariates on different parameters across models -- by default one is called
    the "beta" model and one is called the "p" model referring to which parameter has covariates). The "tight"
    and "loose" model predictions are blended within each covariate covariate model by using a convex combination
    of the predictions over time. Then the two covariate models are averaged together with pre-specified weights.
- `APModel`: Runs group-specific models and introduces a functional prior on the log of the alpha and beta parameters
    for the `erf` family of functions.
- `PreConditionedAPModel`: Runs like an `APModel` with the `erf` family but dynamically adjusts the bounds
    for the fixed effects of group-specific models based on preconditioning that flags groups that 
    still have an exponential rise in the dependent variable with respect to the independent variable.


## Predictive Validity
**`curvefit.pv`**

