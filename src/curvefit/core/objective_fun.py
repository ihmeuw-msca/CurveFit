import numpy
from curvefit.core import effects2params

def objective_fun(
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
    ) :
    """
    {begin_markdown objective_fun}
    {spell_markdown covs gprior param params}

    # Curve Fitting Objective Function of Fixed and Random Effects

    ## Syntax
    val = objective_fun(
        x, t, obs, obs_se, covs, group_sizes,
        fun, loss_fun, link_fun, var_link_fun,
        fe_gprior, re_gprior, param_gprior,
    ) :

    ## Notation

    1. *num_obs* = `len(obs)` is the number of observations (measurements)

    2. *num_param* = `len(covs)` is the number of parameters in the model.

    3. *num_fe* = `len(fe_gprior)` is the number of fixed effects.

    4. *num_group~ = `len(group_sizes)` is the number of groups

    5. *params* = `effects2params(x, group_sizes, covs, link_fun, var_link_fun)
    is a *num_param* by *num_obs* matrix containing the parameters
    corresponding to each observation; see [effects2params](effects2params.md).

    6. A vector is either a `list` or a one dimension `numpy.array`.

    ## x
    is a one dimensional numpy array contain a value for the fixed effects
    followed by the random effects. The random effects are divided into
    sub-vectors with length equal to the number of fixed effects.
    The j-th sub-vector corresponds to the j-th group of observations.

    ## t
    is a vector with length *num_obs* containing the value
    of the independent variable corresponding to each observation.

    ## obs
    is a vector with length *num_obs* containing the observations
    (i.e. measurements).

    ## obs_se
    is a vector with length *num_obs* containing the standard deviation
    for the corresponding observation.

    ## covs
    For *k* = 1, ... , *num_param*-1,
    the value `len(covs[k])` is the number of fixed effects
    corresponding to the k-th parameter.
    The vector `covs[k][j]` has length *num_obs* and is the
    j-th covariate vector corresponding to the k-th parameter.

    ## group_sizes
    The observations are divided into groups.
    The first `group_sizes[0]` observations correspond to the first group,
    the next `group_sizes[1]` corresponds to the section group, and so on.
    The the sum of the group sizes is equal to *num_obs*.

    ## model_fun
    This vector valued function vector values maps parameter values,
    [params](effects2params.md) returned by `effects2params`,
    to the model for the corresponding noiseless observations.
    The residual vector has length *num_obs* and is given by
    ```python
        residual = (obs - model_fun(t, params)) / obs_se
    ```

    ## loss_fun
    This scalar value function maps the residual vector to the corresponding
    contribution to the objective function. For example, a Gaussian likelihood
    corresponds to
    ```python
        loss_fun(residual) = 0.5 * sum( residual * residual )
    ```

    ## link_fun, var_link_fun
    are used to compute *params*; see [Notation](#notation)

    ## fe_gprior
    is an *num_fe* by two numpy array. The value `fe_gprior[i][0]`
    is the prior mean for the i-th fixed effect and
    `fe_gprior[i][1]` is its standard deviation.

    ## re_gprior
    is an *num_fe* by *num_groups* by two numpy array, `re_gprior[i,j,0]`
    ( `re_gprior[i,j,1]` ) is the mean (standard deviation) for the
    random effect corresponding to the i-th fixed effect and the j-th group.

    ## param_gprior
    is a list with three elements. The first element is a function
    of the *params* and its result is a numpy array. We use the notation
    ```python
        range_gprior = param_gprior[0](params)
    ```
    The value `param_gprior[1][0]` ( `param_gprior[1][1]` ) is a numpy array
    corresponding to the mean (standard deviation) for *range_gprior*.

    {end_markdown objective_fun}
    """
    num_groups = len(group_sizes)
    num_fe     = len(fe_gprior)
    fe, re     = effects2params.unzip_x(x, num_groups, num_fe)
    #
    # params
    params = effects2params.effects2params(
        x,
        group_sizes,
        covs,
        link_fun,
        var_link_fun
    )
    # residual
    residual = (obs - model_fun(t, params))/obs_se
    #
    # loss
    val = loss_fun(residual)
    # fe_gprior
    val += 0.5*numpy.sum(
        (fe - fe_gprior.T[0])**2/fe_gprior.T[1]**2
    )
    # re_gprior
    val += 0.5*numpy.sum(
        (re - re_gprior.T[0])**2/re_gprior.T[1]**2
    )
    # param_gprior
    if param_gprior is not None:
        params = effects2params.effects2params(
            x,
            group_sizes,
            covs,
            link_fun,
            var_link_fun,
            expand=False
        )
        val += 0.5*numpy.sum(
            (param_gprior[0](params) - param_gprior[1][0])**2/
            param_gprior[1][1]**2
        )
    return val
