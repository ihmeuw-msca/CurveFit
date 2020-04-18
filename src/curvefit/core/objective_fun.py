import numpy
import curvefit

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
    {spell_markdown covs gprior param params obj}

    # Curve Fitting Objective Function of Fixed and Random Effects

    ## Syntax
    obj_val = curvefit.core.objective_fun.objective_fun(
        x, t, obs, obs_se, covs, group_sizes, model_fun, loss_fun,
        link_fun, var_link_fun, fe_gprior, re_gprior, param_gprior,
    ) :

    ## Notation

    1. *num_obs* = `len(obs)` is the number of observations (measurements)

    2. *num_param* = `len(covs)` is the number of parameters in the model.

    3. *num_fe* = `fe_gprior.shape[0]` is the number of fixed effects.

    4. *num_group* = `len(group_sizes)` is the number of groups

    5. *params* = `effects2params(x, group_sizes, covs, link_fun, var_link_fun)
    is a *num_param* by *num_obs* numpy array containing the parameters
    corresponding to each observation; see [effects2params](effects2params.md).

    6. A vector is either a `list` or a one dimension `numpy.array`.

    ## x
    is a one dimension numpy array contain a value for the fixed effects
    followed by the random effects. The random effects are divided into
    sub-vectors with length equal to the number of fixed effects.
    The i-th sub-vector corresponds to the i-th group of observations.

    ## t
    is a one dimension numpy array with length *num_obs* containing the value
    of the independent variable corresponding to each observation.

    ## obs
    is a one dimension numpy array with length *num_obs* containing the
    observations (i.e. measurements).

    ## obs_se
    is a one dimension numpy array with length *num_obs* containing the
    standard deviation for the corresponding observation.

    ## covs
    Is a `list` with length equal to the number of parameters and `covs[k]`
    is a two dimension numpy array with the following contents:

    -- `covs[k].shape[0]` is the number of observations

    -- `covs[k].shape[1]` is the number of fixed effects corresponding to the
    k-th parameter.

    -- `covs[k][i, ell]` is the covariate value corresponding to the
    i-th observation and ell-th covariate for the k-th parameter.

    ## group_sizes
    The observations are divided into groups.
    The first `group_sizes[0]` observations correspond to the first group,
    the next `group_sizes[1]` corresponds to the section group, and so on.
    The the sum of the group sizes is equal to *num_obs*.

    ## model_fun
    This vector valued function vector values maps parameter values,
    [params](effects2params.md) returned by `effects2params`,
    to the model for the corresponding noiseless observations.
    The observation residual vector has length *num_obs* and is given by
    ```python
        obs_res = (obs - model_fun(t, params)) / obs_se
    ```

    ## loss_fun
    This scalar value function maps the observation residual vector to the
    corresponding contribution to the objective function.
    For example, if *loss_fun* corresponds to a Gaussian likelihood,
    it is equal to
    ```python
        gaussian_loss(obs_res) = 0.5 * sum( obs_res * obs_res )
    ```

    ## link_fun, var_link_fun
    are used to compute *params*; see [Notation](#notation)

    ## fe_gprior
    is an *num_fe* by two numpy array. The value `fe_gprior[j][0]`
    is the prior mean for the j-th fixed effect and
    `fe_gprior[j][1]` is its standard deviation.
    If `fe` is the fixed effect sub-vector of `x`, the prior residual
    for the fixed effects is
    ```python
        fe_res = ( fe.T - fe_gprior[:,0] ) / fe_gprior[:,1]
    ```
    were `fe.T` denotes the transpose of `fe`.

    ## re_gprior
    is an *num_fe* by *num_groups* by by two numpy array, `re_gprior[j,i,0]`
    ( `re_gprior[j,i,1]` ) is the mean (standard deviation) for the
    random effect corresponding to the j-th fixed effect and the i-th group.
    If `re` is the matrix of random effect corresponding to`x`,
    the prior residual for the random effects is
    ```python
        re_res = ( re.T - re_gprior[:,:,0] ) / re_gprior[:,:,1]
    ```

    ## param_gprior
    is a list with three elements. The first element is a function
    of the *params* and its result is a numpy array. We use the notation
    ```python
        range_gprior = param_gprior[0](params)
    ```
    There is a subtlety here, column dimension of the *params* above
    is *num_groups* (not *num_obs).
    The value `param_gprior[1][0]` ( `param_gprior[1][1]` ) is a numpy array
    corresponding to the mean (standard deviation) for *range_gprior*.
    The prior residual for the parameters is
    ```python
        param_res = (range_gprior - param_gprior[1][[0]]) / param_gprior[1][1]
    ```

    ## obj_val
    The return *val* is a `float` equal to the objective function
    ```python
        obj_val = loss_fun(obs_res) + gaussian_loss(fe_res)
            + gaussian_loss(re_res) + gaussian_loss(param_res)
    ```

    ## Example
    [objective_fun_xam](objective_fun_xam.md)

    {end_markdown objective_fun}
    """
    num_groups = len(group_sizes)
    num_fe     = len(fe_gprior)
    fe, re     = curvefit.core.effects2params.unzip_x(x, num_groups, num_fe)
    #
    # params
    params = curvefit.core.effects2params.effects2params(
        x,
        group_sizes,
        covs,
        link_fun,
        var_link_fun,
        expand=True
    )
    # residual
    residual = (obs - model_fun(t, params))/obs_se
    #
    # loss
    obj_val = loss_fun(residual)
    #
    # fe_gprior
    obj_val += 0.5*numpy.sum(
        (fe - fe_gprior.T[0])**2/fe_gprior.T[1]**2
    )
    #
    # re_gprior
    obj_val += 0.5*numpy.sum(
        (re - re_gprior.T[0])**2/re_gprior.T[1]**2
    )
    #
    # param_gprior
    if param_gprior is not None:
        params = curvefit.core.effects2params.effects2params(
            x,
            group_sizes,
            covs,
            link_fun,
            var_link_fun,
            expand=False
        )
        obj_val += 0.5*numpy.sum(
            (param_gprior[0](params) - param_gprior[1][0])**2/
            param_gprior[1][1]**2
        )
    return obj_val
