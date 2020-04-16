import numpy
import curvefit

def unzip_x(x, num_groups, num_fe) :
    '''{begin_markdown unzip_x}
    {spell_markdown params}

    # Extract Fixed and Random Effects from Single Vector Form

    ## Syntax
    `fe, re = curvefit.core.effects2params.unzip_x(x, num_groups, num_fe)`

    ## num_groups
    is the number of data groups.

    ## num_fe
    is the number of fixed effects.

    ## x
    is a numpy vector with length equal to `(num_groups + 1)*num_fe`

    ## fe
    this return value
    is a numpy vector containing the fist *num_fe* elements of *x*.

    ## re
    this return value
    is a numpy two dimensional array with row dimension *num_groups*
    and column dimension *num_fe*.
    The i-th row of *re* contains the following sub-vector of *x*
    ```python
        re[i,:] = x[(i+1)*num_fe : (i+2)*num_fe]
    ```

    ## Example
    [unzip_x_xam](unzip_x_xam.md)

    {end_markdown unzip_x}'''
    fe = x[: num_fe]
    re = x[num_fe :].reshape(num_groups, num_fe, order='C')
    return fe, re

def effects2params(x, group_sizes, covs, link_fun, var_link_fun, expand=True) :
    '''{begin_markdown effects2params}
    {spell_markdown params covs}

    # Map Vector of Fixed and Random Effects to Parameter Matrix

    ## Syntax
    `params = curvefit.core.effects2params(
        x, group_sizes, covs, link_fun, var_link_fun, expand=True
    )`

    ## group_sizes
    The observations (measurements) are divided into groups.
    The first `group_sizes[0]` observations correspond to the first group,
    the next `group_sizes[1]` corresponds to the section group, and so on.
    The total number of observations is the sum of the group sizes.

    ## covs
    The value `len(covs)` is the number of parameters in the model.
    The value `len(covs[k])` is the number of fixed effects
    corresponding to the k-th parameter.
    The vector `covs[k][j]` is the j-th covariate vector corresponding
    to the k-th parameter.
    The length of `covs[k][j]` is equal to the total number of observations.

    ## link_fun
    The value `len(link_fun)` is equal to the number of parameters and
    `link_fun[k]` is a function with one argument and one result that
    transforms the k-th parameter.

    ## var_link_fun
    The value `len(var_link_fun)` is equal to the number of fixed effects and
    `link_fun[i]` is a function with one argument and one result that
    transforms the i-th fixed effect.
    The first `len(covs[0])` fixed effects correspond to the first parameter,
    the next `len(covs[1])` fixed effects correspond to the second parameter
    and so on.

    ## expand
    If *expand* is `True` (`False`), create parameters for each observation
    (for each group of observations).

    ## x
    This is a one dimensional numpy array contain a value for the fixed effects
    followed by the random effects. The random effects are divided into
    sub-vectors with length equal to the number of fixed effects.
    The j-th sub-vector corresponds to the j-th group of observations.

    ## params
    Let \( f_i \) be the vector of fixed effects and
    \( r_{i,j} \) the matrix of random effects corresponding to *x*.
    We define the vector, with length equal to the number of fixed effects,
    \[
        v_{i,j} = V_i \left( f_i + r_{i,j} \right)
    \]
    where \( V_i \) is the function `var_link_fun[i]`.
    If *expand* is true (false) \( j \) indexes observations (groups).
    (If *expand* is true the random effect for a group gets repeated
    for all the observations in the group.)
    The return value `params` is a numpy array with row dimension
    equal to the number of parameters.
    If *expand* is true (false), its column  column dimension
    equal to the number of observations (number of groups).
    The value `params[k][j]` is
    \[
        P_k \left( \sum_{i(k)} v_i c_{i,j} \right)
    \]
    where \( P_k \) is the function `link_fun[k]`,
    \( i(k) \) is the set of fixed effects indices
    corresponding to the k-th parameter,
    \( c_{i,j} \) is the covariate value corresponding to the
    i-th fixed effect and the j-th observation, if *expand* is true,
    or j-th group, if *expand* is false.

    {end_markdown effects2params}'''
    num_obs    = numpy.sum(group_sizes)
    num_groups = len(group_sizes)
    num_params = len(covs)
    group_idx  = numpy.cumsum(group_sizes) - 1
    fe_sizes   = numpy.array([ covs[i].shape[1] for i in range(num_params) ])
    num_fe     = fe_sizes.sum()
    num_re     = num_groups * num_fe
    fe_idx     = curvefit.core.utils.sizes_to_indices(fe_sizes)
    #
    # asserts
    for k in range(num_params) :
        assert covs[k].shape[0] == num_obs
    assert len(link_fun) == num_params
    #
    #
    # unpack fe and re from x
    fe, re   = unzip_x(x, num_groups, num_fe)
    if expand :
        # expand random effects
        re = numpy.repeat(re, group_sizes, axis=0)
    else :
        # subsample covariates
        covs = [ covs[k][group_idx,:] for k in range(num_params) ]
    #
    # var  = var_link_fun( fe + re )
    var = fe + re
    for i in range(num_fe) :
        var[:, i] = var_link_fun[i]( var[:, i] )
    #
    # params[k][j] = link_fun[k] ( sum_{i(k)} covs[i, j] * var[j, i] )
    shape  = (num_params, num_obs) if expand else (num_params, num_groups)
    params = numpy.empty( shape, dtype = type(x[0]) )
    for k in range(num_params) :
        # covariate times variable for i-th parameter
        prod      = covs[k] * var[:, fe_idx[k]]
        # sum of produces for i-th parameter
        params[k] = numpy.sum(prod, axis=1)
        # transform the sum
        params[k] = link_fun[k]( params[k] )
    #
    return params
