import numpy
import curvefit.core.utils
import curvefit.core.effects2params
import curvefit.core.objective_fun
from test_ad import a_functions
from cppad_py import a_double

def a_effects2params(
    ax, group_sizes, covs, a_link_fun, a_var_link_fun, expand=True
) :
    num_obs    = numpy.sum(group_sizes)
    num_groups = len(group_sizes)
    num_params = len(covs)
    group_idx  = numpy.cumsum(group_sizes) - 1
    fe_sizes   = numpy.array([ covs[k].shape[1] for k in range(num_params) ])
    num_fe     = fe_sizes.sum()
    num_re     = num_groups * num_fe
    fe_idx     = curvefit.core.utils.sizes_to_indices(fe_sizes)
    #
    # asserts
    for k in range(num_params) :
        assert covs[k].shape[0] == num_obs
    assert len(a_link_fun) == num_params
    #
    #
    # unpack fe and re from x
    afe, are   = curvefit.core.effects2params.unzip_x(ax, num_groups, num_fe)
    if expand :
        # expand random effects
        are = numpy.repeat(are, group_sizes, axis=0)
    else :
        # subsample covariates
        covs = [ covs[k][group_idx,:] for k in range(num_params) ]
    #
    # var  = var_link_fun( fe + re )
    avar = afe + are
    for i in range(num_fe) :
        avar[:, i] = a_var_link_fun[i]( avar[:, i] )
    #
    # params[k][j] = link_fun[k] ( sum_{i(k)} covs[i, j] * var[j, i] )
    shape  = (num_params, num_obs) if expand else (num_params, num_groups)
    aparams = numpy.empty( shape, dtype = a_double )
    for k in range(num_params) :
        # covariate times variable for i-th parameter
        acovs_k    = a_functions.array2a_double( covs[k] )
        aprod      = acovs_k * avar[:, fe_idx[k]]
        # sum of produces for i-th parameter
        aparams[k] = numpy.sum(aprod, axis=1)
        # transform the sum
        aparams[k] = a_link_fun[k]( aparams[k] )
    #
    return aparams
