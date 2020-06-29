import numpy
from test_ad import a_functions
from cppad_py import a_double
from test_ad.a_effects2params import a_effects2params

def identity_fun(x) :
    return x

def test_params() :
    # -----------------------------------------------------------------------
    # Test parameters
    num_param    = 3
    num_group    = 2
    # -----------------------------------------------------------------------
    # call effects2params
    num_fe          = num_param
    num_x           = (num_group + 1) * num_fe
    x               = numpy.array( range(num_x), dtype = float ) / num_x
    ax              = a_functions.array2a_double(x)
    group_sizes     = numpy.arange(num_group) * 2 + 1
    num_obs         = sum( group_sizes )
    covs            = list()
    for k in range(num_param) :
        covs.append( numpy.ones( (num_obs, 1), dtype = float ) )
    a_exp           = a_functions.a_exp
    a_link_fun      = [ a_exp, identity_fun, a_exp ]
    a_var_link_fun  = num_param * [ identity_fun ]
    expand          = False
    aparam          = a_effects2params(
        ax, group_sizes, covs, a_link_fun, a_var_link_fun, expand
    )
    # ----------------------------------------------------------------------
    # check result
    eps99  = 99.0  * numpy.finfo(float).eps
    afe    = ax[0 : num_fe]
    are    = ax[num_fe :].reshape( (num_group, num_fe), order='C')
    asum   = afe + are
    avar   = numpy.empty( (num_group, num_fe), dtype=a_double)
    for j in range(num_fe) :
        avar[:,j] = a_var_link_fun[j]( asum[:,j] )
    acheck = numpy.empty( (num_param, num_group), dtype=a_double )
    for k in range(num_param) :
        acheck[k,:] = a_link_fun[k]( avar[:,k] * covs[k][0] )
    #
    rel_error = aparam / acheck - a_double(1.0)
    for k in range(num_param) :
        for i in range(num_group) :
            assert rel_error[k,i].value() < eps99
