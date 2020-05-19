import sys
import numpy
import curvefit.core.utils
import curvefit.core.effects2params
import curvefit.core.objective_fun
from test_ad import a_functions
from test_ad import a_objective_fun
import cppad_py
#
def identity_fun (x):
    return x
#
def gaussian_loss(x) :
    return numpy.sum( x * x ) / 2.0
#
def test_objective() :
    # -----------------------------------------------------------------------
    # Test parameters
    num_param    = 3
    num_group    = 2
    # -----------------------------------------------------------------------
    # arguments to objective_fun
    #
    num_fe          = num_param
    num_re          = num_group * num_fe
    fe              = numpy.arange(num_fe, dtype = float ) / num_fe
    re              = numpy.arange(num_re, dtype = float ) / num_re
    group_sizes     = (numpy.arange(num_group) + 1 ) * 2
    #
    x               = numpy.concatenate( (fe, re) )
    num_obs         = sum( group_sizes )
    t               = numpy.arange(num_obs, dtype = float )
    obs             = numpy.array( range(num_obs), dtype = float) / num_obs
    obs_se          = (obs  + 1.0 )/ 10.0
    # covs
    covs            = list()
    for k in range(num_param) :
        covs.append( numpy.ones( (num_obs, 1), dtype = float ) )
    #
    model_fun       = curvefit.core.functions.gaussian_cdf
    loss_fun        = gaussian_loss
    link_fun        = [ numpy.exp, identity_fun, numpy.exp ]
    var_link_fun    = num_param * [ identity_fun ]
    # fe_gprior
    fe_gprior       = numpy.empty( (num_fe,2), dtype=float )
    for j in range(num_fe) :
        fe_gprior[j,0] = j / (2.0 * num_fe)
    fe_gprior[:,1] = 1.0 + fe_gprior[:,0] * 1.2
    #
    # re_gprior, param_gprior
    re_gprior         = numpy.empty( (num_fe, num_group, 2), dtype=float )
    param_gprior_mean = numpy.empty( (num_param, num_group), dtype = float )
    for i in range(num_group) :
        for j in range(num_fe) :
            # the matrix re_gprior[:,:,0] is the transposed from the order in re
            re_gprior[j, i ,0]     = (i + j) / (2.0 * (num_fe + num_re))
            k                      = j
            param_gprior_mean[k,i] = (i + k) / (3.0 * (num_fe + num_re))
    re_gprior[:, :, 1] = (1.0 + re_gprior[:, :, 0] / 3.0 )
    param_gprior_std   = (1.0 + param_gprior_mean / 2.0 )
    param_gprior_fun   = identity_fun
    param_gprior = [ param_gprior_fun, param_gprior_mean, param_gprior_std ]
    re_zero_sum_std = num_fe * [ numpy.inf ]
    # -----------------------------------------------------------------------
    # call float objective_fun
    obj_val = curvefit.core.objective_fun.objective_fun(
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
    # call a_double objective_fun
    ax                = cppad_py.independent(x)
    a_model_fun       = a_functions.a_gaussian_cdf
    a_loss_fun        = gaussian_loss
    a_link_fun        = [ a_functions.a_exp, identity_fun, a_functions.a_exp ]
    a_var_link_fun    = var_link_fun
    a_param_gprior    = param_gprior
    aobj_val = a_objective_fun.a_objective_fun(
            ax,
            t,
            obs,
            obs_se,
            covs,
            group_sizes,
            a_model_fun,
            a_loss_fun,
            a_link_fun,
            a_var_link_fun,
            fe_gprior,
            re_gprior,
            a_param_gprior,
            re_zero_sum_std,
    )
    # f(x) = obj_val
    ay    = numpy.empty(1, dtype = cppad_py.a_double)
    ay[0] = aobj_val
    f     = cppad_py.d_fun(ax, ay)
    # -----------------------------------------------------------------------
    # compare function values
    y         = f.forward(0, x)
    rel_error = y[0] / obj_val - 1.0
    eps99     = 99.0 * numpy.finfo(float).eps
    assert abs(rel_error) < eps99
    # -----------------------------------------------------------------------
    # compute derivative
    yq      = numpy.empty((1,1), dtype = float)
    yq[0,0] = 1.0
    xq      = f.reverse(1, yq)
    #
    def objective(x) :
        obj_val = curvefit.core.objective_fun.objective_fun(
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
        return obj_val
    #
    finfo = numpy.finfo(float)
    step  = finfo.tiny / finfo.eps
    x_c = x + 0j
    for j in range(x.size):
        x_c[j]    += step * 1j
        check      = objective(x_c).imag / step
        x_c[j]    -= step * 1j
        rel_error  =  xq[j, 0] / check - 1.0
        assert abs(rel_error) < eps99
