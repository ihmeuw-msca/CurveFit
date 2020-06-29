import numpy
import curvefit.core.utils
import curvefit.core.effects2params
import curvefit.core.objective_fun
from test_ad import a_effects2params
from test_ad import a_functions

def a_objective_fun(
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
    ) :
    num_groups = len(group_sizes)
    num_fe     = len(fe_gprior)
    afe, are   = curvefit.core.effects2params.unzip_x(ax, num_groups, num_fe)
    #
    # params
    aparams = a_effects2params.a_effects2params(
        ax,
        group_sizes,
        covs,
        a_link_fun,
        a_var_link_fun,
        expand=True
    )
    # residual
    aobs = a_functions.array2a_double( obs )
    at   = a_functions.array2a_double( t )
    aresidual = (aobs - a_model_fun(at, aparams))/obs_se
    #
    # loss
    a_obj_val = a_loss_fun(aresidual)
    #
    # fe_gprior
    aresidual = afe - fe_gprior.T[0]
    a_obj_val += numpy.sum(
        aresidual * aresidual / ( 2.0 * fe_gprior.T[1]**2 )
    )
    #
    # re_gprior
    aresidual = are - re_gprior.T[0]
    a_obj_val += numpy.sum(
        aresidual * aresidual / ( 2.0 * re_gprior.T[1]**2 )
    )
    #
    # zero_sum_std
    for j in range(num_fe) :
        ares_j    = numpy.sum( are[:,j] ) / re_zero_sum_std[j];
        a_obj_val += 0.5 * ares_j * ares_j
    #
    # param_gprior
    if a_param_gprior is not None:
        aparams = a_effects2params.a_effects2params(
            ax,
            group_sizes,
            covs,
            a_link_fun,
            a_var_link_fun,
            expand=False
        )
        aresidual = a_param_gprior[0](aparams) - a_param_gprior[1][0]
        a_obj_val += numpy.sum(
            aresidual * aresidual / ( 2.0 * a_param_gprior[1][1]**2 )
        )
    return a_obj_val
