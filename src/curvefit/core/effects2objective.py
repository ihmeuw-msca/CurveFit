import numpy
from curvefit.core import effects2params
def effects2objective(
        x,
        t,
        obs,
        obs_se,
        covs,
        group_sizes,
        fun,
        loss_fun,
        link_fun,
        var_link_fun,
        fe_gprior,
        re_gprior,
        fun_gprior,
    ) :
    """Objective function.

    Args:
        x (numpy.ndarray):
            Model parameters.

    Returns:
        float:
            Objective value.
    """
    num_groups = len(group_sizes)
    num_fe     = len(fe_gprior)
    fe, re     = effects2params.unzip_x(x, num_groups, num_fe)
    #
    params = effects2params.effects2params(
        x,
        group_sizes,
        covs,
        link_fun,
        var_link_fun
    )
    residual = (obs - fun(t, params))/obs_se
    # val = 0.5*numpy.sum(residual**2)
    val = loss_fun(residual)
    # gprior from fixed effects
    val += 0.5*numpy.sum(
        (fe - fe_gprior.T[0])**2/fe_gprior.T[1]**2
    )
    # gprior from random effects
    val += 0.5*numpy.sum(
        (re - re_gprior.T[0])**2/re_gprior.T[1]**2
    )
    # other functional gprior
    if fun_gprior is not None:
        params = effects2params.effects2params(
            x,
            group_sizes,
            covs,
            link_fun,
            var_link_fun,
            expand=False
        )
        val += 0.5*numpy.sum(
            (fun_gprior[0](params) - fun_gprior[1][0])**2/
            fun_gprior[1][1]**2
        )
    return val
