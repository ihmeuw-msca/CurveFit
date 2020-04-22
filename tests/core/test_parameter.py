import numpy as np

from curvefit.core.parameter import Parameter
from curvefit.core.parameter import ParameterSet


def test_parameter():
    parameter = Parameter(
        param_name='alpha', link_fun=lambda x: x,
        var_link_fun=[lambda x: x], covariates=['covariate1'],
        fe_init=[0.], re_init=[0.]
    )

    assert parameter.fe_init == [0.]
    assert parameter.re_init == [0.]

    assert len(parameter.fe_gprior) == 1
    assert len(parameter.re_gprior) == 1

    assert parameter.fe_gprior[0] == [0., np.inf]
    assert parameter.re_gprior[0] == [0., np.inf]

    assert len(parameter.fe_bounds) == 1
    assert len(parameter.re_gprior) == 1

    assert parameter.fe_bounds[0] == [-np.inf, np.inf]
    assert parameter.re_bounds[0] == [-np.inf, np.inf]


def test_parameter_gprior():
    parameter = Parameter(
        param_name='alpha', link_fun=lambda x: x,
        var_link_fun=[lambda x: x], covariates=['covariate1'],
        fe_init=[0.], re_init=[0.],
        fe_gprior=[[0., 0.001]], re_gprior=[[0., 0.0001]]
    )

    assert len(parameter.fe_gprior) == 1
    assert len(parameter.re_gprior) == 1

    assert parameter.fe_gprior[0] == [0., 1e-3]
    assert parameter.re_gprior[0] == [0., 1e-4]


def test_parameter_bounds():
    parameter = Parameter(
        param_name='alpha', link_fun=lambda x: x,
        var_link_fun=[lambda x: x], covariates=['covariate1'],
        fe_init=[0.], re_init=[0.],
        fe_bounds=[[-10., 10.]], re_bounds=[[-3., 1.]]
    )

    assert len(parameter.fe_bounds) == 1
    assert len(parameter.re_bounds) == 1

    assert parameter.fe_bounds[0] == [-10., 10.]
    assert parameter.re_bounds[0] == [-3., 1.]


def test_parameter_set():
    pass


def test_parameter_set_no_function():
    pass


def test_parameter_set_functional_prior():
    pass
