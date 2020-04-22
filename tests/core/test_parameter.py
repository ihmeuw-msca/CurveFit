import numpy as np
import pytest

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


@pytest.fixture
def parameter1():
    return Parameter(
        param_name='alpha', link_fun=lambda x: x,
        var_link_fun=[lambda x: x], covariates=['covariate1'],
        fe_init=[0.], re_init=[0.]
    )


@pytest.fixture
def parameter2():
    return Parameter(
        param_name='beta', link_fun=lambda x: x,
        var_link_fun=[lambda x: x], covariates=['covariate2'],
        fe_init=[0.], re_init=[0.]
    )


def test_parameter_set(parameter1, parameter2):
    param_set = ParameterSet(
        parameter_list=[parameter1, parameter2],
        parameter_functions=[lambda params: params[0] * params[1]],
    )
    assert len(param_set.parameter_list) == 2
    assert param_set.num_params == 2
    assert param_set.num_fe == 2
    assert len(param_set.parameter_functions) == 1
    assert len(param_set.parameter_function_priors) == 1
    assert param_set.parameter_function_priors[0] == [0., np.inf]


def test_parameter_set_no_function(parameter1, parameter2):
    with pytest.raises(RuntimeError):
        ParameterSet(
            parameter_list=[parameter1, parameter2],
            parameter_function_priors=[[0.0, np.inf]]
        )


def test_parameter_set_functional_prior(parameter1, parameter2):
    param_set = ParameterSet(
        parameter_list=[parameter1, parameter2],
        parameter_functions=[lambda params: params[0] * params[1]],
        parameter_function_priors=[[0.1, 1.0]]
    )
    assert param_set.parameter_function_priors[0] == [0.1, 1.0]
