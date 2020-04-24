import numpy as np
import pytest

from curvefit.core.parameter import Variable, Parameter, ParameterSet


@pytest.fixture
def variable():
    return Variable(
        covariate='covariate1',
        var_link_fun=lambda x: x,
        fe_init=0.,
        re_init=0.
    )


def test_variable(variable):
    assert variable.fe_init == 0.
    assert variable.re_init == 0.

    assert variable.fe_gprior == [0., np.inf]
    assert variable.re_gprior == [0., np.inf]

    assert variable.fe_bounds == [-np.inf, np.inf]
    assert variable.re_bounds == [-np.inf, np.inf]


def test_variable_gprior():
    variable = Variable(
        var_link_fun=lambda x: x, covariate='covariate1',
        fe_init=0., re_init=0.,
        fe_gprior=[0., 0.001], re_gprior=[0., 0.0001]
    )

    assert variable.fe_gprior == [0., 1e-3]
    assert variable.re_gprior == [0., 1e-4]


def test_variable_bounds():
    variable = Variable(
        var_link_fun=lambda x: x, covariate='covariate1',
        fe_init=0., re_init=0.,
        fe_bounds=[-10., 10.], re_bounds=[-3., 1.]
    )

    assert variable.fe_bounds == [-10., 10.]
    assert variable.re_bounds == [-3., 1.]


def test_parameter():
    var1 = Variable(
        covariate='covariate1', var_link_fun=lambda x: x,
        fe_init=1., re_init=1.
    )
    var2 = Variable(
        covariate='covariate2', var_link_fun=lambda x: x,
        fe_init=1., re_init=1.
    )
    parameter = Parameter(
        param_name='alpha',
        variables=[var1, var2],
        link_fun=lambda x: x,
    )
    assert parameter.num_fe == 2
    assert callable(parameter.link_fun)

    assert len(parameter.fe_init) == 2
    assert len(parameter.re_init) == 2
    assert len(parameter.fe_gprior) == 2
    assert len(parameter.re_gprior) == 2
    assert len(parameter.fe_bounds) == 2
    assert len(parameter.re_bounds) == 2


def test_parameter_set():
    var1 = Variable(
        covariate='covariate1', var_link_fun=lambda x: x,
        fe_init=1., re_init=1.
    )
    var2 = Variable(
        covariate='covariate2', var_link_fun=lambda x: x,
        fe_init=1., re_init=1.
    )
    parameter1 = Parameter(
        param_name='alpha',
        variables=[var1],
        link_fun=lambda x: x,
    )
    parameter2 = Parameter(
        param_name='beta',
        variables=[var2],
        link_fun=lambda x: x,
    )
    param_set = ParameterSet(
        parameters=[parameter1, parameter2],
        parameter_functions=[(lambda params: params[0] * params[1], [0.0, np.inf])],
    )

    assert param_set.num_fe == 2
    assert len(param_set.parameter_functions) == 1
    assert callable(param_set.parameter_functions[0][0])
    assert len(param_set.parameter_functions[0][1]) == 2
    assert param_set.parameter_functions[0][1] == [0., np.inf]

    assert param_set.parameter_functions[0][0]([2, 3]) == 6
