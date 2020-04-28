import pytest
import numpy as np

from curvefit.initializer.initializer_component import PriorInitializerComponent
from curvefit.initializer.initializer_component import JointPriorInitializerComponent
from curvefit.initializer.initializer_component import IndividualPriorInitializerComponent
from curvefit.initializer.initializer_component import BetaPrior
from curvefit.initializer.initializer_component import LnAlphaBetaPrior
from curvefit.core.parameter import Variable, ParameterSet, Parameter, ParameterFunction
from curvefit.solvers.solvers import Solver
from curvefit.models.core_model import CoreModel
from curvefit.models.base import DataInputs


@pytest.fixture
def parameter_set():
    var1 = Variable(
        covariate='intercept',
        var_link_fun=lambda x: x,
        fe_init=0., re_init=0.
    )
    var2 = Variable(
        covariate='intercept',
        var_link_fun=lambda x: x,
        fe_init=0., re_init=0.
    )
    alpha = Parameter(
        param_name='alpha',
        link_fun=lambda x: x,
        variables=[var1]
    )
    beta = Parameter(
        param_name='beta',
        link_fun=lambda x: x,
        variables=[var2]
    )
    ln_alpha_beta = ParameterFunction(
        param_function_name='ln-alpha-beta',
        param_function=lambda params: np.log(params[0] * params[1])
    )
    param_set = ParameterSet(
        parameters=[alpha, beta],
        parameter_functions=[ln_alpha_beta]
    )
    return param_set


@pytest.fixture
def fake_beta_solver(parameter_set):
    solver = Solver()
    # Fake arguments
    model = CoreModel(param_set=parameter_set, curve_fun=lambda x: x, loss_fun=lambda x: x)
    model.data_inputs = DataInputs(t=np.array([0., 1., 2.]), obs=np.array([0., 1., 2.]),
                                   obs_se=np.array([0., 1., 2.]), group_sizes=[3],
                                   covariates_matrices=[
                                       np.array([[1.], [1.], [1.]]),
                                       np.array([[1.], [1.], [1.]])
                                   ])
    model.data_inputs.var_link_fun = [lambda x: x, lambda x: x]
    model.data_inputs.num_groups = 3
    solver.set_model_instance(model)
    return solver


@pytest.fixture
def fake_alpha_beta_solver(parameter_set):
    solver = Solver()
    # Fake arguments
    model = CoreModel(param_set=parameter_set, curve_fun=lambda x: x, loss_fun=lambda x: x)
    model.data_inputs = DataInputs(t=np.array([0., 1., 2.]), obs=np.array([0., 1., 2.]),
                                   obs_se=np.array([0., 1., 2.]), group_sizes=[1],
                                   covariates_matrices=[
                                       np.array([[1.]]),
                                       np.array([[1.]])
                                   ])
    model.data_inputs.var_link_fun = [lambda x: x, lambda x: x]
    model.data_inputs.num_groups = 1
    solver.set_model_instance(model)
    return solver


def test_prior_initializer_component():
    p = PriorInitializerComponent()
    assert p.component_type is None


def test_joint_prior_initializer_component():
    p = JointPriorInitializerComponent()
    assert p.component_type == 'joint'


def test_individual_prior_initializer_component():
    p = IndividualPriorInitializerComponent()
    assert p.component_type == 'individual'


def test_beta_prior(fake_beta_solver, parameter_set):
    solver = fake_beta_solver.clone()
    p = BetaPrior()
    assert p.component_type == 'joint'

    solver.x_opt = np.array([5., 2., -4., -1., 0., 4., 1., 0.])
    fake_prior = p._extract_prior(solver=solver)
    assert fake_prior == [2., np.std([-1., 4., 0.])]
    fake_new_params = p._update_parameter_set(
        parameter_set_prototype=parameter_set,
        solver=solver
    )
    assert fake_new_params.fe_gprior[1] == fake_prior


def test_ln_alpha_beta_prior(fake_alpha_beta_solver, parameter_set):
    p = LnAlphaBetaPrior()
    assert p.component_type == 'individual'

    solvers = [fake_alpha_beta_solver.clone() for i in range(10)]
    opt_alphas = np.abs(np.random.randn(10))
    opt_betas = np.abs(np.random.randn(10))

    for i, sol in enumerate(solvers):
        sol.x_opt = np.array([opt_alphas[i], opt_betas[i], 0., 0.])

    fake_prior = p._extract_prior(solver=solvers)

    prior = np.log(opt_alphas * opt_betas)
    prior_mean = prior.mean()
    prior_std = prior.std()

    assert fake_prior == [prior_mean, prior_std]

    fake_new_params = p._update_parameter_set(
        parameter_set_prototype=parameter_set,
        solver=solvers
    )
    assert fake_new_params.param_function_fe_gprior[0] == fake_prior
