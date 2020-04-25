import pytest
import numpy as np
import pandas as pd 

from curvefit.core.data import DataSpecs
from curvefit.core.functions import normal_loss, st_loss, ln_gaussian_cdf, ln_gaussian_pdf, gaussian_cdf, gaussian_pdf
from curvefit.models.core_model import CoreModel
from curvefit.core.parameter import Variable, Parameter, ParameterSet


n_A = 5
n_B = 6
n_C = 7
n_total = n_A + n_B + n_C

@pytest.fixture
def data():
    df = pd.DataFrame({
        't': np.concatenate((np.arange(n_B), np.arange(n_A), np.arange(n_C))),
        'obs': np.random.rand(n_total),
        'group': ['B'] * n_B + ['A'] * n_A + ['C'] * n_C,
        'intercept': np.ones(n_total),
        'se': np.random.randn(n_total) * 1e-3,

    })
    data_specs = DataSpecs('t', 'obs', ['intercept'], 'group', ln_gaussian_cdf, 'se')

    return df, data_specs


@pytest.fixture
def param_set():
    variable1 = Variable('intercept', lambda x:x, 0.0, 0.0, re_bounds=[0.0, 1.0])
    variable2 = Variable('intercept', lambda x:x, 0.0, 0.0, re_bounds=[0.0, 2.0])
    variable3 = Variable('intercept', lambda x:x, 0.0, 0.0, re_bounds=[0.0, 3.0])
    parameter1 = Parameter('p1', np.exp, [variable1])
    parameter2 = Parameter('p2', np.exp, [variable2])
    parameter3 = Parameter('p2', np.exp, [variable3] * 2)
    parameter_set = ParameterSet([parameter1, parameter2, parameter3])
    assert parameter_set.num_fe == 4
    return parameter_set


@pytest.mark.parametrize('curve_fun', [
    ln_gaussian_cdf, 
    ln_gaussian_pdf,
    gaussian_cdf,
    gaussian_pdf,
])
@pytest.mark.parametrize('loss_fun', [normal_loss, st_loss])
def test_core_model_run(data, param_set, curve_fun, loss_fun):
    model = CoreModel(param_set, curve_fun, loss_fun)
    x0 = np.array([0.0] * param_set.num_fe * 4)
    model.objective(x0, data)
    
    covs_mat = model.data_inputs.covariates_matrices
    assert covs_mat[0].shape[1] == 1
    assert covs_mat[1].shape[1] == 1
    assert covs_mat[2].shape[1] == 2

    assert model.data_inputs.group_sizes == [n_B, n_A, n_C]
    assert len(model.data_inputs.var_link_fun) == 4

    assert model.bounds.shape == (4 * (3 + 1), 2)
    ub = [b[1] for b in model.bounds]
    assert ub[:4] == [np.inf] * 4
    assert ub[4:] == [1.0, 2.0, 3.0, 3.0] * 3

    model.gradient(x0, data)
    model.forward(x0, np.arange(10, 16))
