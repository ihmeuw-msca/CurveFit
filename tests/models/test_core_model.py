import pytest
import numpy as np
import pandas as pd 

from curvefit.core.data import DataSpecs
from curvefit.core.functions import normal_loss, st_loss, ln_gaussian_cdf, ln_gaussian_pdf, gaussian_cdf, gaussian_pdf
from curvefit.models.core_model import Model
from curvefit.core.parameter import Variable, Parameter, ParameterSet

@pytest.fixture(scope='module')
def data(seed=123):
    np.random.seed(seed)
    df = pd.DataFrame({
        't': np.arange(5),
        'obs': np.random.rand(5),
        'group': 'All',
        'intercept': np.ones(5),
        'se': np.random.randn(5) * 1e-3,

    })
    data_specs = DataSpecs('t', 'obs', ['intercept'], 'group', ln_gaussian_cdf, 'se')
    return df, data_specs

@pytest.fixture(scope='module')
def param_set():
    variable = Variable('intercept', lambda x:x, 0.0, 0.0)
    parameter = Parameter('p1', np.exp, [variable])
    parameter_set = ParameterSet([parameter] * 3)
    assert parameter_set.num_fe == 3
    return parameter_set

class TestCoreModel:

    @pytest.mark.parametrize('curve_fun', [
        ln_gaussian_cdf, 
        ln_gaussian_pdf,
        gaussian_cdf,
        gaussian_pdf,
    ])
    @pytest.mark.parametrize('loss_fun', [normal_loss, st_loss])
    def test_core_model_sanity(self, data, param_set, curve_fun, loss_fun):
        model = Model(param_set, curve_fun, loss_fun)
        x0 = np.array([0.0] * param_set.num_fe * 2)
        model.objective(x0, data)