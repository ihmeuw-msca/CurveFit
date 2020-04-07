# -*- coding: utf-8 -*-
"""
    Test model module.
"""
import numpy as np
import pandas as pd
import pytest
from curvefit import CurveModel
from curvefit.core.functions import log_erf
from curvefit.core.functions import normal_loss, st_loss


@pytest.fixture
def test_data(seed=123):
    np.random.seed(seed)
    df = pd.DataFrame({
        't': np.arange(5),
        'obs': np.random.rand(5),
        'group': 'All',
        'intercept': np.ones(5),
    })
    return df


@pytest.mark.parametrize('param_names', [['alpha', 'beta', 'p']])
@pytest.mark.parametrize('fun', [log_erf])
@pytest.mark.parametrize('link_fun', [[np.exp, lambda x: x, np.exp]])
@pytest.mark.parametrize('var_link_fun', [[lambda x: x]*3])
@pytest.mark.parametrize('loss_fun', [normal_loss, st_loss])
def test_loss_fun(test_data, param_names,
                  fun, link_fun, var_link_fun, loss_fun):
    model = CurveModel(test_data, 't', 'obs',
                       [['intercept']]*3,
                       'group',
                       param_names,
                       link_fun,
                       var_link_fun,
                       fun,
                       loss_fun=loss_fun)

    x = np.hstack((np.ones(3), np.zeros(3)))
    params = model.compute_params(x, expand=False)[:, 0]
    residual = model.obs - fun(model.t, params)

    val = model.objective(x)
    my_val = loss_fun(residual)
    assert np.abs(val - my_val) < 1e-10


