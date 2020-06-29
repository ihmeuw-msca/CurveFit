import numpy as np
import pandas as pd

from curvefit.core.parameter import Variable, Parameter, ParameterSet
from curvefit.core.data import DataSpecs


CHARS = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')


def simulate_params(n_groups):
    var1 = Variable('constant_one', lambda x: x, 0.0, 0.0, fe_bounds=[-np.inf, 0.0])
    var2 = Variable('constant_one', np.exp, 0.0, 0.0, fe_bounds=[-np.inf, 0.0])

    if n_groups == 1:
        var1.re_bounds = [0.0, 0.0]
        var2.re_bounds = [0.0, 0.0]
    else:
        var1.re_bounds = [-1.0, 1.0]
        var2.re_bounds = [-1.0, 1.0]

    param1 = Parameter('p1', np.exp, [var1])
    param2 = Parameter('p2', lambda x: x, [var1, var2])
    param3 = Parameter('p3', np.exp, [var1])

    n_var = 4
    fe_true = -np.random.rand(n_var) * 3
    if n_groups == 1:
        re_true = np.zeros((n_groups, n_var))
    else:
        re_true = np.random.randn(n_groups, n_var)

    x_true = re_true + fe_true
    param_true = np.zeros((n_groups, 3))
    param_true[:, 0] = np.exp(x_true[:, 0])
    param_true[:, 1] = x_true[:, 1] + np.exp(x_true[:, 2])
    param_true[:, 2] = np.exp(x_true[:, 3])

    return ParameterSet([param1, param2, param3]), param_true, x_true


def simulate_data(curve_fun, params_true):
    n_groups = params_true.shape[0]

    n_data = np.random.randint(low=10, high=30, size=n_groups)
    n_data_total = sum(n_data)
    y = np.zeros(n_data_total)
    t = np.zeros(n_data_total)
    start = 0
    for i in range(n_groups):
        t[start: start + n_data[i]] = np.arange(n_data[i])
        y[start: start + n_data[i]] = curve_fun(np.arange(n_data[i]), params_true[i, :])
        start += n_data[i]

    group_names = np.random.choice(CHARS, size=n_groups, replace=False)
    group_col = []
    for i in range(n_groups):
        group_col.extend([group_names[i]] * n_data[i])

    df = pd.DataFrame({
        't': t,
        'obs': y,
        'obs_se': np.ones(n_data_total) * 1e-3,
        'constant_one': np.ones(n_data_total),
        'group': group_col,
    })

    data_specs = DataSpecs('t', 'obs', ['constant_one'], 'group', curve_fun, 'obs_se')

    return df, data_specs
