import pytest
import pandas as pd
import numpy as np
from curvefit.core.model import CurveModel


# model for the mean of the data
def generalized_logistic(t, params):
    alpha = params[0]
    beta = params[1]
    p = params[2]
    return p / (1.0 + np.exp(- alpha * (t - beta)))


# link function used for beta
def identity_fun(x):
    return x


# link function used for alpha, p
def exp_fun(x):
    return np.exp(x)


# inverse of function used for alpha, p
def log_fun(x):
    return np.log(x)


@pytest.mark.parametrize("alpha_true, beta_true, p_true, n_data", [
    (2.0, 3.0, 4.0, 20),
    (2.0, 3.0, 4.0, 10),
    (1.0, 3.0, 4.0, 10),
    (1.0, 3.0, 4.0, 20),
    (1.0, 3.0, 5.0, 20),
    (1.0, 3.0, 5.0, 10),
    (1.0, 3.0, 10.0, 20),
    (1.0, 3.0, 10.0, 10),
])
def test_curve_model(alpha_true, beta_true, p_true, n_data):
    num_params = 3
    params_true = np.array([alpha_true, beta_true, p_true])

    independent_var = np.array(range(n_data)) * beta_true / (n_data - 1)
    df = pd.DataFrame({
            'independent_var': independent_var,
            'measurement_value': generalized_logistic(independent_var, params_true),
            'measurement_std': n_data * [0.1],
            'constant_one': n_data * [1.0],
            'data_group': n_data * ['world'],
        })

    # Initialize a model
    cm = CurveModel(
        df=df,
        col_t='independent_var',
        col_obs='measurement_value',
        col_covs=num_params*[['constant_one']],
        col_group='data_group',
        param_names=['alpha', 'beta', 'p'],
        link_fun=[exp_fun, identity_fun, exp_fun],
        var_link_fun=[exp_fun, identity_fun, exp_fun],
        fun=generalized_logistic,
        col_obs_se='measurement_std'
    )
    inv_link_fun = [log_fun, identity_fun, log_fun]
    fe_init = np.zeros(num_params)
    for j in range(num_params):
        fe_init[j] = inv_link_fun[j](params_true[j] / 3.0)

    # Fit the parameters
    cm.fit_params(
        fe_init=fe_init,
        options={
            'ftol': 1e-16,
            'gtol': 1e-16,
            'maxiter': 1000
        }
    )
    params_estimate = cm.params

    for i in range(num_params):
        rel_error = params_estimate[i] / params_true[i] - 1.0
        assert abs(rel_error) < 1e-6