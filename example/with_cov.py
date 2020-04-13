#! /bin/python3
# vim: set expandtab:
# -------------------------------------------------------------------------
import sys
import pandas
import numpy
import curvefit

numpy.random.seed(123)

n_days = 200
alpha_true = 2.0
beta_0_true = 1.0
beta_1_true = -1.0
p_true = 4.0
rel_tol = 1e-5

cov1 = numpy.arange(n_days) - n_days / 2
cov2 = numpy.random.randn(n_days) * 2

def generalized_logistic(t, params):
    return params[0] / (1.0 + numpy.exp(-params[1] * (t - params[2])))

def generate_data(t, x):
    return x[0] / (1.0 + numpy.exp(-x[1] * (t - cov1 * x[2] - cov2 * x[3])))

x_true = numpy.array([p_true, alpha_true, beta_0_true, beta_1_true])
days = numpy.array(range(n_days)) / (n_days - 1)
measurements = generate_data(days, x_true)
data_dict = {
    'days': days,
    'measurements': measurements,
    'std': 0.01 * numpy.ones(n_days),
    'constant_one': numpy.ones(n_days),
    'cov1': cov1,
    'cov2': cov2,
    'data_group': n_days * ['world'],
}
data_frame = pandas.DataFrame(data_dict)

col_t        = 'days'
col_obs      = 'measurements'
col_covs     = [['constant_one'], ['constant_one'], ['cov1', 'cov2']]
col_group    = 'data_group'
param_names  = ['p', 'alpha', 'beta']
link_fun     = [lambda x: x ,lambda x: x, lambda x: x]
var_link_fun = [lambda x: x, lambda x: x, lambda x: x, lambda x: x]
fun          = generalized_logistic
col_obs_se   = 'std'
#
curve_model = curvefit.CurveModel(
    data_frame,
    col_t,
    col_obs,
    col_covs,
    col_group,
    param_names,
    link_fun,
    var_link_fun,
    fun,
    col_obs_se,
)
#
# fit_params
fe_init = x_true + numpy.random.randn(4)
curve_model.fit_params(fe_init, re_bounds=[[0.0, 0.0]] * 4)
x_est = curve_model.result.x
params_est = curve_model.compute_params(x_est)
print("fe_init size:", fe_init.shape, "params size:", params_est.shape)
print(x_est)
for true, est in zip(x_true, x_est):
    assert (est - true) / true < rel_tol
print("OK")
sys.exit(0)
