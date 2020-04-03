#! /bin/python3
# vim: set expandtab:
# -------------------------------------------------------------------------
n_data  = 21
n_param = 3
alpha   = 2.0
beta    = 3.0
p       = 4.0
# -------------------------------------------------------------------------
import pandas
import numpy
import curvefit
#
# unzip(x) :
#   fe = x[:num_fe]
#   re = x[num_fe:].reshape(num_groups, num_fe)
#
# compute_params(self, x, expand=True) :
#   var = fe + re
#
def generalized_logistic(t, params) :
    alpha = params[0]
    beta  = params[1]
    p     = params[2]
    return p / ( 1.0 + numpy.exp( - alpha * ( t - beta ) ) )
def identity_fun(x) :
    return x
def exp_fun(x) :
    return numpy.exp(x)

#
# data_frame
independent_var   = numpy.array(range(n_data)) * beta / (n_data-1)
params            = [ alpha, beta, p ]
measurement_value = generalized_logistic(independent_var, params)
measurement_std   = n_data * [ 0.1 ]
constant_one      = n_data * [ 1.0 ]
data_group        = n_data * [ 'world' ]
data_dict         = {
    'independent_var'   : independent_var   ,
    'measurement_value' : measurement_value ,
    'measurement_std'   : measurement_std   ,
    'constant_one'      : constant_one      ,
    'data_group'        : data_group        ,
}
data_frame        = pandas.DataFrame(data_dict)
#
# curve_model
col_t        = 'independent_var'
col_obs      = 'measurement_value'
col_covs     = n_param *[ [ 'constant_one' ] ]
col_group    = 'data_group'
param_names  = [ 'alpha', 'beta',       'p'     ]
link_fun     = [ exp_fun, identity_fun, exp_fun ]
var_link_fun = link_fun
fun          = generalized_logistic
col_obs_se   = 'measurement_std'
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
    col_obs_se
)
#
# fit_params
fe_init     = numpy.array( [ alpha, beta, p] ) / 2.0
curve_model.fit_params(fe_init)
print(curve_model.params)
