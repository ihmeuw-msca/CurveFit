# -*- coding: utf-8 -*-
"""
    Functions used for curve fitting.
"""
import numpy as np
from scipy import special


# logistic function
def expit(t, params):
    return params[2]*special.expit(params[0]*(t - params[1]))


# log logistic function
def log_expit(t, params):
    return np.log(logistic(t, params))


# error function cdf of the normal distribution
def erf(t, params):
    return 0.5*params[2]*(special.erf(params[0]*(t - params[1])) + 1.0)


# log error function
def log_erf(t, params):
    tmp = erf(t, params)
    result = np.zeros(t.size, dtype=params.dtype)
    zidx = tmp == 0.0
    oidx = ~zidx
    result[oidx] = np.log(tmp[oidx])
    if params.ndim == 2:
        result[zidx] = np.log(params[2][zidx]) - \
                       (params[0][zidx]*(t[zidx] - params[1][zidx]))**2
    else:
        result[zidx] = np.log(params[2]) - (params[0]*(t[zidx] - params[1]))**2
    return result


# derivative of erf function
def derf(t, params):
    return params[0]*params[2]*np.exp(
        -(params[0]*(t - params[1]))**2
    )/np.sqrt(np.pi)


# log derivative of erf function
def log_derf(t, params):
    return np.log(params[0]) + np.log(params[2]) - \
        (params[0]*(t - params[1]))**2 - 0.5*np.log(np.pi)

