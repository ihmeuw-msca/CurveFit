# -*- coding: utf-8 -*-
"""
    Functions used for curve fitting.
"""
import numpy as np
from scipy import special


# logistic function
def expit(t, params):
    tmp = params[0]*(t - params[1])
    negidx = tmp < 0.0
    posidx = ~negidx
    result = np.zeros(t.size, dtype=params.dtype)
    if params.ndim == 2:
        result[negidx] = params[2][negidx]*np.exp(tmp[negidx])/ \
                         (1.0 + np.exp(tmp[negidx]))
        result[posidx] = params[2][posidx]/(1.0 + np.exp(-tmp[posidx]))
    else:
        result[negidx] = params[2]*np.exp(tmp[negidx])/ \
                         (1.0 + np.exp(tmp[negidx]))
        result[posidx] = params[2]/(1.0 + np.exp(-tmp[posidx]))
    return result


# log logistic function
def log_expit(t, params):
    tmp = expit(t, params)
    result = np.zeros(t.size, dtype=params.dtype)
    zidx = tmp == 0.0
    oidx = ~zidx
    result[oidx] = np.log(tmp[oidx])
    if params.ndim == 2:
        result[zidx] = np.log(params[2][zidx]) + \
                       params[0][zidx]*(t[zidx] - params[1][zidx])
    else:
        result[zidx] = np.log(params[2]) + params[0]*(t[zidx] - params[1])
    return result


# error function cdf of the normal distribution
def erf(t, params):
    return 0.5*params[2]*(special.erf(params[0]*(t - params[1])) + 1.0)


# log error function
def log_erf(t, params):
    tmp = erf(t, params)
    x = params[0]*(t - params[1])
    result = np.zeros(t.size, dtype=params.dtype)
    zidx = tmp == 0.0
    oidx = ~zidx
    result[oidx] = np.log(tmp[oidx])
    if params.ndim == 2:
        result[zidx] = np.log(params[2][zidx]/2) - x[zidx]**2 - \
                       np.log(-x[zidx]) - 0.5*np.log(np.pi)
    else:
        result[zidx] = np.log(params[2]/2) - x[zidx]**2 - \
                       np.log(-x[zidx]) - 0.5*np.log(np.pi)
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


# second order dervivative of erf function
def dderf(t, params):
    a = params[0]
    b = params[1]
    p = params[2]
    tmp = a*(t - b)
    return -2.0*a**2*p*tmp*np.exp(-tmp**2)/np.sqrt(np.pi)


# Student's T loss function
def st_loss(x, nu=1.0):
    return np.sum(np.log(1.0 + x**2/nu))


# Gaussian loss function
def normal_loss(x):
    return 0.5*sum(x**2)
