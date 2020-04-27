# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Parameteric Functions of Time
# ---------------------------------------------------------------------------
import numpy as np
from scipy import special
#
from curvefit.core.loss import \
    st_loss,  \
    normal_loss
#
from curvefit.core.param_model import \
    expit, \
    ln_expit,\
    gaussian_cdf,\
    ln_gaussian_cdf,\
    gaussian_pdf,\
    ln_gaussian_pdf,\
    dgaussian_pdf


"""
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
def ln_expit(t, params):
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
def gaussian_cdf(t, params):
    return 0.5*params[2]*(special.erf(params[0]*(t - params[1])) + 1.0)


# log error function
def ln_gaussian_cdf(t, params):
    tmp = gaussian_cdf(t, params)
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


# derivative of gaussian_cdf function
def gaussian_pdf(t, params):
    return params[0]*params[2]*np.exp(
        -(params[0]*(t - params[1]))**2
    )/np.sqrt(np.pi)


# log derivative of gaussian_cdf function
def ln_gaussian_pdf(t, params):
    return np.log(params[0]) + np.log(params[2]) - \
        (params[0]*(t - params[1]))**2 - 0.5*np.log(np.pi)


# second order dervivative of gaussian_cdf function
def dgaussian_pdf(t, params):
    a = params[0]
    b = params[1]
    p = params[2]
    tmp = a*(t - b)
    return -2.0*a**2*p*tmp*np.exp(-tmp**2)/np.sqrt(np.pi)
"""
