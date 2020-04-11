# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# model functions:
# ---------------------------------------------------------------------------
'''{begin_markdown param_time_fun}
{spell_markdown
    params
    expit
    erf
    derf
    dderf
}
# Predefined Parametric Functions of Time

@# head Syntax
`result = curvefit.fun(t, params)`

## t
This is a `list` or one dimensional `numpy.array`.

## params
This is either a `list`, or `numpy.array` with one or two dimensions.
In any case, `len(params) == 3`.
If `params` is a two dimensional array, `params.shape[1] == len(t)`.
We use the notation below for the values in `params`:

Notation | Definition
--- | ---
\( \alpha \) | `params[0]`
\( \beta \) | `params[1]`
\( p \) | `params[2]`

## fun
The possible values for *fun* are listed in the subheadings below:

### expit
This is the generalized logistic function which is defined by
\[
    \mbox{expit} ( t , \alpha , \beta , p ) =
    \frac{p}{ 1.0 + \exp [ - \alpha ( t - \beta ) ] }
\]

### log_expit
This is the log of the generalized logistic function which is defined by
\[
    \mbox{log_expit} ( t , \alpha , \beta , p ) =
        \log \circ \; \mbox{expit} ( t , \alpha , \beta , p )
\]

### erf
This is the generalized Gaussian error function which is defined by
\[
    \mbox{erf} ( t , \alpha , \beta , p ) = \frac{p}{2} \left[
        1.0 + \frac{2}{\pi} \int_0^{\alpha(t-\beta)}
            \exp ( - \tau^2 ) d \tau
    \right]
\]

### log_erf
This is the log of the
generalized Gaussian error function which is defined by
\[
    \mbox{log_erf} ( t , \alpha , \beta , p ) =
        \log \circ \; \mbox{erf} ( t , \alpha , \beta , p )
\]

### derf
This is the derivative of the
generalized Gaussian error function which is defined by
\[
    \mbox{derf} ( t , \alpha , \beta , p ) =
        \partial_t \; \mbox{erf} ( t , \alpha , \beta , p )
\]

### log_derf
This is the log of the derivative of the
generalized Gaussian error function which is defined by
\[
    \mbox{log_erf} ( t , \alpha , \beta , p ) =
        \log \circ \; \mbox{derf} ( t , \alpha , \beta , p )
\]

### dderf
This is the second derivative of the
generalized Gaussian error function which is defined by
\[
    \mbox{dderf} ( t , \alpha , \beta , p ) =
        \partial_t \; \mbox{derf} ( t , \alpha , \beta , p )
\]


## result
The result is a `list` or one dimensional `numpy.array` with
`len(result) == len(t)`.
If *params* is a `list` or one dimensional array
```python
    result[i] = fun(t[i], alpha, beta, p)
```
If *params* is a two dimensional array
```python
    result[i] = fun(t[i], alpha[i], beta[i], p[i])
```

{end_markdown param_time_fun}'''
# ----------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Other fuctions
# ---------------------------------------------------------------------------


# Student's T loss function
def st_loss(x, nu=1.0):
    return np.sum(np.log(1.0 + x**2/nu))


# Gaussian loss function
def normal_loss(x):
    return 0.5*sum(x**2)
