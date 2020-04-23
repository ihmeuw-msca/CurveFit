# -*- coding: utf-8 -*-
# ---------------------------------------------------------------------------
# Parameteric Functions of Time
# ---------------------------------------------------------------------------
'''{begin_markdown param_time_fun}
{spell_markdown
    params
    expit
    gaussian_cdf
    gaussian_pdf
    dgaussian_pdf
    param
}
# Predefined Parametric Functions of Time

## head Syntax
`result = curvefit.core.functions.fun(t, params)`

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

### ln_expit
This is the log of the generalized logistic function which is defined by
\[
    \mbox{ln_expit} ( t , \alpha , \beta , p ) =
        \log \circ \; \mbox{expit} ( t , \alpha , \beta , p )
\]

### gaussian_cdf
This is the generalized Gaussian cumulative distribution function which is defined by
\[
    \mbox{gaussian_cdf} ( t , \alpha , \beta , p ) = \frac{p}{2} \left[
        1.0 + \frac{2}{\pi} \int_0^{\alpha(t-\beta)}
            \exp ( - \tau^2 ) d \tau
    \right]
\]

### ln_gaussian_cdf
This is the log of the
generalized Gaussian cumulative distribution function which is defined by
\[
    \mbox{ln_gaussian_cdf} ( t , \alpha , \beta , p ) =
        \log \circ \; \mbox{gaussian_cdf} ( t , \alpha , \beta , p )
\]

### gaussian_pdf
This is the derivative of the
generalized Gaussian cumulative distribution function which is defined by
\[
    \mbox{gaussian_pdf} ( t , \alpha , \beta , p ) =
        \partial_t \; \mbox{gaussian_cdf} ( t , \alpha , \beta , p )
\]

### ln_gaussian_pdf
This is the log of the derivative of the
generalized Gaussian cumulative distribution function which is defined by
\[
    \mbox{ln_gaussian_cdf} ( t , \alpha , \beta , p ) =
        \log \circ \; \mbox{gaussian_pdf} ( t , \alpha , \beta , p )
\]

### dgaussian_pdf
This is the second derivative of the
generalized Gaussian cumulative distribution function which is defined by
\[
    \mbox{dgaussian_pdf} ( t , \alpha , \beta , p ) =
        \partial_t \; \mbox{gaussian_pdf} ( t , \alpha , \beta , p )
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

## Example
[param_time_fun_xam](param_time_fun_xam.md)

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
# -------------------------------------------------------------------------
# Loss Functions
# --------------------------------------------------------------------------
# Student's T loss function
def st_loss(r, nu=1.0):
    '''{begin_markdown st_loss}
    {spell_markdown }

    # Student's t Loss Function

    ## Syntax
    `loss = curvefit.core.loss_fun.st_loss(r, nu = 1.0)`

    ## t
    is a numpy vector of residual values. We use \( n \)
    for the length of the vector.
    The elements of this vector can be `float` or `a_double` values.

    ## nu
    is the number of degrees of freedom in the t distribution \( \nu \).
    This can be a `float` or `a_double` value.


    ## Distribution
    The student's t-distribution is
    \[
        f(r) = ( 1 + r^2 / \nu )^{- (\nu + 1) / 2 }
             \Gamma[ ( \nu + 1) / 2 ] / [ \sqrt{ \nu \pi } \Gamma( \nu / 2 ) ]
    \]
    where \( \nu \) is the number of degrees of freedom and
    \( \Gamma \) is the gamma function.

    ## Negative log
    Taking the negative log of the distribution function we get
    \[
        - \log [ f(r) ] = \log ( 1 + r^2 / \nu ) (\nu + 1) / 2  + c
    \]
    where \( c \) is constant w.r.t. \( r \).

    ## loss
    The return value `loss` is a scalar equal to
    \[
        \frac{\nu + 1}{2} \sum_{i=0}^{n-1} \log( 1 + r_i^2 / \nu )
    \]

    ## Example
    [loss_xam](loss_xam.md)

    {end_markdown st_loss}
    '''
    return np.sum( np.log(1.0 + r * r / nu) )

# Gaussian loss function
def normal_loss(r):
    '''{begin_markdown normal_loss}
    {spell_markdown }

    # Gaussian Loss Function

    ## Syntax
    `loss = curvefit.core.loss_fun.normal_loss(r)`

    ## r
    is a numpy vector of normalized residual values. We use \( n \)
    for the length of the vector.
    The elements of this vector can be `float` or `a_double` values.


    ## Distribution
    The Gaussian distribution is
    \[
        f(x) = \exp \left[ - (1/2) ( x - \mu )^2 / \sigma^2 \right] /
            \left( \sigma \sqrt{ 2 \pi } \right)
    \]
    where \( \mu \) is the mean and \( \sigma \) is the standard deviation.

    ## Negative log
    Taking the negative log of the distribution function we get
    \[
        - \log [ f(x) ] = (1/2) ( x - \mu )^2 / \sigma^2 + c
    \]
    where \( c \) is constant w.r.t. \( x \).

    ## loss
    The return value `loss` is a scalar equal to
    \[
        \frac{1}{2} \sum_{i=1}^{n-1} r_i^2
    \]
    where \( r_i = ( x_i - \mu)) / \sigma \).

    ## Example
    [loss_xam](loss_xam.md)

    {end_markdown normal_loss}
    '''
    return 0.5 * np.sum(r * r)
