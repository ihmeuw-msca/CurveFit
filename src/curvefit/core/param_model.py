import numpy
from cppad_py import a_double
from curvefit.core.utils import unpack_param
from curvefit.core.numpy_ufunc import erf

# ---------------------------------------------------------------------------
# Model Functions
# ---------------------------------------------------------------------------
"""
{begin_markdown param_model}
{spell_markdown
    params
    expit
    gaussian_cdf
    gaussian_pdf
    dgaussian_pdf
    param
}
# Predefined Parametric Model for Functions of Time

## head Syntax
`result = curvefit.core.param_model.fun(t, param)`

## t
This is a numpy vector with elements of type `float`.
We use *n* to denote the length of this vector.

## param
must be a numpy vector or matrix with elements of type `float` or `a_double`.
If it is a vector, its length must be three.
If it is a matrix, its row dimension must be three
and its column dimension must be *n*.

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
The result is a numpy vector with elements of the same type
as *params*.  If *params* is a vector,
```python
    result[i] = fun(t[i], alpha, beta, p)
```
If *params* is a matrix
```python
    result[i] = fun(t[i], alpha[i], beta[i], p[i])
```

## Example
[param_model_xam](param_model_xam.md)

{end_markdown param_model}
"""
def gaussian_cdf(t, param) :
    """
    See param_model in documentation
    """
    alpha, beta, p = unpack_param(t, param)
    z              = alpha * (t - beta)
    return p * ( 1.0 + erf(z) ) / 2.0
#
def expit(t, param) :
    """
    See param_model in documentation
    """
    alpha, beta, p = unpack_param(t, param)
    z              = alpha * (t - beta)
    return p / ( 1.0 + numpy.exp(-z) )
#
def ln_expit(t, param) :
    """
    See param_model in documentation
    """
    alpha, beta, p = unpack_param(t, param)
    return numpy.log( expit(t, param) )
#
def ln_gaussian_cdf(t, param) :
    """
    See param_model in documentation
    """
    alpha, beta, p = unpack_param(t, param)
    return numpy.log( gaussian_cdf(t, param) )
#
def gaussian_pdf(t, param) :
    """
    See param_model in documentation
    """
    alpha, beta, p = unpack_param(t, param)
    z              = alpha * (t - beta)
    return alpha * p * numpy.exp( - z * z ) / numpy.sqrt(numpy.pi)
#
def ln_gaussian_pdf(t, param) :
    """
    See param_model in documentation
    """
    alpha, beta, p = unpack_param(t, param)
    z              = alpha * (t - beta)
    return numpy.log( alpha * p / numpy.sqrt(numpy.pi) ) - z * z
#
def dgaussian_pdf(t, param) :
    """
    See param_model in documentation
    """
    alpha, beta, p = unpack_param(t, param)
    z              = alpha * (t - beta)
    return - 2.0 * z * alpha * gaussian_pdf(t, param)
