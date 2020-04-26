import numpy
from cppad_py import a_double
from curvefit.core.utils import unpack_param
from curvefit.core.numpy_ufunc import erf

# ---------------------------------------------------------------------------
# Model Functions
# ---------------------------------------------------------------------------
def a_gaussian_cdf(t, param) :
    alpha, beta, p = unpack_param(t, param)
    z              = alpha * (t - beta)
    return p * ( a_double(1.0) + erf(z) ) / a_double(2.0)
#
def a_expit(t, param) :
    alpha, beta, p = unpack_param(t, param)
    z              = alpha * (t - beta)
    return p / ( a_double(1.0) + numpy.exp(-z) )
#
def a_ln_expit(t, param) :
    alpha, beta, p = unpack_param(t, param)
    return numpy.log( a_expit(t, param) )
#
def a_ln_gaussian_cdf(t, param) :
    alpha, beta, p = unpack_param(t, param)
    return numpy.log( a_gaussian_cdf(t, param) )
#
def a_gaussian_pdf(t, param) :
    alpha, beta, p = unpack_param(t, param)
    z              = alpha * (t - beta)
    return alpha * p * numpy.exp( - z * z ) / numpy.sqrt(numpy.pi)
#
def a_ln_gaussian_pdf(t, param) :
    alpha, beta, p = unpack_param(t, param)
    z              = alpha * (t - beta)
    return numpy.log( alpha * p / numpy.sqrt(numpy.pi) ) - z * z
#
def a_dgaussian_pdf(t, param) :
    alpha, beta, p = unpack_param(t, param)
    z              = alpha * (t - beta)
    two            = a_double(2.0)
    return - two * z * alpha * a_gaussian_pdf(t, param)
