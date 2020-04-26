import numpy
from cppad_py import a_double
from curvefit.core.utils import unpack_param

# ---------------------------------------------------------------------------
# Local Functions
# ---------------------------------------------------------------------------
def constant_array(shape, value) :
    size = 1
    for dim in shape :
        size *= dim
    vec = numpy.empty(size, dtype=type(value))
    for i in range(size) :
        vec[i] = value
    return numpy.reshape(vec, shape)
#
def array2a_double(array) :
    shape = array.shape
    size  = array.size
    vec   = numpy.reshape(array, size, order='C')
    a_vec = numpy.empty(shape, dtype=a_double)
    for i in range(size) :
        a_vec[i] = a_double( vec[i] )
    return a_vec
#
def a_erf(vec) :
    result = numpy.empty(len(vec), dtype = a_double )
    for i in range( len(vec) ) :
        result[i] = vec[i].erf()
    return result
#
# ---------------------------------------------------------------------------
# Model Functions
# ---------------------------------------------------------------------------
def a_gaussian_cdf(t, param) :
    alpha, beta, p = unpack_param(t, param)
    z              = alpha * (t - beta)
    return p * ( a_double(1.0) + a_erf(z) ) / a_double(2.0)
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
