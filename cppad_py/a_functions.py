import numpy
import scipy
from cppad_py import a_double

def constant_array(shape, value) :
    size = 1
    for dim in shape :
        size *= dim
    vec = numpy.empty(size, dtype=type(value))
    for i in range(size) :
        vec[i] = value
    return numpy.reshape(vec, shape)
#
def a_erf(vec) :
    result = numpy.empty(len(vec), dtype = a_double )
    for i in range( len(vec) ) :
        result[i] = vec[i].erf()
    return result
#
def a_exp(vec) :
    result = numpy.empty(len(vec), dtype = a_double )
    for i in range( len(vec) ) :
        result[i] = vec[i].exp()
    return result
#
def unpack_param(t, param) :
    assert param.shape[0] == 3
    assert t.ndim == 1
    if param.ndim == 2 :
        assert param.shape[1] == t.shape[0]
    assert t.dtype == a_double
    assert param.dtype == a_double
    #
    alpha  = param[0]
    beta   = param[1]
    p      = param[2]
    return alpha, beta, p
#
def a_gaussian_cdf(t, param) :
    alpha, beta, p = unpack_param(t, param)
    z              = alpha * (t - beta)
    return p * ( a_double(1.0) + a_erf(z) ) / a_double(2.0)
#
def a_expit(t, param) :
    alpha, beta, p = unpack_param(t, param)
    z              = alpha * (t - beta)
    return p / ( a_double(1.0) + a_exp(-z) )
#
def a_log_expit(t, param) :
    alpha, beta, p = unpack_param(t, param)
    return numpy.log( a_expit(t, param) )
#
def a_log_gaussian_cdf(t, param) :
    alpha, beta, p = unpack_param(t, param)
    return numpy.log( a_gaussian_cdf(t, param) )
#
def a_dgaussian_cdf(t, param) :
    alpha, beta, p = unpack_param(t, param)
    z              = alpha * (t - beta)
    return alpha * p * a_exp( - z * z ) / numpy.sqrt(numpy.pi)
#
def a_log_dgaussian_cdf(t, param) :
    alpha, beta, p = unpack_param(t, param)
    z              = alpha * (t - beta)
    return numpy.log( alpha * p / numpy.sqrt(numpy.pi) ) - z * z
#
def a_ddgaussian_cdf(t, param) :
    alpha, beta, p = unpack_param(t, param)
    z              = alpha * (t - beta)
    two            = a_double(2.0)
    return - two * z * alpha * a_dgaussian_cdf(t, param)
