import numpy
import scipy
from cppad_py import a_double

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
def gaussian_cdf(t, param) :
    alpha, beta, p = unpack_param(t, param)
    z      = alpha * (t - beta)
    return p * ( a_double(1.0) + a_erf(z) ) / a_double(2.0)
#
def expit(t, param) :
    alpha, beta, p = unpack_param(t, param)
    z      = alpha * (t - beta)
    return p / ( a_double(1.0) + a_exp(-z) )
#
