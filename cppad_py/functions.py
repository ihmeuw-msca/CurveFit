import numpy
import cppad_py
import scipy

def erf(vec) :
    assert isinstance(vec, numpy.ndarray)
    assert vec.ndim == 1
    #
    if isinstance(vec[0], cppad_py.a_double) :
        result = numpy.empty(len(vec), dtype = cppad_py.a_double )
        for i in range( len(vec) ) :
            result[i] = vec[i].erf()
    else :
        result = scipy.special.erf(vec)
    return result
#
def exp(vec) :
    assert isinstance(vec, numpy.ndarray)
    assert vec.ndim == 1
    #
    if isinstance(vec[0], cppad_py.a_double) :
        result = numpy.empty(len(vec), dtype = cppad_py.a_double )
        for i in range( len(vec) ) :
            result[i] = vec[i].exp()
    else :
        result = numpy.exp(vec)
    return result
#
def gaussian_cdf(t, param) :
    assert param.shape[0] == 3
    assert t.ndim == 1
    if param.ndim == 2 :
        assert param.shape[1] == t.shape[0]
    alpha  = param[0]
    beta   = param[1]
    p      = param[2]
    z      = alpha * (t - beta)
    return p * ( cppad_py.a_double(1.0) + erf(z) ) / cppad_py.a_double(2.0)
#
def expit(t, param) :
    assert param.shape[0] == 3
    assert t.ndim == 1
    if param.ndim == 2 :
        assert param.shape[1] == t.shape[0]
    alpha  = param[0]
    beta   = param[1]
    p      = param[2]
    z      = alpha * (t - beta)
    return p / ( cppad_py.a_double(1.0) + exp(-z) )
#
