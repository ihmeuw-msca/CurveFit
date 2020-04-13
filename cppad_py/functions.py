import warnings
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
def a_log_gaussian_cdf(t, param) :
    alpha, beta, p = unpack_param(t, param)
    z        = alpha * (t - beta)
    cop      = '>'
    left     = a_gaussian_cdf(t, param)
    right    = a_double(0.0)
    #
    # will do a conditional assignment so supress warnings
    warnings.filterwarnings("ignore")
    if_true  = numpy.log( left )
    if_false  = numpy.log( p / a_double(2.0) ) - z*z - \
        numpy.log(-z) - a_double( 0.5 * numpy.log( numpy.pi ) )
    warnings.filterwarnings("default")
    #
    # There is a default constructor for a_double, but numpy
    # initializes the elements to None instead of a_dobule.
    result  = numpy.empty( len(t), dtype=a_double)
    for i in range( len(t) ) :
        result[i] = a_double(0.0)
        result[i].cond_assign(cop, left[i], right, if_true[i], if_false[i])
    return result
