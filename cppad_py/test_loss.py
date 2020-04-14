import numpy
import cppad_py
import curvefit
import a_functions
#
def test_loss() :
    eps99  = 99.0 * numpy.finfo(float).eps
    #
    # test values for t, param
    r  = numpy.array( [ 1, 2, 3], dtype=float )
    nu = numpy.array( [ 3, 2, 1], dtype=float )
    # -----------------------------------------------------------------------
    # f(t) = st_loss
    ar     = cppad_py.independent(r)
    anu    = a_functions.array2a_double(nu)
    aloss  = a_functions.a_st_loss(ar, anu)
    ay     = numpy.array( [ aloss ] )
    f      = cppad_py.d_fun(ar, ay)
    #
    y          = f.forward(0, r)
    check      = curvefit.core.functions.st_loss(r, nu)
    rel_error  = y[0] / check - 1.0
    # -----------------------------------------------------------------------
    # f(t) = normal_loss
    ar     = cppad_py.independent(r)
    aloss  = a_functions.a_normal_loss(ar)
    ay     = numpy.array( [ aloss ] )
    f      = cppad_py.d_fun(ar, ay)
    #
    y          = f.forward(0, r)
    check      = curvefit.core.functions.normal_loss(r)
    rel_error  = y[0] / check - 1.0
