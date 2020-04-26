import numpy
import cppad_py
import curvefit
#
def test_loss() :
    eps99  = 99.0 * numpy.finfo(float).eps
    #
    # test values for t, nu
    r  = numpy.array( [ 1, 2, 3], dtype=float )
    nu = 2.0
    # -----------------------------------------------------------------------
    # f(t) = st_loss
    ar     = cppad_py.independent(r)
    aloss  = curvefit.core.loss.st_loss(ar, nu)
    ay     = numpy.array( [ aloss ] )
    f      = cppad_py.d_fun(ar, ay)
    #
    y          = f.forward(0, r)
    check      = curvefit.core.loss.st_loss(r, nu)
    rel_error  = y[0] / check - 1.0
    # -----------------------------------------------------------------------
    # f(t) = normal_loss
    ar     = cppad_py.independent(r)
    aloss  = curvefit.core.loss.normal_loss(ar)
    ay     = numpy.array( [ aloss ] )
    f      = cppad_py.d_fun(ar, ay)
    #
    y          = f.forward(0, r)
    check      = curvefit.core.loss.normal_loss(r)
    rel_error  = y[0] / check - 1.0
