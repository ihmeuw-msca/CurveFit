import numpy
import cppad_py
import curvefit
import a_functions
#
def test_gaussian_pdf() :
    eps99  = 99.0 * numpy.finfo(float).eps
    #
    # test values for t, param
    t      = numpy.array( [ 5.0 , 10.0 ] )
    beta   = numpy.array( [ 30.0 , 20.0 ] )
    alpha  = 2.0 / beta
    p      = numpy.array( [ 0.1, 0.2 ] )
    param  = numpy.vstack( (alpha, beta, p) )
    #
    # aparam
    aparam = numpy.empty( param.shape , dtype = cppad_py.a_double )
    for i in range( param.shape[0] ) :
        for j in range( param.shape[1] ) :
            aparam[i][j] = cppad_py.a_double( param[i][j] )
    # -----------------------------------------------------------------------
    # f(t) = gaussian_cdf(t, param)
    at = cppad_py.independent(t)
    ay = a_functions.a_gaussian_cdf(at, aparam)
    f  = cppad_py.d_fun(at, ay)
    #
    # g(t) = d/dt gaussian_cdf(t, param)
    at = cppad_py.independent(t)
    ay = a_functions.a_gaussian_pdf(at, aparam)
    g  = cppad_py.d_fun(at, ay)
    #
    # check a_gaussian_pdf
    f.forward(0, t)
    g0  = g.forward(0, t)
    dt  = a_functions.constant_array((t.size,), 0.0)
    for i in range(len(t)) :
        dt[i]     = 1.0
        df        = f.forward(1, dt)
        rel_error = g0[i] / df[i] - 1.0
        dt[i]     = 0.0
        assert abs(rel_error) < eps99
    # -----------------------------------------------------------------------
    # check a_ln_dgaussain_cdf
    at = cppad_py.independent(t)
    ay = a_functions.a_ln_gaussian_pdf(at, aparam)
    f  = cppad_py.d_fun(at, ay)
    f0 = f.forward(0, t)
    rel_error = f0 / numpy.log(g0) - 1
    assert all( abs(rel_error) < eps99 )
