#! /usr/bin/env python3
import sys
import numpy
import cppad_py
from cppad_py import a_double
#
import sandbox
sandbox.path()
import curvefit

def erf(array) :
    vec  = numpy.reshape(array, array.size, order='C')
    for i in range(array.size) :
        vec[i] = vec[i].erf()
    return numpy.reshape(vec, array.shape, order='C')

def gaussian_cdf(t, param) :
    assert param.shape[0] == 3
    assert t.ndim == 1
    if param.ndim == 2 :
        assert param.shape[1] == t.shape[0]
    alpha  = param[0]
    beta   = param[1]
    p      = param[2]
    z      = alpha * (t - beta)
    return p * ( a_double(1.0) + erf(z) ) / a_double(2.0)
#
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
aparam = numpy.empty( param.shape , dtype = a_double )
for i in range( param.shape[0] ) :
    for j in range( param.shape[1] ) :
        aparam[i][j] = a_double( param[i][j] )
#
# f(t) = gaussian_cdf(t, param)
at = cppad_py.independent(t)
ay = gaussian_cdf(at, aparam)
f  = cppad_py.d_fun(at, ay)
#
# zero order foward mode using same values as during recording
y  = f.forward(0, t)
#
# check using curvefit values for same function
check     = curvefit.core.functions.erf(t, param)
rel_error = y / check - 1.0
assert all( abs( rel_error ) < eps99 )
#
# compute entire Jacobian of f
# (could instead calculate a sparse Jacobian here).
J = f.jacobian(t)
assert J.shape[0] == t.size
assert J.shape[1] == t.size
#
# check using curvefitl values for derivative function
check = curvefit.core.functions.derf(t, param)
for i in range( t.size ) :
    for i in range( t.size ) :
        if i == j :
            rel_error = J[i,j] / check[i] - 1.0
            assert abs( rel_error ) < eps99
        else :
            assert J[i,j] == 0.0
#
print('gaussian_cdf_xam.py: OK')
sys.exit(0)
