import numpy
import scipy
"""
{begin_markdown numpy_ufunc}
{spell_markdown erf ufunc obj}

# Extend Numpy Universal Functions

## Syntax
`result = fun(arg)`

## Notation
An object `obj` supports *fun* if it has the member function *fun*;
i.e., `obj.fun()` is supported.

## arg
is a numpy array or a scalar value.
If it is an array, each element of the array is a `float` or supports *fun*.
If it is a scalar, it is a `float` or it supports *fun*.

## fun
The functions listed below are the possible values for *fun*.
More functions may be added to the list in the future.

### erf
The error function.

## result
If *arg* is a numpy array, *result* is an array with the same shape
where each element is the function value for the
corresponding element of *arg*.
If *arg* is a scalar, the result is the function value for *arg*.

# Example
[numpy_ufunc](numpy_ufunc_xam.md)

{end_markdown numpy_ufunc}
"""
def erf(arg) :
    """
    see numpy_ufunc in developer documentation
    """
    if isinstance(arg, numpy.ndarray) :
        if arg.dtype == numpy.dtype('O') :
            shape  = arg.shape
            size   = arg.size
            vec    = numpy.reshape(arg, size, order='C')
            result = numpy.empty(size, dtype = numpy.dtype('O') )
            for i in range( len(vec) ) :
                result[i] = vec[i].erf()
            result = numpy.reshape(result, shape, order='C')
        else :
            result  = scipy.special.erf(arg)
    elif isinstance(arg, float) :
        result  = scipy.special.erf(arg)
    else :
        result = arg.erf()
    return result
