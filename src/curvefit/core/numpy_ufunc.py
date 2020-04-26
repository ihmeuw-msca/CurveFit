import numpy
import scipy
"""
{begin_markdown numpy_ufunc}
{spell_markdown erf ufunc}

# Extend Numpy Universal Functions

## Syntax
`result = fun(array)`

## array
is a numpy array with `dtype` equal to `float` or `numpy.dtype('O')`
(also referred to as object type).
If the data type is object and `element` is an element of the array,
the object must support the member function `element.fun()`.

## fun
The functions listed below are the possible values for *fun$*.
More functions may be added to the list in the future.

### erf
The error function.

## result
is a numpy array, with the same shape as *array*,
where each element of *result* is the function value for the
corresponding element of *array*.

# Example
[numpy_ufunc](numpy_ufunc_xam.md)

{end_markdown numpy_ufunc}
"""
def erf(array) :
    """
    see numpy_ufunc in developer documentation
    """
    if array.dtype == numpy.dtype('O') :
        shape  = array.shape
        size   = array.size
        vec    = numpy.reshape(array, size, order='C')
        result = numpy.empty(size, dtype = numpy.dtype('O') )
        for i in range( len(vec) ) :
            result[i] = vec[i].erf()
            result = numpy.reshape(result, shape, order='C')
    else :
        result  = scipy.special.erf(array)
    return result
