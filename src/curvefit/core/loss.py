# -------------------------------------------------------------------------
# Loss Functions
# --------------------------------------------------------------------------
import numpy
#
# Student's T loss function
def st_loss(r, nu=1.0):
    '''{begin_markdown st_loss}
    {spell_markdown }

    # Student's t Loss Function

    ## Syntax
    `loss = curvefit.core.loss.st_loss(r, nu = 1.0)`

    ## t
    is a numpy vector of residual values. We use \( n \)
    for the length of the vector.
    The elements of this vector can be `float` or `a_double` values.

    ## nu
    is the number of degrees of freedom in the t distribution \( \nu \).
    This can be a `float` or `a_double` value.


    ## Distribution
    The student's t-distribution is
    \[
        f(r) = ( 1 + r^2 / \nu )^{- (\nu + 1) / 2 }
             \Gamma[ ( \nu + 1) / 2 ] / [ \sqrt{ \nu \pi } \Gamma( \nu / 2 ) ]
    \]
    where \( \nu \) is the number of degrees of freedom and
    \( \Gamma \) is the gamma function.

    ## Negative log
    Taking the negative log of the distribution function we get
    \[
        - \log [ f(r) ] = \log ( 1 + r^2 / \nu ) (\nu + 1) / 2  + c
    \]
    where \( c \) is constant w.r.t. \( r \).

    ## loss
    The return value `loss` is a scalar equal to
    \[
        \frac{\nu + 1}{2} \sum_{i=0}^{n-1} \log( 1 + r_i^2 / \nu )
    \]

    ## Example
    [loss_xam](loss_xam.md)

    {end_markdown st_loss}
    '''
    return numpy.sum( numpy.log(1.0 + r * r / nu) )

# Gaussian loss function
def normal_loss(r):
    '''{begin_markdown normal_loss}
    {spell_markdown }

    # Gaussian Loss Function

    ## Syntax
    `loss = curvefit.core.loss.normal_loss(r)`

    ## r
    is a numpy vector of normalized residual values. We use \( n \)
    for the length of the vector.
    The elements of this vector can be `float` or `a_double` values.


    ## Distribution
    The Gaussian distribution is
    \[
        f(x) = \exp \left[ - (1/2) ( x - \mu )^2 / \sigma^2 \right] /
            \left( \sigma \sqrt{ 2 \pi } \right)
    \]
    where \( \mu \) is the mean and \( \sigma \) is the standard deviation.

    ## Negative log
    Taking the negative log of the distribution function we get
    \[
        - \log [ f(x) ] = (1/2) ( x - \mu )^2 / \sigma^2 + c
    \]
    where \( c \) is constant w.r.t. \( x \).

    ## loss
    The return value `loss` is a scalar equal to
    \[
        \frac{1}{2} \sum_{i=1}^{n-1} r_i^2
    \]
    where \( r_i = ( x_i - \mu)) / \sigma \).

    ## Example
    [loss_xam](loss_xam.md)

    {end_markdown normal_loss}
    '''
    return 0.5 * numpy.sum(r * r)
