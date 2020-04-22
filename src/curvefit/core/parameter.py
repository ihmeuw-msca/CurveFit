import numpy as np


class Parameter:
    """
    {begin_markdown Parameter}

    {spell_markdown }

    # `curvefit.core.parameter.Parameter`
    ## Parameter metadata including priors, bounds, and initial values

    Contains all information for a particular parameter to be used
    in a functional form for curve fitting. This class will be passed
    to CoreModel.fit() and ModelProcess.fit(). Most use-cases will only
    have one fixed effect (covariate) per parameter. For a general use
    this class allows for any number of fixed effects for one parameter
    and is determined by what is passed in for the `covariates` argument.
    Parameter- versus fixed effect-specific arguments are highlighted
    [arguments](#arguments) below.

    ## Syntax
    ```python
    parameter = Parameter(
        param_name, link_fun, var_link_fun, covariates,
        fe_init, re_init, fe_gprior, re_gprior, fe_bounds, re_bounds
    )
    ```

    ## Arguments
    Parameter-specific arguments are denoted with (*), in contrast to
    fixed effect-specific arguments (e.g. having multiple covariates).

    - (*) `param_name (str)`: name of parameter, e.g. 'alpha'
    - (*) `link_fun (callable)`: the link function for the parameter
    - `covariates (str)`: list of covariate names to be applied to the parameter
    - `var_link_fun (List[callable])`: list of link functions for each
        fixed effect
    - `fe_init (List[float])`: list of the initial values for the fixed effects
        to be used in the optimization
    - (*) `re_init (List[float])`: list of the initial values for the random effects
        to be used in the optimization
    - `fe_gprior (optional, List[List[float]])`: list of lists of Gaussian priors
        for each fixed effect on the parameter where the inner list
        has two elements, one for the mean and one for the standard deviation
    - (*) `re_gprior (optional, List[List[float]])`: list of lists of Gaussian priors
        for each random effect on fixed effect where the inner list
        has two-elements list of mean and standard deviation for the random effects
    - `fe_bounds (optional, List[List[float]])`: list of lists of box constraints
        for the fixed effects during the optimization where the inner
        list has two elements: lower and upper constraint
    - (*) `re_bounds (optional, List[List[float]])`: list of lists of box constraints
        for the random effects during the optimization where the inner
        list has two elements: lower and upper constraint

    ## Attributes

    - `self.num_fe (int)`: number of fixed effects

    ## Usage

    ```python
    parameter = Parameter(
        param_name='alpha', link_fun=lambda x: x,
        var_link_fun=[lambda x: x, lambda x: x],
        covariates=['covariate1', 'covariate2'],
        fe_gprior=[[0.0, 0.01], [0.0, 0.1]],
        re_gprior=[[0.0, 1e-4], [0.0, 1e-4]],
        fe_init=[2., 1.],
        re_init=[0., 0.]
    )
    ```

    {end_markdown Parameter}
    """

    def __init__(self, param_name, link_fun, covariates,
                 var_link_fun, fe_init, re_init,
                 fe_gprior=None, re_gprior=None, fe_bounds=None, re_bounds=None):

        self.param_name = param_name
        self.link_fun = link_fun
        self.covariates = covariates
        self.var_link_fun = var_link_fun

        self.fe_init = fe_init
        self.re_init = re_init

        self.num_fe = len(self.covariates)

        self.fe_gprior = fe_gprior
        self.re_gprior = re_gprior
        self.fe_bounds = fe_bounds
        self.re_bounds = re_bounds

        if self.fe_gprior is None:
            self.fe_gprior = [[0., np.inf]] * self.num_fe
        if self.re_gprior is None:
            self.re_gprior = [[0., np.inf]] * self.num_fe
        if self.fe_bounds is None:
            self.fe_bounds = [[-np.inf, np.inf]] * self.num_fe
        if self.re_bounds is None:
            self.re_bounds = [[-np.inf, np.inf]] * self.num_fe

        assert type(self.param_name) == str
        assert callable(self.link_fun)

        assert len(self.var_link_fun) == self.num_fe
        assert len(self.fe_gprior) == self.num_fe
        assert len(self.fe_bounds) == self.num_fe
        assert len(self.fe_init) == self.num_fe

        for i in range(self.num_fe):

            assert callable(self.var_link_fun[i])

            assert len(self.fe_bounds[i]) == 2
            for j in self.fe_bounds[i]:
                assert type(j) == float

            assert len(self.re_bounds[i]) == 2
            for j in self.re_bounds[i]:
                assert type(j) == float

            assert len(self.fe_gprior[i]) == 2
            assert type(self.fe_gprior[i][0]) == float
            assert type(self.fe_gprior[i][1]) == float
            assert self.fe_gprior[i][1] > 0.

            assert len(self.re_gprior[i]) == 2
            assert type(self.re_gprior[i][0]) == float
            assert type(self.re_gprior[i][1]) == float
            assert self.re_gprior[i][1] > 0.

            assert type(self.fe_init[i]) == float
            assert type(self.re_init[i]) == float


class ParameterSet:
    """
    {begin_markdown ParameterSet}

    {spell_markdown }

    # `curvefit.core.parameter.ParameterSet`
    ## Set of Parameters that will be used in a model

    Links instances of the Parameter class with one another to include all the parameters
    that will go into a model. Can also include functional priors based on functions of the parameters.

    ## Syntax
    ```python
    parameter = ParameterSet(
        parameter_list,
        parameter_functions,
        parameter_function_priors
    )
    ```

    ## Arguments

    - `parameter_list (List[Parameter])`: list of instances of Parameter
        class
    - `parameter_functions (optional, List[callable])`: list of functions of the parameter_list
    - `parameter_function_priors (optional, List[List[float]])`: list of functional priors
        that matches with `parameter_functions`

    ## Attributes

    - `self.num_fe (int)`: total number of fixed effects across parameters
    - `self.num_params (int)`: number of parameters

    ## Usage

    ```python
    parameter1 = Parameter(
        param_name='alpha', link_fun=lambda x: x,
        var_link_fun=[lambda x: x], covariates=['covariate1'],
        fe_init=[0.], re_init=[0.]
    )
    parameter2 = Parameter(
        param_name='beta', link_fun=lambda x: x,
        var_link_fun=[lambda x: x], covariates=['covariate2'],
        fe_init=[0.], re_init=[0.]
    )
    parameter = ParameterSet(
        parameter_list=[parameter1, parameter2],
        parameter_functions=[lambda params: params[0] * params[1]],
        parameter_function_priors=[[0.0, np.inf]]
    )
    ```

    {end_markdown ParameterSet}
    """
    def __init__(self, parameter_list, parameter_functions=None, parameter_function_priors=None):

        self.parameter_list = parameter_list
        self.parameter_functions = parameter_functions
        self.parameter_function_priors = parameter_function_priors

        assert type(self.parameter_list) == list
        for param in self.parameter_list:
            assert type(param) == Parameter

        self.num_params = len(self.parameter_list)
        self.num_fe = 0
        for param in self.parameter_list:
            self.num_fe += param.num_fe

        if self.parameter_functions is None and self.parameter_function_priors is not None:
            raise RuntimeError("Can't use a functional prior without providing the function.")

        if self.parameter_functions is not None:

            assert type(self.parameter_functions) == list
            for param_func in self.parameter_functions:
                assert callable(param_func)

            if self.parameter_function_priors is None:
                self.parameter_function_priors = [[0., np.inf]] * len(self.parameter_functions)

            assert len(self.parameter_function_priors) == len(self.parameter_functions)
            for param_func_prior in self.parameter_function_priors:
                assert len(param_func_prior) == 2
                assert type(param_func_prior[0]) == float
                assert type(param_func_prior[1]) == float
                assert param_func_prior[1] > 0.
