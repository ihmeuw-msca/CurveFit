from dataclasses import fields, field, InitVar
from pydantic.dataclasses import dataclass
from typing import List, Callable
import numpy as np

from curvefit.core.prototype import Prototype


@dataclass
class Variable:
    """
    {begin_markdown Variable}

    {spell_markdown init gprior}

    # `curvefit.core.parameter.Variable`
    ## A variable to be estimated during the fit

    A `Variable` is the most detailed unit to be estimated for curve fitting. A `Variable` corresponds
    to an effect on a parameter -- and can contain a combination of both a fixed effect and random effects.
    A `Variable` needs a `"covariate"`, but the covariate in the data can be just a column of 1's, in which
    case a `Variable` is equivalent to a [`Parameter`](Parameter.md). If instead the values of
    the `"covariate"` argument differ for different rows of the data, `Variable` multiplies the covariate
    to get the `Parameter` for that data row.

    A `curvefit` model is made up of multiple parameters. For more information, see
    [`Parameter`](Parameter.md) and [`ParameterSet`](ParameterSet.md).

    ## Arguments

    - `covariate (str)`: name of the covariate for this variable (corresponds to what it will be in the data
        that is eventually used to fit the model)
    - `var_link_fun (Callable)`: link function for the variable
    - `fe_init (float)`: initial value to be used in the optimization for the fixed effect
    - `re_init (float)`: initial value to be used in the optimization for the random effect
    - `fe_gprior (optional, List[float])`: list of Gaussian priors
        the fixed effect where the first element is the prior
        mean and the second element is the prior standard deviation
    - `re_gprior (optional, List[float])`: list of Gaussian priors
        the random effect where the first element is the prior
        mean and the second element is the prior standard deviation
    - `fe_bounds (optional, List[float])`: list of box constraints
        for the fixed effects during the optimization where the first element is the lower bound
        and the second element is the upper bound
    - `re_bounds (optional, List[float])`: list of box constraints
        for the fixed effects during the optimization where the first element is the lower bound
        and the second element is the upper bound

    ## Usage

    ```python
    from curvefit.core.parameter import Variable

    var = Variable(covariate='ones', var_link_fun=lambda x: x, fe_init=0., re_init=0.)
    ```

    {end_markdown Variable}
    """

    covariate: str 
    var_link_fun: Callable
    fe_init: float
    re_init: float
    fe_gprior: List[float] = field(default_factory=lambda: [0.0, np.inf])
    re_gprior: List[float] = field(default_factory=lambda: [0.0, np.inf])
    fe_bounds: List[float] = field(default_factory=lambda: [-np.inf, np.inf])
    re_bounds: List[float] = field(default_factory=lambda: [-np.inf, np.inf]) 

    def __post_init__(self):
        assert isinstance(self.covariate, str)
        assert len(self.fe_gprior) == 2
        assert len(self.re_gprior) == 2
        assert len(self.fe_bounds) == 2
        assert len(self.re_bounds) == 2
        assert self.fe_gprior[1] > 0.0
        assert self.re_gprior[1] > 0.0


@dataclass
class Parameter:
    """
    {begin_markdown Parameter}

    {spell_markdown param init}

    # `curvefit.core.parameter.Parameter`
    ## A parameter for the functional form of the curve

    A `Parameter` is a parameter of the functional form for the curve. For example, if the parametric curve you want
    to fit has three parameters then you need 3 `Parameter` objects. A `Parameter` is made up of one or more
    [`Variable`](Variable.md) objects, which represent the fixed and random effects. Parameters may have
    link functions that transform the parameter into some other
     space to enforce, for example, positivity of the parameter (e.g. the parameter representing the scale of a
     Gaussian distribution can't be negative).

    ## Arguments

    - `param_name (str)`: name of parameter, e.g. 'alpha'
    - `link_fun (Callable)`: the link function for the parameter
    - `variables (List[curvefit.core.parameter.Variable])`: a list of `Variable` instances

    ## Attributes

    All attributes from the `Variable`s in the list in the `variables` argument are carried over to
    `Parameter` but they are put into a list. For example, the `fe_init` attribute for `Parameter` is a list of
    `fe_init` attributes for each `Variable` in the order that they were passed in `variables` list.

    *Additional* attributes that are not lists of the individual `Variable` attributes are listed below.

    - `self.num_fe (int)`: total number of effects for the parameter (number of variables)

    ## Usage

    ```python
    from curvefit.core.parameter import Parameter, Variable

    var = Variable(covariate='ones', var_link_fun=lambda x: x, fe_init=0., re_init=0.)
    param = Parameter(param_name='alpha', link_fun=lambda x: x, variables=[var])
    ```

    {end_markdown Parameter}
    """

    param_name: str
    link_fun: Callable
    variables: InitVar[List[Variable]]

    num_fe: int = field(init=False)
    covariate: List[str] = field(init=False)
    var_link_fun: List[Callable] = field(init=False)
    fe_init: List[float] = field(init=False)
    re_init: List[float] = field(init=False)
    fe_gprior: List[List[float]] = field(init=False)
    re_gprior: List[List[float]] = field(init=False)
    fe_bounds: List[List[float]] = field(init=False)
    re_bounds: List[List[float]] = field(init=False)

    def __post_init__(self, variables):
        assert isinstance(variables, list)
        assert len(variables) > 0
        assert isinstance(variables[0], Variable)
        self.num_fe = len(variables)
        for k, v in consolidate(Variable, variables).items():
            self.__setattr__(k, v)


@dataclass
class ParameterFunction:

    param_function_name: str
    param_function: Callable
    param_function_fe_gprior: List[float] = field(default_factory=lambda: [0.0, np.inf])

    def __post_init__(self):
        assert isinstance(self.param_function_name, str)
        assert len(self.param_function_fe_gprior) == 2
        assert self.param_function_fe_gprior[1] > 0.0


@dataclass
class ParameterSet(Prototype):
    """
    {begin_markdown ParameterSet}

    {spell_markdown init inits param params gprior}

    # `curvefit.core.parameter.ParameterSet`
    ## A set of parameters that together specify the functional form of a curve

    A `ParameterSet` is a set of parameters that define the functional form for the curve.
    For example, if the parametric curve you want
    to fit has three parameters then you need 1 `ParameterSet` objects that consists of 3 `Parameter` objects.

    A `ParameterSet` is made up of one or more
    [`Parameter`](Parameter.md) objects, which are each made up of one or more [`Variable`](Variable.md) objects.
    Please refer to their documentation for more details on those objects.

    A `ParameterSet` can also encode functional priors -- priors for functions of the parameter list that is
    passed into a `ParameterSet`.

    ## Arguments

    - `parameters (List[curvefit.core.parameter.Parameter])`: a list of `Parameter` instances
    - `parameter_functions (List[Tuple[Callable, List[float]]]`: a list of tuples which each contain
    (0) functions to apply to the `parameters` list and
    (1) a prior for the parameter function (mean and standard deviation --
    see [`Variable`](Variable.md#arguments) for more details about priors)

    ## Attributes

    All attributes from the `Parameter`s in the list in the `parameters` argument are carried over to
    `ParameterSet` but they are put into a list. For example, the `fe_init` attribute for `ParameterSet` is a list of
    the `fe_init` attributes for each `Parameter` in the order that they were passed in `parameters` list (which
    are lists of `fe_inits` for each `Variable` within a `Parameter` (see [here](Parameter.md#attributes) for more).

    *Additional* attributes that are not lists of the individual `Parameter` attributes are listed below.

    - `self.num_fe (int)`: total number of effects for the parameter set (number of variables)

    ## Usage

    ```python
    from curvefit.core.parameter import Parameter, Variable, ParameterSet

    var = Variable(covariate='ones', var_link_fun=lambda x: x, fe_init=0., re_init=0.)
    param = Parameter(param_name='alpha', link_fun=lambda x: x, variables=[var])
    param_function = ParameterFunction(
        param_function_name='alpha_squared',
        param_function=lambda params: params[0] ** 2,
        param_function_fe_gprior=[0., np.inf]
    )
    param_set = ParameterSet(
        parameters=[param], parameter_functions=[param_function]
    )
    ```

    {end_markdown ParameterSet}
    """

    parameters: InitVar[List[Parameter]]
    parameter_functions: InitVar[List[ParameterFunction]] = None

    param_name: List[str] = field(init=False)
    num_fe: int = field(init=False)
    link_fun: List[Callable] = field(init=False)
    covariate: List[List[str]] = field(init=False)
    var_link_fun: List[List[Callable]] = field(init=False)
    fe_init: List[List[float]] = field(init=False)
    re_init: List[List[float]] = field(init=False)
    fe_gprior: List[List[List[float]]] = field(init=False)
    re_gprior: List[List[List[float]]] = field(init=False)
    fe_bounds: List[List[List[float]]] = field(init=False)
    re_bounds: List[List[List[float]]] = field(init=False)

    param_function_name: List[str] = field(init=False)
    param_function: List[Callable] = field(init=False)
    param_function_fe_gprior: List[List[float]] = field(init=False)

    def __post_init__(self, parameters, parameter_functions):

        for k, v in consolidate(Parameter, parameters, exclude=['num_fe']).items():
            self.__setattr__(k, v)

        for k, v in consolidate(ParameterFunction, parameter_functions).items():
            self.__setattr__(k, v)

        if len(set(self.param_name)) != len(self.param_name):
            raise RuntimeError("Cannot have duplicate parameters in a set.")
        if len(set(self.param_function_name)) != len(self.param_function_name):
            raise RuntimeError("Cannot have duplicate parameter functions in a set.")

        self.num_fe = 0
        for param in parameters:
            self.num_fe += param.num_fe

    def get_param_index(self, param_name):
        try:
            param_index = self.param_name.index(param_name)
        except ValueError:
            raise RuntimeError(f"No {param_name} parameter in this parameter set.")
        return param_index


def consolidate(cls, instance_list, exclude=None):
    if exclude is None:
        exclude = []
    consolidated = {}
    for f in fields(cls):
        if f.name not in exclude:
            if instance_list is not None:
                consolidated[f.name] = [instance.__getattribute__(f.name) for instance in instance_list]
            else:
                consolidated[f.name] = list()
    return consolidated
