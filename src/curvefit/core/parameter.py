from dataclasses import fields, field, InitVar
from pydantic.dataclasses import dataclass
from typing import List, Callable, Tuple

import pdb

import numpy as np 

@dataclass  
class Variable:

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

    param_name: str
    link_fun: Callable
    variables: InitVar[List[Variable]]
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
        for k, v in consolidate(Variable, variables).items():
            self.__setattr__(k, v)


@dataclass
class ParameterSet:
    parameters: List[Parameter]
    parameter_functions: List[Tuple[Callable, List[float]]] = None
    param_name: List[str] = field(init=False)
    link_fun: List[Callable] = field(init=False)
    covariate: List[List[str]] = field(init=False)
    var_link_fun: List[List[Callable]] = field(init=False)
    fe_init: List[List[float]] = field(init=False)
    re_init: List[List[float]] = field(init=False)
    fe_gprior: List[List[List[float]]] = field(init=False)
    re_gprior: List[List[List[float]]] = field(init=False)
    fe_bounds: List[List[List[float]]] = field(init=False)
    re_bounds: List[List[List[float]]] = field(init=False)

    def __post_init__(self):
        if self.parameter_functions is not None:
            for fun in self.parameter_functions:
                assert len(fun[1]) == 2
                assert isinstance(fun[0], Callable)

        for k, v in consolidate(Parameter, self.parameters).items():
            self.__setattr__(k, v)


def consolidate(cls, instance_list, exclude=None):
    if exclude is None:
        exclude = []
    consolidated = {}
    for field in fields(cls):
        if field.name not in exclude:
            consolidated[field.name] = [instance.__getattribute__(field.name) for instance in instance_list]
    return consolidated
