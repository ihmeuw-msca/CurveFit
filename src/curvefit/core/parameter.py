from pydantic.dataclasses import dataclass, fields 
from typing import List

import numpy as np 

@dataclass  
class Variable:

    covariate: str 
    var_link_fun: callable 
    fe_init: float
    re_init: float
    fe_gprior: List[float] = None
    re_gprior: List[float] = None 
    fe_bounds: List[float] = None 
    re_bounds: List[float] = None 

    def __post_init__(self):
        if self.fe_gprior is None:
            self.fe_gprior = [0.0, np.inf]
        if self.re_gprior is None:
            self.re_gprior = [0.0, np.inf]
        if self.fe_bounds is None:
            self.fe_bounds = [-np.inf, np.inf]
        if self.re_bounds is None:
            self.re_bounds = [-np.inf, np.inf]
        
        assert len(self.fe_gprior) == 2
        assert len(self.re_gprior) == 2
        assert len(self.fe_bounds) == 2
        assert len(self.re_bounds) == 2
        assert self.fe_gprior[1] > 0.0
        assert self.re_gprior[1] > 0.0


@dataclass
class Parameter:

    param_name: str
    link_fun: callable
    variables: List[Variable]

    def __post_init__(self):
        for k, v in consolidate(Variable, self.variables).items():
            self.__setattr__(k, v)


@dataclass
class ParameterSet:
    parameters: List[Parameter]
    parameter_functions: List[List[callable, List[float]]]

    def __post_init__(self):
        for k, v in consolidate(Parameter, self.parameters).items():
            self.__setattr__(k, v)
        
        for fun in self.parameter_functions:
            assert len(fun[1]) == 2


def consolidate(cls, instance_list):
    consolidated = {}
    for field in fields(cls):
        consolidated[field.name] = [instance.__getattribute__(field.name) for instance in instance_list]
    return consolidated




    

        



