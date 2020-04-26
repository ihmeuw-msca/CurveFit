from dataclasses import dataclass
from typing import List, Callable, Tuple
import numpy as np

from curvefit.core.prototype import Prototype


class DataNotFoundError(Exception):
    pass


@dataclass
class DataInputs:
    """
    {begin_markdown DataInputs}

    {spell_markdown ndarray gprior param}

    # `curvefit.core.core_model.DataInputs`
    ## Provides the required data inputs for a `curvefit.core.core_model.Model`

    The `DataInputs` class holds all of the inputs that are needed for fitting
    a core model. It is only used in the `Model.convert_inputs()` method (
    see [here](Model.md). The purpose is to extract only the required elements
    of a `Data` class that are needed for model fitting in order to reduce the memory
    usage, but also keep key information for model debugging.

    ## Arguments

    - `t (np.ndarray)`: the time variable (or independent variable) in the curve
        fitting
    - `obs (np.ndarray)`: the observation variable (or dependent variable) in the
        curve fitting
    - `obs_se (np.ndarray)`: the observation standard error to attach to the observations
    - `covariates_matrices (List[np.ndarray])`: list of covariate matrices for each parameter
        (in many cases these covariate matrices will just be one column of ones)
    - `group_sizes (List[int])`: size of the groups
    - `num_groups (int)`: number of groups
    - `link_fun (List[Callable])`: list of link functions for the parameters
    - `var_link_fun (List[Callable])`: list of variable link functions for the variables
    - `x_init (np.ndarray)`: initial values for variables
    - `bounds (np.ndarray)`: bounds for variables
    - `fe_gprior (np.ndarray)`: array of fixed effects Gaussian priors for the variables
    - `re_gprior (np.ndarray)`: array of random effects Gaussian priors for the variables
    - `param_gprior_info (Tuple[Callable, List[float], List[float]])`: tuple of
        information about the parameter functional Gaussian priors;
        first element is a composite function of all of the parameter functional priors;
        second element is a list of means; third element is a list of standard deviations

    {end_markdown DataInputs}
    """

    t: np.ndarray
    obs: np.ndarray
    obs_se: np.ndarray
    covariates_matrices: List[np.ndarray] = None
    group_sizes: List[int] = None
    num_groups: int = None
    link_fun: List[Callable] = None
    var_link_fun: List[Callable] = None
    x_init: np.ndarray = None
    bounds: np.ndarray = None
    fe_gprior: np.ndarray = None
    re_gprior: np.ndarray = None
    param_gprior_info: Tuple[Callable, List[float], List[float]] = None


class Model(Prototype):

    def __init__(self):
        self.data_inputs = None

    def get_data(self):
        if self.data_inputs is not None:
            return self.data_inputs
        else:
            raise DataNotFoundError()

    def erase_data(self):
        self.data_inputs = None

    def detach_data(self):
        data_inputs = self.get_data()
        self.data_inputs = None
        return data_inputs

    def convert_inputs(self, data):
        raise NotImplementedError()

    def objective(self, x, data):
        raise NotImplementedError()

    def gradient(self, x, data):
        if self.data_inputs is None:
            self.data_inputs = self.convert_inputs(data)
        finfo = np.finfo(float)
        step = finfo.tiny / finfo.eps
        x_c = x + 0j
        grad = np.zeros(x.size)
        for i in range(x.size):
            x_c[i] += step*1j
            grad[i] = self.objective(x_c, data).imag/step
            x_c[i] -= step*1j

        return grad



