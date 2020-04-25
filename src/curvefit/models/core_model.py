from functools import reduce
from operator import iconcat
from dataclasses import dataclass
from typing import List, Callable, Tuple
import numpy as np
from curvefit.core.objective_fun import objective_fun


@dataclass
class DataInputs:
    """
    {begin_markdown DataInputs}

    {spell_markdown }

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
    - `link_fun (List[Callable])`: list of link functions for the parameters
    - `var_link_fun (List[Callable])`: list of variable link functions for the variables
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
    covariates_matrices: List[np.ndarray]
    group_sizes: List[int]
    link_fun: List[Callable]
    var_link_fun: List[Callable]
    fe_gprior: np.ndarray
    re_gprior: np.ndarray
    param_gprior_info: Tuple[Callable, List[float], List[float]] = None


class Model:

    def __init__(self, param_set, curve_fun, loss_fun):

        self.param_set = param_set
        self.curve_fun = curve_fun
        self.loss_fun = loss_fun
        self.data_inputs = None

    def detach_data(self):
        self.data_inputs = None

    def convert_inputs(self, data):
        df = data[0]
        data_specs = data[1]

        t = df[data_specs.col_t].to_numpy()
        obs = df[data_specs.col_obs].to_numpy()
        obs_se = df[data_specs.col_obs_se].to_numpy()

        covs_mat = []
        for covs in self.param_set.covariate:
            covs_mat.append(df[covs].to_numpy())

        group_sizes = [df.shape[0]]

        var_link_fun = reduce(iconcat, self.param_set.var_link_fun, [])

        fe_gprior = np.array(reduce(iconcat, self.param_set.fe_gprior, []))
        assert fe_gprior.shape == (self.param_set.num_fe, 2)

        re_gprior = []
        for priors in self.param_set.re_gprior:
            for prior in priors:
                re_gprior.append([prior])
        re_gprior = np.array(re_gprior)
        assert re_gprior.shape == (self.param_set.num_fe, 1, 2)

        param_gprior = [[], [], []]
        if self.param_set.parameter_functions is not None:
            for fun in self.param_set.parameter_functions:
                param_gprior[0].append(fun[0])
                param_gprior[1].append(fun[1][0])
                param_gprior[2].append(fun[1][1])

            def param_gprior_fun(p):
                return [f(p) for f in param_gprior[0]]

            param_fun = (param_gprior_fun, param_gprior[1], param_gprior[2])
        else:
            param_fun = None

        self.data_inputs = DataInputs(
            t=t,
            obs=obs,
            obs_se=obs_se,
            covariates_matrices=covs_mat,
            group_sizes=group_sizes,
            link_fun=self.param_set.link_fun,
            var_link_fun=var_link_fun,
            fe_gprior=fe_gprior,
            re_gprior=re_gprior,
            param_gprior_info=param_fun,
        )

    def objective(self, x, data):
        if self.data_inputs is None:
            self.convert_inputs(data)
        return objective_fun(
            x=x,
            t=self.data_inputs.t,
            obs=self.data_inputs.obs,
            obs_se=self.data_inputs.obs_se,
            covs=self.data_inputs.covariates_matrices,
            group_sizes=self.data_inputs.group_sizes,
            model_fun=self.curve_fun,
            loss_fun=self.loss_fun,
            link_fun=self.data_inputs.link_fun,
            var_link_fun=self.data_inputs.var_link_fun,
            fe_gprior=self.data_inputs.fe_gprior,
            re_gprior=self.data_inputs.re_gprior,
            param_gprior=self.data_inputs.param_gprior_info,
        ) 

    @property
    def bounds(self):
        all_bounds = []

        fe_bounds = np.array(reduce(iconcat, self.param_set.fe_bounds, []))
        re_bounds = np.array(reduce(iconcat, self.param_set.re_bounds, []))

        for fb, rb in zip(fe_bounds, re_bounds):
            all_bounds.append(fb)
            all_bounds.append(rb)

        return np.array(all_bounds)
