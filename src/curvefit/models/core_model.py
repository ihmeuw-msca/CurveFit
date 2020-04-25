from functools import reduce
from operator import iconcat
from dataclasses import dataclass
from typing import List, Callable, Tuple
import numpy as np

from curvefit.core.objective_fun import objective_fun
from curvefit.core.effects2params import effects2params


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
    covariates_matrices: List[np.ndarray]
    group_sizes: List[int]
    num_groups: int
    link_fun: List[Callable]
    var_link_fun: List[Callable]
    x_init: np.ndarray
    bounds: np.ndarray
    fe_gprior: np.ndarray
    re_gprior: np.ndarray
    param_gprior_info: Tuple[Callable, List[float], List[float]] = None


class DataNotFoundError(Exception):
    pass


class CoreModel:

    {end_markdown Model}
    """
    def __init__(self, param_set, curve_fun, loss_fun):

        self.param_set = param_set
        self.curve_fun = curve_fun
        self.loss_fun = loss_fun
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

    def objective(self, x, data):
        if self.data_inputs is None:
            self.data_inputs = convert_inputs(self.param_set, data)
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

    def gradient(self, x, data):
        if self.data_inputs is None:
            self.data_inputs = convert_inputs(self.param_set, data)
        finfo = np.finfo(float)
        step  = finfo.tiny / finfo.eps
        x_c = x + 0j
        grad = np.zeros(x.size)
        for i in range(x.size):
            x_c[i] += step*1j
            grad[i] = self.objective(x_c, data).imag/step
            x_c[i] -= step*1j

        return grad

    def forward(self, x, t):
        params = effects2params(
            x,
            self.data_inputs.group_sizes,
            self.data_inputs.covariates_matrices,
            self.param_set.link_fun,
            self.data_inputs.var_link_fun,
        )
        return self.curve_fun(t, params[:, 0])

    @property
    def bounds(self):
        return self.data_inputs.bounds

    @property
    def x_init(self):
        return self.data_inputs.x_init


def convert_inputs(param_set, data):
    if isinstance(data, DataInputs):
        return data

    df = data[0]
    data_specs = data[1]

    t = df[data_specs.col_t].to_numpy()
    obs = df[data_specs.col_obs].to_numpy()
    obs_se = df[data_specs.col_obs_se].to_numpy()

    covs_mat = []
    for covs in param_set.covariate:
        covs_mat.append(df[covs].to_numpy())

    group_names = df[data_specs.col_group].unique()
    group_sizes_dict = {
        name: np.sum(df[data_specs.col_group].values == name)
        for name in group_names
    }
    group_sizes = list(group_sizes_dict.values())
    num_groups = len(group_names)

    var_link_fun = reduce(iconcat, param_set.var_link_fun, [])

    fe_init = np.array(reduce(iconcat, param_set.fe_init, []))
    re_init = np.array(reduce(iconcat, param_set.re_init, []))
    re_init = np.repeat(re_init[None, :], num_groups, axis=0).flatten()
    x_init = np.concatenate((fe_init, re_init))

    fe_bounds = np.array(reduce(iconcat, param_set.fe_bounds, []))
    re_bounds = np.array(reduce(iconcat, param_set.re_bounds, []))
    re_bounds = np.repeat(re_bounds[None, :, :], num_groups, axis=0)
    bounds = np.vstack([fe_bounds, re_bounds.reshape(param_set.num_fe * num_groups , 2)])

    fe_gprior = np.array(reduce(iconcat, param_set.fe_gprior, []))
    assert fe_gprior.shape == (param_set.num_fe, 2)

    re_gprior = []
    for priors in param_set.re_gprior:
        for prior in priors:
            re_gprior.append([prior] * num_groups)
    re_gprior = np.array(re_gprior)
    assert re_gprior.shape == (param_set.num_fe, num_groups, 2)

    param_gprior_funs = []
    param_gprior_means = []
    param_gprior_stds = []
    if param_set.parameter_functions is not None:
        for fun_prior in param_set.parameter_functions:
            param_gprior_funs.append(fun_prior[0])
            param_gprior_means.append(fun_prior[1][0])
            param_gprior_stds.append(fun_prior[1][1])

        def param_gprior_fun(p):
            return [f(p) for f in param_gprior_funs[0]]

        param_gprior_info = (param_gprior_fun, param_gprior_means, param_gprior_stds)
    else:
        param_gprior_info = None

    return DataInputs(
        t=t,
        obs=obs,
        obs_se=obs_se,
        covariates_matrices=covs_mat,
        group_sizes=group_sizes,
        num_groups=num_groups,
        link_fun=param_set.link_fun,
        var_link_fun=var_link_fun,
        x_init=x_init,
        bounds=bounds,
        fe_gprior=fe_gprior,
        re_gprior=re_gprior,
        param_gprior_info=param_gprior_info,
    )
