from functools import reduce
from operator import iconcat
from dataclasses import dataclass
from typing import List, Callable, Tuple
import numpy as np
from curvefit.core.objective_fun import objective_fun

@dataclass
class DataInputs:
    t : np.ndarray 
    obs: np.ndarray 
    obs_se: np.ndarray
    covariates_matrices: List[np.ndarray]
    group_sizes: List[int]
    fe_gprior: np.ndarray
    re_gprior: np.ndarray
    param_gprior_info: Tuple[Callable, List[float], List[float]]


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
        
        fe_gprior = np.array(reduce(iconcat, self.param_set.fe_gprior, []))
        assert fe_gprior.shape == (self.param_set.num_fe, 2)
        
        re_gprior = []
        for priors in self.param_set.re_gprior:
            for prior in priors:
                re_gprior.append([prior])
        re_gprior = np.array(re_gprior)
        assert re_gprior.shape == (self.param_set.num_fe, 1, 2)

        param_gprior = [[], [], []]
        for fun in self.param_set.parameter_functions:
            param_gprior[0].append(fun[0])
            param_gprior[1].append(fun[1][0])
            param_gprior[2].append(fun[1][1])

        def param_gprior_fun(p):
            return [f(p) for f in param_gprior[0]]
        
        param_fun = [param_gprior_fun, param_gprior[1], param_gprior[2]]

        self.data_inputs = DataInputs(
            t=t,
            obs=obs,
            obs_se=obs_se,
            covariates_matrices=covs_mat,
            group_sizes=group_sizes,
            fe_gprior=fe_gprior,
            re_gprior=re_gprior,
            param_gprior_info=param_fun,
        )

    def objective(self, x, data):
        if self.data_inputs is None:
            self.convert_inputs(data)
        return objective_fun(
            x, 
            self.data_inputs.t, 
            self.data_inputs.obs, 
            self.data_inputs.obs_se, 
            self.data_inputs.covariates_matrices, 
            self.data_inputs.group_sizes, 
            self.curve_fun, 
            self.loss_fun,
            self.param_set.link_fun,
            self.param_set.var_link_fun,
            self.data_inputs.fe_gprior,
            self.data_inputs.re_gprior,
            self.data_inputs.param_gprior_info,
        )