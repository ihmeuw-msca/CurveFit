from curvefit.core.objective_fun import objective_fun

class Model:

    def __init__(self, param_set, curve_fun, loss_fun):
        self.curve_fun = curve_fun
        self.loss_fun = loss_fun

    def objective(self, x, data):
        df = data[0]
        data_specs = data[1]
        t = df[data_specs.col_t].to_numpy()
        obs = df[data_specs.col_obs].to_numpy()
        obs_se = df[data_specs.col_obs_se].to_numpy()
        covs_mat = []
        for covs in self.param_set.covariate:
            covs_mat.append(df[covs].to_numpy())
        group_sizes = 

        return objective_fun(
            x, 
            t, 
            obs, 
            obs_se, 
            covs_mat, 
            group_sizes, 
            self.curve_fun, 
            self.loss_fun,
            self.param_set.link_fun,
            self.param_set.var_link_fun,
            self.param_set.fe_gprior,
            self.param_set.re_gprior,
            self.param_set.parameter_functions,
        )
