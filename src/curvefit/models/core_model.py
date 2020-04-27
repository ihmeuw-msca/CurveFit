from functools import reduce
from operator import iconcat
import numpy as np

from curvefit.core.objective_fun import objective_fun
from curvefit.core.effects2params import effects2params
from curvefit.models.base import Model, DataInputs


class CoreModel(Model):
    """
    {begin_markdown CoreModel}

    {spell_markdown param}

    # `curvefit.core.core_model.Model`
    ## Base class for a curvefit model

    Add description here.

    ## Arguments

    - `param_set (curvefit.core.parameter.ParameterSet)`
    - `curve_fun (Callable)`: function from `curvefit.core.functions` for the parametric function to fit
    - `loss_fun (Callable)`: function from `curvefit.core.functions` for the loss function

    ## Attributes

    ## Methods

    {end_markdown CoreModel}
    """
    def __init__(self, param_set, curve_fun, loss_fun):
        super().__init__()

        self.param_set = param_set
        self.curve_fun = curve_fun
        self.loss_fun = loss_fun

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

    def get_params(self, x):
        return effects2params(
            x,
            self.data_inputs.group_sizes,
            self.data_inputs.covariates_matrices,
            self.param_set.link_fun,
            self.data_inputs.var_link_fun,
        )

    def predict(self, x, t, predict_fun=None, is_multi_group=False):
        params = self.get_params(x=x)
        if predict_fun is None:
            predict_fun = self.curve_fun
        
        if not is_multi_group:
            return predict_fun(t, params[:, 0])
        else:
            pred = np.zeros((params.shape[1], len(t)))
            for i in range(params.shape[1]):
                pred[i, :] = predict_fun(t, params[:, i])
            return pred

    @property
    def bounds(self):
        return self.data_inputs.bounds

    @property
    def x_init(self):
        return self.data_inputs.x_init

    def convert_inputs(self, data):
        if isinstance(data, DataInputs):
            self.data_inputs = data
            return 

        df = data[0]
        data_specs = data[1]

        t = df[data_specs.col_t].to_numpy()
        obs = df[data_specs.col_obs].to_numpy()
        obs_se = df[data_specs.col_obs_se].to_numpy()

        covs_mat = []
        for covs in self.param_set.covariate:
            covs_mat.append(df[covs].to_numpy())

        group_names = df[data_specs.col_group].unique()
        group_sizes_dict = {
            name: np.sum(df[data_specs.col_group].values == name)
            for name in group_names
        }
        group_sizes = list(group_sizes_dict.values())
        num_groups = len(group_names)

        var_link_fun = reduce(iconcat, self.param_set.var_link_fun, [])

        fe_init = np.array(reduce(iconcat, self.param_set.fe_init, []))
        re_init = np.array(reduce(iconcat, self.param_set.re_init, []))
        re_init = np.repeat(re_init[None, :], num_groups, axis=0).flatten()
        x_init = np.concatenate((fe_init, re_init))

        fe_bounds = np.array(reduce(iconcat, self.param_set.fe_bounds, []))
        re_bounds = np.array(reduce(iconcat, self.param_set.re_bounds, []))
        re_bounds = np.repeat(re_bounds[None, :, :], num_groups, axis=0)
        bounds = np.vstack([fe_bounds, re_bounds.reshape(self.param_set.num_fe * num_groups , 2)])

        fe_gprior = np.array(reduce(iconcat, self.param_set.fe_gprior, []))
        assert fe_gprior.shape == (self.param_set.num_fe, 2)

        re_gprior = []
        for priors in self.param_set.re_gprior:
            for prior in priors:
                re_gprior.append([prior] * num_groups)
        re_gprior = np.array(re_gprior)
        assert re_gprior.shape == (self.param_set.num_fe, num_groups, 2)

        param_gprior_funs = []
        param_gprior_means = []
        param_gprior_stds = []
        if self.param_set.parameter_functions is not None:
            for fun_prior in self.param_set.parameter_functions:
                param_gprior_funs.append(fun_prior[0])
                param_gprior_means.append(fun_prior[1][0])
                param_gprior_stds.append(fun_prior[1][1])

            def param_gprior_fun(p):
                return [f(p) for f in param_gprior_funs[0]]

            param_gprior_info = (param_gprior_fun, param_gprior_means, param_gprior_stds)
        else:
            param_gprior_info = None

        self.data_inputs = DataInputs(
            t=t,
            obs=obs,
            obs_se=obs_se,
            covariates_matrices=covs_mat,
            group_sizes=group_sizes,
            num_groups=num_groups,
            link_fun=self.param_set.link_fun,
            var_link_fun=var_link_fun,
            x_init=x_init,
            bounds=bounds,
            fe_gprior=fe_gprior,
            re_gprior=re_gprior,
            param_gprior_info=param_gprior_info,
        )
