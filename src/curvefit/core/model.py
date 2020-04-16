# -*- coding: utf-8 -*-
"""
    Logistic Curve Fitting
"""
from copy import deepcopy
import numpy as np
from scipy.optimize import minimize
from curvefit.core import utils

from curvefit.core.utils import get_initial_params
from curvefit.core.utils import compute_starting_params
from curvefit.core.functions import normal_loss
from curvefit.core.effects2params import effects2params
from curvefit.core.objective_fun import objective_fun


class CurveModel:
    """Curve Fitting Class
    """
    def __init__(self, df,
                 col_t,
                 col_obs,
                 col_covs,
                 col_group,
                 param_names,
                 link_fun,
                 var_link_fun,
                 fun,
                 col_obs_se=None,
                 loss_fun=None,
                 scale_obs_se=True):
        """Constructor function of LogisticCurveModel.

        Args:
            df (pandas.DataFrame):
                Data frame that contains all the information.
            col_t (str):
                The column name in the data frame that contains independent
                variable.
            col_obs (str):
                The column name in the data frame that contains dependent
                variable.
            col_covs (list{list{str}}):
                List of list of column name in the data frame used as
                covariates. The outer list len should be number of parameters.
            col_group (str):
                The column name in the data frame that contains the grouping
                information.
            param_names (list{str}):
                Names of the parameters in the specific functional form.
            link_fun (list{function}):
                List of link functions for each parameter.
            var_link_fun (list{function}):
                List of link functions for the variables including fixed effects
                and random effects.
            fun (callable):
                Specific functional form that the curve will fit to.
            col_obs_se (str | None, optional):
                Column name of the observation standard error. When `None`,
                assume all the observation standard error to be all one.
            loss_fun(callable | None, optional):
                Loss function, if None, use Gaussian distribution.
            scale_obs_se (bool, optional):
                If scale the observation standard deviation by the absolute mean
                of the observations.
        """
        # input data
        self.df = df.copy()
        self.col_t = col_t
        self.col_obs = col_obs
        self.col_covs = col_covs
        self.col_group = col_group
        self.param_names = np.array(param_names)
        self.link_fun = link_fun
        self.var_link_fun = var_link_fun
        self.fun = fun
        self.loss_fun = normal_loss if loss_fun is None else loss_fun
        self.col_obs_se = col_obs_se

        self.group_names = np.sort(self.df[self.col_group].unique())

        # dimensions
        self.num_obs = self.df.shape[0]
        self.num_params = self.param_names.size
        self.num_groups = self.group_names.size

        # sort the dataframe by group
        self.df.sort_values([self.col_group, self.col_t], inplace=True)

        # extracting information
        self.obs = self.df[self.col_obs].values
        self.obs_se = np.ones(self.num_obs) if self.col_obs_se is None else \
            self.df[col_obs_se].values

        self.scale_obs_se = scale_obs_se
        if self.scale_obs_se:
            self.obs_se *= np.abs(self.obs).mean()/self.obs_se.mean()

        self.t = self.df[self.col_t].values
        self.group = self.df[self.col_group].values
        self.covs = [
            df[name].values
            for name in self.col_covs
        ]
        self.fe_sizes = np.array([
            cov.shape[1]
            for cov in self.covs
        ])
        self.fe_idx = utils.sizes_to_indices(self.fe_sizes)
        self.num_fe = self.fe_sizes.sum()
        self.num_re = self.num_groups*self.num_fe

        # parameter information
        self.param_idx = {
            name: i
            for i, name in enumerate(self.param_names)
        }

        # group information
        self.group_sizes = {
            name: np.sum(self.group == name)
            for name in self.group_names
        }
        self.order_group_sizes = np.array([
            self.group_sizes[name]
            for name in self.group_names
        ])
        self.order_group_idx = np.cumsum(self.order_group_sizes) - 1
        group_idx = utils.sizes_to_indices(np.array([
            self.group_sizes[name]
            for name in self.group_names
        ]))
        self.group_idx = {
            name: group_idx[i]
            for i, name in enumerate(self.group_names)
        }

        # place holder
        self.param_shared = []
        self.result = None
        self.params = None
        self.fe_gprior = np.array([[0.0, np.inf]]*self.num_fe)
        self.re_gprior = np.array([[0.0, np.inf]]*self.num_fe)
        self.fun_gprior = None

    def unzip_x(self, x):
        """Unzip raw input to fixed effects and random effects.

        Args:
            x (numpy.ndarray):
                Array contains all the fixed and random effects.

        Returns:
            fe (numpy.ndarray): fixed effects.
            re (numpy.ndarray): random effects.
        """
        fe = x[:self.num_fe]
        re = x[self.num_fe:].reshape(self.num_groups, self.num_fe)
        return fe, re

    def compute_params(self, x, expand=True):
        """Compute parameters from raw vector.

        Args:
            x (numpy.ndarray):
                Array contains all the fixed and random effects.
            expand (bool, optional):
                If `expand` is `True`, then create parameters for every
                observation, else only create parameters for each group.

        Returns:
            params (numpy.ndarray):
                Array of parameters for the curve functional form, with shape
                (num_params, num_obs) or (num_params, num_groups).
        """
        # 2DO: This function is no longer being used and should be removed
        assert False
        #
        fe, re = self.unzip_x(x)
        covs = self.covs
        if expand:
            re = np.repeat(re, self.order_group_sizes, axis=0)
        else:
            covs = [
                self.covs[i][self.order_group_idx, :]
                for i in range(len(self.covs))
            ]
        var = fe + re
        for i in range(self.num_fe):
            var[:, i] = self.var_link_fun[i](var[:, i])
        params = np.vstack([
            np.sum(cov*var[:, self.fe_idx[i]], axis=1)
            for i, cov in enumerate(covs)
        ])

        for i in range(self.num_params):
            params[i] = self.link_fun[i](params[i])

        return params

    def objective(self, x) :
        return objective_fun(
            x,
            self.t,
            self.obs,
            self.obs_se,
            self.covs,
            self.order_group_sizes,
            self.fun,
            self.loss_fun,
            self.link_fun,
            self.var_link_fun,
            self.fe_gprior,
            self.re_gprior,
            self.fun_gprior
        )

    def gradient(self, x, eps=1e-16):
        """Gradient function.

        Args:
            x (numpy.ndarray):
                Model parameters.
            eps (float, optional):
                Tolerance for automatic differentiation.

        Returns:
            numpy.ndarray:
                Gradient w.r.t. the model parameters.
        """
        finfo = np.finfo(float)
        step  = finfo.tiny / finfo.eps
        x_c = x + 0j
        grad = np.zeros(x.size)
        for i in range(x.size):
            x_c[i] += step*1j
            grad[i] = self.objective(x_c).imag/step
            x_c[i] -= step*1j

        return grad

    def fit_params(self,
                   fe_init,
                   re_init=None,
                   fe_bounds=None,
                   re_bounds=None,
                   fe_gprior=None,
                   re_gprior=None,
                   fun_gprior=None,
                   fixed_params=None,
                   smart_initialize=False,
                   fixed_params_initialize=None,
                   options=None,
                   smart_init_options=None):
        """Fit the parameters.

        Args:
            fe_init (numpy.ndarray):
                Initial value for the fixed effects.
            re_init (numpy.ndarray, optional):
                Initial value for the random effects.
            fe_bounds (list of lists, optional):
                Bounds for fixed effects.
            re_bounds (list of lists, optional):
                Bounds for random effects.
            fe_gprior (list of lists, optional):
                Gaussian prior for fixed effects.
            re_gprior (list of lists, optional):
                Gaussian prior for random effects.
            fun_gprior (list of lists, optional):
                Functional Gaussian prior.
            fixed_params (list{str}, optional):
                A list of parameter names that will be fixed at initial value.
            smart_initialize (bool, optional):
                Whether or not to initialize a model's fixed effects based
                on the average fixed effects across many individual models
                fit with the same settings and the random effects
                based on the fixed effects deviation from the average
                in the individual models
            fixed_params_initialize (list{str}, optional):
                A list of parameter names that will be fixed at initial value during the smart initialization.
                Will be ignored if smart_initialize = False and raise warning.
            options (dict, optional):
                Options for the optimizer.
            smart_init_options (dict, optional):
                Options for the inner model
        """
        assert len(fe_init) == self.num_fe
        if fe_bounds is None:
            fe_bounds = [[-np.inf, np.inf]]*self.num_fe
        if re_bounds is None:
            re_bounds = [[-np.inf, np.inf]]*self.num_fe
        assert len(fe_bounds) == self.num_fe
        assert len(re_bounds) == self.num_fe

        if fe_gprior is not None:
            assert len(fe_gprior) == self.num_fe
            self.fe_gprior = np.array(fe_gprior)
        if re_gprior is not None:
            assert len(re_gprior) == self.num_fe
            self.re_gprior = np.array(re_gprior)
        if re_init is None:
            re_init = np.zeros(self.num_re)

        if fun_gprior is not None:
            assert len(fun_gprior) == 2
            assert fun_gprior[1][1] > 0.0

        self.fun_gprior = fun_gprior

        if fixed_params_initialize is not None:
            if not smart_initialize:
                raise Warning(f"You passed in an initialization parameter "
                              f"fixed_params_initialize {fixed_params_initialize} "
                              f"but set smart_initialize=False. Will ignore fixed_params_initialize.")

        if smart_init_options is not None:
            if options is None:
                raise RuntimeError("Need to pass in options if you pass in smart init options.")

        if smart_initialize:
            smart_initialize_options = deepcopy(options)
            if smart_init_options is not None:
                smart_initialize_options.update(smart_init_options)
            if self.num_groups == 1:
                raise RuntimeError("Don't do initialization for models with only one group.")

            fe_dict = get_initial_params(
                groups=self.group_names,
                model=self,
                fit_arg_dict=dict(
                    fe_init=fe_init,
                    fe_bounds=fe_bounds,
                    fe_gprior=fe_gprior,
                    fixed_params=fixed_params_initialize,
                    options=smart_initialize_options,
                )
            )
            fe_init, re_init = compute_starting_params(fe_dict)
            print(f"Overriding fe_init with {fe_init}.")
            print(f"Overriding re_init with {re_init}.")

        x0 = np.hstack([fe_init, re_init])
        if fe_bounds is None:
            fe_bounds = np.array([[-np.inf, np.inf]]*self.num_fe)
        if re_bounds is None:
            re_bounds = np.array([[-np.inf, np.inf]]*self.num_fe)

        fe_bounds = np.array(fe_bounds)
        re_bounds = np.array(re_bounds)

        if fixed_params is not None:
            for param in fixed_params:
                param_id = self.param_idx[param]
                fe_bounds[param_id] = x0[param_id, None]
                re_bounds[param_id] = 0.0

        re_bounds = np.repeat(re_bounds[None, :, :], self.num_groups, axis=0)
        bounds = np.vstack([fe_bounds,
                            re_bounds.reshape(self.num_re, 2)])

        result = minimize(
            fun=self.objective,
            x0=x0,
            jac=self.gradient,
            method='L-BFGS-B',
            bounds=bounds,
            options=options
        )

        self.result = result
        self.params = effects2params(
            self.result.x,
            self.order_group_sizes,
            self.covs,
            self.link_fun,
            self.var_link_fun,
            expand=False
        )

    def compute_rmse(self, x=None, use_obs_se=True):
        """Compute the Root Mean Squre Error.

        Args:
            x (numpy.ndarray | None, optional):
                Provided solution array, if None use the object solution.
            use_obs_se (bool, optional):
                If True include the observation standard deviation into the
                calculation.

        Returns:
            float: root mean square error.
        """
        if x is None:
            assert self.result is not None
            x = self.result.x

        params = effects2params(
            x,
            self.order_group_sizes,
            self.covs,
            self.link_fun,
            self.var_link_fun
        )
        residual = self.obs - self.fun(self.t, params)

        if use_obs_se:
            return np.sqrt(np.sum(residual**2/self.obs_se**2)/
                           np.sum(1.0/self.obs_se**2))
        else:
            return np.sqrt(np.mean(residual**2))

    def predict(self, t, group_name='all', prediction_functional_form=None):
        """Predict the observation by given independent variable and group name.

        Args:
            t (numpy.ndarray):
                Array of independent variable.
            group_name (dtype(group_names) | str, optional):
                If all will produce average curve and if specific group name
                will produce curve for the group.
            prediction_functional_form (function):
                One of the functions from curvefit.functions
                Needs to have the same parameters as self.fun

        Returns:
            numpy.ndarray:
                Array record the curve.
        """
        if group_name == 'all':
            params = self.params.mean(axis=1)
        else:
            params = self.params[:, np.where(self.group_names == group_name)[0][0]]

        if prediction_functional_form is None:
            fun = self.fun
        else:
            fun = prediction_functional_form

        return fun(t, params)

    def estimate_obs_se(self, radius=3.0, se_floor=0.5):
        """Estimate the observation standard error.

        Args:
            radius (float, optional):
                Radius group to estimate standard error.
            se_floor (float, optional):
                When the standard error is low use this instead.

        Returns:
            numpy.ndarray:
                Vector that contains all the standard error for each
                observation.
        """
        residual = self.obs - self.fun(self.t, self.params)
        residual = [
            residual[self.group_idx[name]]
            for name in self.group_names
        ]
        obs_se = []
        for j, name in enumerate(self.group_names):
            sub_residual = residual[j]
            sub_t = self.t[self.group_idx[name]]
            sub_obs_se = np.zeros(self.group_sizes[name])
            for i in range(self.group_sizes[name]):
                lb = max(0, sub_t[i] - radius)
                ub = min(sub_t.max(), sub_t[i] + radius)
                lb_idx = np.arange(self.group_sizes[name])[sub_t >= lb][0]
                ub_idx = np.arange(self.group_sizes[name])[sub_t <= ub][-1]
                sub_obs_se[i] = max(se_floor,
                                    np.std(sub_residual[lb_idx:ub_idx]))
            obs_se.append(sub_obs_se)
        return np.hstack(obs_se)

    def get_self_model_kwargs(self):
        """
        Gets keyword arguments for a CurveModel
        based on this instance of the CurveModel

        Returns:
            (dict) kwargs for model from self
        """
        return dict(
            col_t=self.col_t,
            col_obs=self.col_obs,
            col_covs=self.col_covs,
            col_obs_se=self.col_obs_se,
            col_group=self.col_group,
            param_names=self.param_names,
            link_fun=self.link_fun,
            var_link_fun=self.var_link_fun,
            fun=self.fun
        )

    def run_one_group_model(self, group, **fit_kwargs):
        """
        Run the exact model as self but instantiate it as a new model
        so that we can run it on a subset of the data defined by the group (no random effects).
        Used for smart initialization of fixed and random effects.

        Args:
            group: (str) the random effect group to include
            in this model.

        Returns:
            np.array of fixed effects for this single group
        """
        df_sub = self.df.loc[self.df[self.col_group] == group].copy()
        model_kwargs = self.get_self_model_kwargs()
        model = CurveModel(df=df_sub, **model_kwargs)
        fit_dict = deepcopy(fit_kwargs)
        fit_dict.update(dict(
            re_bounds=[[0.0, 0.0] for i in range(model.num_fe)],
            smart_initialize=False
        ))
        model.fit_params(**fit_dict)
        return model.result.x[:self.num_fe]
