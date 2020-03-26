# -*- coding: utf-8 -*-
"""
    Logistic Curve Fitting
"""
import numpy as np
from copy import deepcopy
from scipy.optimize import minimize
from scipy.optimize import fmin_l_bfgs_b
from . import utils


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
                 col_obs_se=None):
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
            fun (function):
                Specific functional form that the curve will fit to.
            col_obs_se (str | None, optional):
                Column name of the observation standard error. When `None`,
                assume all the observation standard error to be all one.
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
            self.df[self.col_obs_se].values
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
        group_idx = utils.sizes_to_indices([
            self.group_sizes[name]
            for name in self.group_names
        ])
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

    def unzip_x(self, x):
        """Unzip raw input to fixed effects and random effects.
        """
        fe = x[:self.num_fe]
        re = x[self.num_fe:].reshape(self.num_groups, self.num_fe)
        return fe, re

    def compute_params(self, x, expand=True):
        """Compute parameters from raw vector.
        """
        fe, re = self.unzip_x(x)
        covs = self.covs
        if expand:
            re = np.repeat(re, self.order_group_sizes, axis=0)
        else:
            covs = [
                self.covs[i][self.order_group_idx, :]
                for i in range(len(self.covs))
            ]
        params = np.vstack([
            np.sum(cov*self.var_link_fun[i](
                fe[self.fe_idx[i]] + re[:, self.fe_idx[i]]
            ), axis=1)
            for i, cov in enumerate(covs)
        ])

        for i in range(self.num_params):
            params[i] = self.link_fun[i](params[i])

        return params

    def objective(self, x):
        """Objective function.

        Args:
            x (numpy.ndarray):
                Model parameters.

        Returns:
            float:
                Objective value.
        """
        fe, re = self.unzip_x(x)
        params = self.compute_params(x)
        residual = (self.obs - self.fun(self.t, params))/self.obs_se
        val = 0.5*np.sum(residual**2)
        # gprior from fixed effects
        val += 0.5*np.sum(
            (fe - self.fe_gprior.T[0])**2/self.fe_gprior.T[1]**2
        )
        # gprior from random effects
        val += 0.5*np.sum(
            (re - self.re_gprior.T[0])**2/self.re_gprior.T[1]**2
        )
        return val

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
        x_c = x + 0j
        grad = np.zeros(x.size)
        for i in range(x.size):
            x_c[i] += eps*1j
            grad[i] = self.objective(x_c).imag/eps
            x_c[i] -= eps*1j

        return grad

    def fit_params(self,
                   fe_init,
                   re_init=None,
                   fe_bounds=None,
                   re_bounds=None,
                   fe_gprior=None,
                   re_gprior=None,
                   fixed_params=None,
                   **kwargs):
        """Fit the parameters.

        Args:
            fe_init (numpy.ndarray):
                Initial value for the fixed effects.
            fe_bounds (numpy.ndarray, optional):
                Bounds for fixed effects.
            re_bounds (numpy.ndarray, optional):
                Bounds for random effects.
            param_fixed (list{str}, optional):
                A list of parameter names that will be fixed at initial value.
            options (dict, optional):
                Options for the optimizer.
        """
        if fe_gprior is not None:
            self.fe_gprior = np.array(fe_gprior)
        if re_gprior is not None:
            self.re_gprior = np.array(re_gprior)
        if re_init is None:
            re_init = np.zeros(self.num_re)
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
        result = fmin_l_bfgs_b(
            func=self.objective,
            x0=x0,
            fprime=self.gradient,
            bounds=bounds,
            **kwargs
        )

        self.result = result
        self.params = self.compute_params(self.result[0], expand=False)

    def predict(self, t, group_name='all'):
        if group_name == 'all':
            params = self.params.mean(axis=1)
        else:
            params = self.params[:,
                                 np.where(self.group_names==group_name)[0][0]]

        return self.fun(t, params)

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

    # @classmethod
    # def sample_soln(cls, model,
    #                 radius=3.0,
    #                 se_floor=0.5,
    #                 sample_size=1):
    #     """Sample solution using fit-refit
    #
    #     Args:
    #         model (CurveModel):
    #             Model subject.
    #         radius (float, optional):
    #             Radius variable for the estimate_obs_se.
    #         sample_size(int, optional):
    #             Sample size of the solution.
    #
    #     Returns:
    #         numpy.ndarray:
    #             Solution samples
    #     """
    #     assert model.result is not None, 'Please fit the model'
    #
    #     obs_se = model.estimate_obs_se(radius=radius,
    #                                    se_floor=se_floor)
    #     params_samples = []
    #     for i in range(sample_size):
    #         model_copy = deepcopy(model)
    #         model_copy.obs = model.predict(model.t) + \
    #             np.random.randn(model.num_obs)*obs_se
    #         model_copy.obs_se = obs_se
    #         model_copy.fit_params(
    #             param_init=model.param_init,
    #             param_bounds=model.param_bounds,
    #             param_fixed=model.param_fixed,
    #             options=model.options
    #         )
    #         params_samples.append(model_copy.params)
    #
    #     return np.vstack(params_samples)
