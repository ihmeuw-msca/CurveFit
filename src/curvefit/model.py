# -*- coding: utf-8 -*-
"""
    Logistic Curve Fitting
"""
import numpy as np
from copy import deepcopy
from scipy.optimize import minimize
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
                 fun,
                 col_obs_se=None):
        """Constructor function of LogisticCurveModel.
        """
        # input data
        self.df = df.copy()
        self.col_t = col_t
        self.col_obs = col_obs
        self.col_covs = col_covs
        self.col_group = col_group
        self.param_names = np.array(param_names)
        self.link_fun = link_fun
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

        # parameter information
        self.param_idx = {
            name: i
            for i, name in enumerate(self.param_names)
        }

        # group information
        self.group_sizes = {
            name: (self.group == name).sum()
            for name in self.group_names
        }
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
        self.re_var = np.ones(self.num_params)

    def unzip_x(self, x):
        """Unzip raw input to fixed effects and random effects.
        """
        fe = [
            x[self.fe_idx[i]]
            for i in range(self.num_params)
        ]
        re = x[self.fe_sizes.sum():].reshape(self.num_groups, self.num_params)
        return fe, re

    def compute_params(self, x):
        """Compute parameters from raw vector.
        """
        fe, re = self.unzip_x(x)
        params = np.vstack([
            cov.dot(fe[i])
            for i, cov in enumerate(self.covs)
        ]) + np.repeat(re,
                       [self.group_sizes[name]
                        for name in self.group_names], axis=0).T

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
        val = 0.5*np.sum(residual**2) + 0.5*np.sum(re**2/self.re_var)
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
                   re_var=None,
                   fixed_params=None,
                   options=None):
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
        self.re_var = re_var if re_var is not None else self.re_var
        if fe_bounds is None:
            fe_bounds = np.array([[-np.inf, np.inf]]*self.fe_sizes.sum())
        if re_bounds is None:
            re_bounds = np.array([[-np.inf, np.inf]]*self.num_params)
        fe_bounds = np.array(fe_bounds)
        re_bounds = np.array(re_bounds)
        if fixed_params is not None:
            for param in fixed_params:
                param_id = self.param_idx[param]
                fe_bounds[param_id] = x0[param_id, None]
                re_bounds[param_id] = 0.0

        re_bounds = np.repeat(re_bounds[None, :, :], self.num_groups, axis=0)
        bounds = np.vstack([fe_bounds,
                            re_bounds.reshape(
                                self.num_groups*self.num_params, 2)])

        if re_init is None:
            re_init = np.zeros(self.num_groups*self.num_params)
        x0 = np.hstack([fe_init, re_init])

        result = minimize(fun=self.objective,
                  x0=x0,
                  jac=self.gradient,
                  method='L-BFGS-B',
                  bounds=bounds,
                  options=options)

        self.result = result
        self.params = self.compute_params(self.result.x)

    def predict(self, t, group_name):
        params = self.params[:, self.group_idx[group_name]][:, 0]

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
            for name in enumerate(self.group_names)
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
