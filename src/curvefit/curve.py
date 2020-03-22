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
                 col_group,
                 fun,
                 param_names,
                 col_obs_se=None):
        """Constructor function of LogisticCurveModel.
        """
        # input data
        self.df = df.copy()
        self.col_obs = col_obs
        self.col_t = col_t
        self.col_group = col_group
        self.fun = fun
        self.param_names = np.array(param_names)
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


    def objective(self, params):
        """Objective function.

        Args:
            params (numpy.ndarray):
                Model parameters.

        Returns:
            float:
                Objective value.
        """
        params = params.reshape((self.num_groups, self.num_params))
        for name in self.param_shared:
            params[:, self.param_idx[name]] = params[0, self.param_idx[name]]
        y = [self.fun(self.t[self.group_idx[name]],
                      params[i])
             for i, name in enumerate(self.group_names)]
        residual = np.hstack([
            (self.obs[self.group_idx[name]] -
             y[i])/self.obs_se[self.group_idx[name]]
            for i, name in enumerate(self.group_names)
        ])
        return 0.5*sum(residual**2)

    def gradient(self, params, eps=1e-16):
        """Gradient function.

        Args:
            params (numpy.ndarray):
                Model parameters.
            eps (float, optional):
                Tolerance for automatic differentiation.

        Returns:
            numpy.ndarray:
                Gradient w.r.t. the model parameters.
        """
        params_c = params + 0j
        grad = np.zeros(params.size)
        for i in range(params.size):
            params_c[i] += eps*1j
            grad[i] = self.objective(params_c).imag/eps
            params_c[i] -= eps*1j

        return grad

    def fit_params(self,
                   param_init,
                   param_fixed=[],
                   param_shared=[],
                   param_bounds={},
                   options={}):
        """Fit the parameters.

        Args:
            param_init (dict{str, float}):
                Initial value for the model parameters.
            param_fixed (list{str}, optional):
                A list of parameter names that will be fixed at initial value.
            param_bounds (dict{str, list{float, float}}, optional):
                Bounds for each model parameter.
            options (dict, optional):
                Options for the optimizer.
        """
        self.param_shared = param_shared.copy()
        # convert information to optimizer
        x0 = np.zeros((self.num_groups, self.num_params))
        for param in param_init:
            x0[:, self.param_idx[param]] = param_init[param]
        x0 = x0.ravel()

        bounds = np.array([[[-np.inf, np.inf]]*self.num_params]*self.num_groups)
        for param in param_bounds:
            bounds[:, self.param_idx[param]] = param_bounds[param]
        for param in param_fixed:
            bounds[:, self.param_idx[param]] = param_init[param]
        bounds = bounds.reshape(self.num_params*self.num_groups, 2)
        
        self.param_init = param_init
        self.param_bounds = param_bounds
        self.param_fixed = param_fixed
        self.options = options

        result = minimize(fun=self.objective,
                  x0=x0,
                  jac=self.gradient,
                  method='L-BFGS-B',
                  bounds=bounds,
                  options=options)

        self.result = result
        self.params = result.x.reshape(self.num_groups, self.num_params)

    def predict(self, t, group_name='all', agg_fun=np.mean):
        if group_name == 'all':
            params = agg_fun(self.params, axis=0)
        else:
            idx = np.where(self.group_names == group_name)[0]
            params = self.params[idx][0]

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
        y = [self.fun(self.t[self.group_idx[name]], self.params[i])
             for i, name in enumerate(self.group_names)]
        residual = [
            self.obs[self.group_idx[name]] - y[i]
            for i, name in enumerate(self.group_names)
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

    @classmethod
    def sample_soln(cls, model,
                    radius=3.0,
                    se_floor=0.5,
                    sample_size=1):
        """Sample solution using fit-refit

        Args:
            model (CurveModel):
                Model subject.
            radius (float, optional):
                Radius variable for the estimate_obs_se.
            sample_size(int, optional):
                Sample size of the solution.

        Returns:
            numpy.ndarray:
                Solution samples
        """
        assert model.result is not None, 'Please fit the model'

        obs_se = model.estimate_obs_se(radius=radius,
                                       se_floor=se_floor)
        params_samples = []
        for i in range(sample_size):
            model_copy = deepcopy(model)
            model_copy.obs = model.predict(model.t) + \
                np.random.randn(model.num_obs)*obs_se
            model_copy.obs_se = obs_se
            model_copy.fit_params(
                param_init=model.param_init,
                param_bounds=model.param_bounds,
                param_fixed=model.param_fixed,
                options=model.options
            )
            params_samples.append(model_copy.params)

        return np.vstack(params_samples)
