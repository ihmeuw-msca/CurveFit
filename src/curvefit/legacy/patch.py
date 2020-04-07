# -*- coding: utf-8 -*-
"""
    patch
    ~~~~~

    Some patch script here for temporary use.
"""
import numpy as np
from curvefit.core.utils import *
from curvefit.core.model import CurveModel


class ModelRunner:
    """Simple model runner.
    """
    def __init__(self, df,
                 col_t,
                 col_obs,
                 col_covs,
                 col_group,
                 link_fun,
                 var_link_fun,
                 fun,
                 col_obs_se=None):
        """Constructor of model runner.
        """
        self.df = df
        self.col_t = col_t
        self.col_obs = col_obs
        self.col_covs = col_covs
        self.col_group = col_group
        self.param_names = ['alpha', 'beta', 'p']
        self.link_fun = link_fun
        self.var_link_fun = var_link_fun
        self.fun = fun
        self.col_obs_se = col_obs_se
        self.groups = self.df[self.col_group].unique()

    def run_model(self, group, **fit_kwargs):
        """Construct and run the model.
        """
        model = CurveModel(
            self.df[self.df[self.col_group] == group].copy(),
            col_t=self.col_t,
            col_obs=self.col_obs,
            col_covs=self.col_covs,
            col_group=self.col_group,
            param_names=self.param_names,
            link_fun=self.link_fun,
            var_link_fun=self.var_link_fun,
            fun=self.fun,
            col_obs_se=self.col_obs_se
        )

        model.fit_params(**fit_kwargs)

        return model

    def run_all_models(self, **fit_kwargs):
        """Run all models, each location independently.
        """
        models = {}
        for group in self.groups:
            models.update({
                group: self.run_model(group, **fit_kwargs)
            })

        return models

    def run_filtered_models(self, obs_bounds,
                        **fit_kwargs):
        """Run all the data rich models.
        """
        models = {}
        for group in self.groups:
            num_obs = np.sum(self.df[self.col_group] == group)
            if num_obs < obs_bounds[0] or num_obs > obs_bounds[1]:
                continue
            models.update({
                group: self.run_model(group, **fit_kwargs)
            })

        return models

    def create_param_samples(self, models, params,
                             sample_size=100,
                             slope_at=14):
        """Create parameter samples from given models.

        Args:
            models (dict{str, CurveModel}):
                Curve fit models.
            params (list{str}):
                Parameter names that we want samples for.
            sample_size (int):
                Number of samples
            slope_at (int | float):
                If return slopes samples, this is where to evaluation the slope.

        Returns:
            dict{str, ndarray}:
                samples for parameters.
        """
        samples = {}
        if 'alpha' in params:
            alpha = np.array([
                model.result.x[0]
                for group, model in models.items()
            ])
            samples.update({
                'alpha_fe': sample_from_samples(alpha, sample_size)
            })

        if 'beta' in params:
            beta = np.array([
                model.result.x[1]
                for group, model in models.items()
            ])
            samples.update({
                'beta_fe': sample_from_samples(beta, sample_size)
            })

        if 'p' in params:
            p = np.array([
                model.result.x[2]
                for group, model in models.items()
            ])
            samples.update({
                'p_fe': sample_from_samples(p, sample_size)
            })

        if 'slope' in params:
            slope = np.array([
                dderf(slope_at, model.params[:, 0])
                for group, model in models.items()
            ])
            log_slope = np.log(slope)
            samples.update({
                'slope': np.exp(sample_from_samples(log_slope, sample_size))
            })

        return samples


    def create_draws(self, t, models, covs, predict_fun,
                     alpha_times_beta=None,
                     sample_size=100,
                     slope_at=14):
        """Create draws from given models.

        Args:
            models (dict{str, CurveModel}):
                Curve fit models.
            params (list{str}):
                Parameter names that we want samples for.
            covs (np.ndarray):
                Covariates for the group want have the draws.
            predict_fun (callable):
                Prediction function.
            alpha_times_beta (float | None, optional):
                If alpha_times_beta is `None` use the empirical distribution
                for alpha samples, otherwise use the relation from beta to get
                alpha samples.
            sample_size (int, optional):
                Number of samples
            slope_at (int | float, optional):
                If return slopes samples, this is where to evaluation the slope.

        Returns:
            np.ndarray:
                Draws, with shape (sample_size, t.size).
        """
        # gathering samples
        samples = self.create_param_samples(models,
                                            ['alpha', 'beta', 'slope'],
                                            sample_size=sample_size,
                                            slope_at=slope_at)
        if alpha_times_beta is None:
            fe_samples = np.vstack([
                samples['alpha_fe'],
                samples['beta_fe']
            ])

            for i in range(2):
                fe_samples[i] = self.var_link_fun[i](fe_samples[i])

            param_samples = np.zeros(fe_samples.shape)
            for i in range(2):
                param_samples[i] = self.link_fun[i](
                    fe_samples[i]*covs[i]
                )
        else:
            beta_samples = self.link_fun[1](
                self.var_link_fun[1](samples['beta_fe'])*covs[1])
            alpha_samples = alpha_times_beta/beta_samples
            param_samples = np.vstack([alpha_samples, beta_samples])


        p_samples = solve_p_from_dderf(param_samples[0],
                                       param_samples[1],
                                       samples['slope'],
                                       slope_at=slope_at)

        param_samples = np.vstack([
            param_samples,
            p_samples
        ])

        draws = []
        for i in range(sample_size):
            draws.append(
                predict_fun(t, param_samples[:, i])
            )

        return np.vstack(draws)
