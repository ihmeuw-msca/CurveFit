# -*- coding: utf-8 -*-
"""
    test new pipeline
"""
import numpy as np
from .model_generators import BasicModel
from .model import CurveModel
from .utils import *
from copy import deepcopy


class APModel(BasicModel):
    """Alapha prior model.
    """
    def __init__(self, obs_bounds=None,
                 prior_modifier=None, **kwargs):

        self.obs_bounds = [-np.inf, np.inf] if obs_bounds is None else obs_bounds
        self.fun_gprior = None
        self.models = {}
        self.prior_modifier = prior_modifier
        if self.prior_modifier is None:
            self.prior_modifier = lambda x: 10**(min(1.0, max(-2.0,
                0.3*x - 3.5
            )))

        super().__init__(**kwargs)

    def run_init_model(self):
        if 'fun_gprior' not in self.fit_dict or \
            self.fit_dict['fun_gprior'] is None:
            models = self.run_filtered_models(self.all_data,
                                              self.obs_bounds)
            a = np.array([model.params[0, 0]
                          for group, model in models.items()])
            b = np.array([model.params[1, 0]
                          for group, model in models.items()])
            prior_mean = np.log(a*b).mean()
            prior_std = np.log(a*b).std()
            self.fun_gprior = [lambda params: np.log(params[0]*params[1]),
                               [prior_mean, prior_std]]
            self.fit_dict.update({
                'fun_gprior': self.fun_gprior
            })

    def run_model(self, df, group):
        """Run each individual model.
        """
        model = CurveModel(
            df=df[df[self.col_group] == group].copy(),
            **self.basic_model_dict
        )

        fit_dict = deepcopy(self.fit_dict)
        fe_gprior = fit_dict['fe_gprior']
        fe_gprior[1][1] *= self.prior_modifier(model.num_obs)

        fit_dict.update({
            'fe_gprior': fe_gprior
        })
        # print(fit_dict['fe_gprior'])
        model.fit_params(**fit_dict)
        return model

    def run_filtered_models(self, df, obs_bounds):
        """Run filtered models.
        """
        models = {}
        for group in df[self.col_group].unique():
            num_obs = np.sum(df[self.col_group] == group)
            if num_obs < obs_bounds[0] or num_obs > obs_bounds[1]:
                continue
            models.update({
                group: self.run_model(df, group)
            })

        return models

    def fit(self, df, group=None):
        """Fit models by the alpha-beta prior.
        """
        if ('fun_gprior' not in self.fit_dict or
            self.fit_dict['fun_gprior'] is None):
            self.run_init_model()

        if group is None:
            self.models = self.run_filtered_models(df, [-np.inf, np.inf])
        else:
            model = self.run_model(df, group)
            self.models.update({group: model})

    def refresh(self):
        self.models = None

    def predict(self, times, predict_space, predict_group):
        predictions = self.models[predict_group].predict(
            t=times,
            group_name=predict_group,
            prediction_functional_form=predict_space
        )
        return predictions

    def create_overall_draws(self, t, models, covs, predict_fun,
                             alpha_times_beta=None,
                             sample_size=100,
                             slope_at=14):
        """Create draws from given models.

        Args:
            t (np.ndarray):
                Time points for the draws.
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
        var_link_fun = self.basic_model_dict['var_link_fun']
        link_fun = self.basic_model_dict['link_fun']
        if alpha_times_beta is None:
            fe_samples = np.vstack([
                samples['alpha_fe'],
                samples['beta_fe']
            ])

            for i in range(2):
                fe_samples[i] = var_link_fun[i](
                    fe_samples[i]
                )

            param_samples = np.zeros(fe_samples.shape)
            for i in range(2):
                param_samples[i] = link_fun[i](
                    fe_samples[i]*covs[i]
                )
        else:
            beta_samples = link_fun[1](
                var_link_fun[1](samples['beta_fe'])*covs[1])
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



