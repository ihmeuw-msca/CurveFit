# -*- coding: utf-8 -*-
"""
Alpha Prior Model Pipeline
"""
import numpy as np
from curvefit.pipelines.basic_model import BasicModel
from curvefit.core.model import CurveModel
from curvefit.core.utils import *
from curvefit.core.functions import *
from copy import deepcopy
import matplotlib.pyplot as plt


class APModel(BasicModel):
    """
    Alpha prior model.
    """

    def __init__(self, obs_bounds=None,
                 prior_modifier=None,
                 peaked_groups=None,
                 joint_model_fit_dict=None,
                 **kwargs):

        self.obs_bounds = [-np.inf, np.inf] if obs_bounds is None else obs_bounds
        self.fun_gprior = None
        self.models = {}
        self.prior_modifier = prior_modifier
        if self.prior_modifier is None:
            self.prior_modifier = lambda x: 10 ** (min(0.0, max(-2.0, 0.3 * x - 3.5)))

        self.peaked_groups = peaked_groups
        self.joint_model_fit_dict = {} if joint_model_fit_dict is None else \
            joint_model_fit_dict

        self.joint_model_fit_dict = {
            **deepcopy(kwargs['fit_dict']),
            **deepcopy(self.joint_model_fit_dict)
        }

        super().__init__(**kwargs)
        self.run_init_model()

    def run_init_model(self):
        # update functional prior
        if 'fun_gprior' not in self.fit_dict or \
                self.fit_dict['fun_gprior'] is None:
            groups = self.peaked_groups if self.peaked_groups is not None else \
                self.groups
            models = self.run_models(self.all_data, groups)
            self.fun_gprior = self.get_ln_alpha_beta_prior(models)
            print('create log-alpha-beta prior', self.fun_gprior[1])
            self.fit_dict.update({
                'fun_gprior': self.fun_gprior
            })

        if self.peaked_groups is not None:
            model = self.run_joint_model(self.all_data, self.peaked_groups)
            beta_fe_gprior = self.get_beta_fe_gprior(model)
            print('update beta fe gprior to', beta_fe_gprior)
            if 'fe_gprior' in self.fit_dict:
                fe_gprior = self.fit_dict['fe_gprior']
            else:
                fe_gprior = [[0.0, np.inf]] * model.num_fe
            fe_gprior[1] = beta_fe_gprior
            self.fit_dict.update({
                'fe_gprior': deepcopy(fe_gprior)
            })

    @staticmethod
    def get_ln_alpha_beta_prior(models):
        a = np.array([model.params[0, 0]
                      for group, model in models.items()])
        b = np.array([model.params[1, 0]
                      for group, model in models.items()])
        prior_mean = np.log(a * b).mean()
        prior_std = np.log(a * b).std()

        return [lambda params: np.log(params[0] * params[1]),
                [prior_mean, prior_std]]

    @staticmethod
    def get_beta_fe_gprior(model):
        fe, re = model.unzip_x(model.result.x)
        beta_fe_mean = fe[1]
        beta_fe_std = np.std(re[:, 1])

        return [beta_fe_mean, beta_fe_std]

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
        print(group)
        print('\t update beta fe_gprior to', fe_gprior)

        fit_dict.update({
            'fe_gprior': fe_gprior
        })
        model.fit_params(**fit_dict)
        return model

    def run_models(self, df, groups):
        models = {}
        for group in groups:
            models.update({
                group: self.run_model(df, group)
            })

        return models

    def run_joint_model(self, df, groups):
        model = CurveModel(
            df=df[df[self.col_group].isin(groups)].copy(),
            **self.basic_model_dict
        )
        model.fit_params(**self.joint_model_fit_dict)

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
        if 'fun_gprior' not in self.fit_dict or self.fit_dict['fun_gprior'] is None:
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

    def plot_result(self, t):
        models = self.models
        fig, ax = plt.subplots(len(models), 2, figsize=(8 * 2, 4 * len(models)))
        for i, (location, model) in enumerate(models.items()):
            y = model.fun(t, model.params[:, 0])
            ax[i, 0].scatter(model.t, model.obs)
            ax[i, 0].plot(t, y)
            ax[i, 0].set_title(location)

            dy = gaussian_pdf(t, model.params[:, 0])
            gaussian_pdf_obs = data_translator(
                model.obs, self.basic_model_dict['fun'], 'gaussian_pdf'
            )
            ax[i, 1].scatter(model.t, gaussian_pdf_obs)
            ax[i, 1].plot(t, dy)
            ax[i, 1].set_title(location)
            ax[i, 1].set_ylim(0.0, max(dy.max(), gaussian_pdf_obs.max()) * 1.1)

    def summarize_result(self):
        models = self.models
        df_summary = pd.DataFrame({}, columns=['Location', 'RMSE ERF', 'RMSE DERF'])

        location_list = []
        rmse_gaussian_cdf_list = []
        rmse_gaussian_pdf_list = []

        for i, (location, model) in enumerate(models.items()):
            gaussian_cdf_pred = model.fun(model.t, model.params[:, 0])
            rmse_gaussian_cdf = np.linalg.norm(gaussian_cdf_pred - model.obs) ** 2
            gaussian_pdf_obs = data_translator(model.obs, self.basic_model_dict['fun'], 'gaussian_pdf')
            gaussian_pdf_pred = gaussian_pdf(model.t, model.params[:, 0])
            rmse_gaussian_pdf = np.linalg.norm(gaussian_pdf_obs - gaussian_pdf_pred) ** 2

            location_list.append(location)
            rmse_gaussian_cdf_list.append(rmse_gaussian_cdf)
            rmse_gaussian_pdf_list.append(rmse_gaussian_pdf)

        df_summary['Location'] = location_list
        df_summary['RMSE ERF'] = rmse_gaussian_cdf_list
        df_summary['RMSE DERF'] = rmse_gaussian_pdf_list

        return df_summary

    def create_overall_draws(self, t, models, covs,
                             alpha_times_beta=None,
                             sample_size=100,
                             slope_at=14,
                             epsilon=1e-2):
        """Create draws from given models.

        Args:
            t (np.ndarray):
                Time points for the draws.
            models (dict{str, CurveModel}):
                Curve fit models.
            covs (np.ndarray):
                Covariates for the group want have the draws.
            alpha_times_beta (float | None, optional):
                If alpha_times_beta is `None` use the empirical distribution
                for alpha samples, otherwise use the relation from beta to get
                alpha samples.
            sample_size (int, optional):
                Number of samples
            slope_at (int | float, optional):
                If return slopes samples, this is where to evaluation the slope.
            epsilon (float, optional):
                Floor of CV.

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
                    fe_samples[i] * covs[i]
                )
        else:
            beta_samples = link_fun[1](
                var_link_fun[1](samples['beta_fe']) * covs[1])
            alpha_samples = alpha_times_beta / beta_samples
            param_samples = np.vstack([alpha_samples, beta_samples])

        # print(param_samples)

        alpha = np.median(param_samples[0])
        beta = max(slope_at + 1.0, np.median(param_samples[1]))
        slope = np.median(samples['slope'])

        p = solve_p_from_dgaussian_pdf(
            np.array([alpha]), np.array([beta]), np.array([slope]),
            slope_at=slope_at
        )[0]

        params = np.array([alpha, beta, p])

        # create mean curve
        mean_curve = self.predict_space(t, params)

        # create draws for the residual
        error = self.forecaster.create_residual_samples(
            sample_size, t, 1, epsilon
        )

        return mean_curve - (mean_curve ** self.theta) * error - np.var(error, axis=0) * 0.5

    @staticmethod
    def create_param_samples(models, params,
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
                dgaussian_pdf(slope_at, model.params[:, 0])
                for group, model in models.items()
            ])
            ln_slope = np.log(slope)
            samples.update({
                'slope': np.exp(sample_from_samples(ln_slope, sample_size))
            })

        return samples

    def process_draws(self, t):
        """Process draws.
        """
        draws = {}
        for group, draw in self.draws.items():
            truncated_draws = truncate_draws(
                t=t,
                draws=self.draws[group],
                draw_space=self.predict_space,
                last_day=self.models[group].t[-1],
                last_obs=self.models[group].obs[-1],
                last_obs_space=self.fun
            )
            truncated_time = t[int(np.round(self.models[group].t[-1])) + 1:]
            draws.update({
                group: (truncated_time, truncated_draws)
            })

        return draws
