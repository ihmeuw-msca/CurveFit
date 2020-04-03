# -*- coding: utf-8 -*-
"""
    test new pipeline
"""
import numpy as np
from .model_generators import BasicModel
from .model import CurveModel


class APModel(BasicModel):
    """Alapha prior model.
    """
    def __init__(self, obs_bounds=None, **kwargs):
        super().__init__(**kwargs)

        self.obs_bounds = [-np.inf, np.inf] if obs_bounds is None else obs_bounds
        self.models = None
        self.fun_gprior = None

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

        model.fit_params(**self.fit_dict)
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
            group_name = predict_group,
            predict_functional_form = predict_space
        )
        return predictions
