"""
All classes of model generators should have a model_function that takes arguments
df and times and returns predictions at those times.
"""

from curvefit.model import CurveModel


class BasicModelGenerator:
    def __init__(self, col_t, col_obs, col_covs, col_group,
                 param_names, link_fun, fit_fun, predict_fun, var_link_fun, predict_group='all',
                 **fit_kwargs):
        """
        Generic class for a function to produce predictions from a model
        with the following attributes.

        Args:
            col_t:
            col_obs:
            col_covs:
            col_group:
            param_names:
            link_fun:
            fit_fun:
            predict_fun:
            var_link_fun:
            predict_group:
            **fit_kwargs: keyword arguments to CurveModel.fit_params()
        """
        self.col_t = col_t
        self.col_obs = col_obs
        self.col_covs = col_covs
        self.col_group = col_group
        self.param_names = param_names
        self.link_fun = link_fun
        self.fit_fun = fit_fun
        self.predict_fun = predict_fun
        self.var_link_fun = var_link_fun
        self.predict_group = predict_group
        self.fit_kwargs = fit_kwargs

    def model_function(self, df, times):
        mod = CurveModel(
            df=df,
            col_t=self.col_t,
            col_obs=self.col_obs,
            col_covs=self.col_covs,
            col_group=self.col_group,
            param_names=self.param_names,
            link_fun=self.link_fun,
            fun=self.fit_fun,
            var_link_fun=self.var_link_fun
        )
        mod.fit_params(**self.fit_kwargs)
        return mod.predict(
            t=times, group_name=self.predict_group,
            prediction_functional_form=self.predict_fun
        )
