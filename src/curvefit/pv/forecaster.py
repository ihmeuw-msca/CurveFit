"""
The Forecaster class is meant to fit regression models to the residuals
coming from evaluating predictive validity. We want to predict the residuals
forward with respect to how much data is currently in the model and how far out into the future.
"""

from curvefit.core.utils import data_translator
from curvefit.pv.residual_model import *


class Forecaster:
    def __init__(self):
        """
        A Forecaster will generate forecasts of residuals to create
        new, potential future datasets that can then be fit by the ModelPipeline
        """

        self.residual_model = None

    def fit_residuals(self, residual_data, col, covariates, residual_model_type, smooth_radius=None, num_smooths=None):
        """
        Run a regression for the mean and standard deviation
        of the scaled residuals.

        Args:
            residual_data: (pd.DataFrame) data frame of residuals
                that has the columns listed in the covariate
                of the residuals
            col: (str) the name of the column that has the residuals
            covariates: (str) the covariates to include in the model for residuals
            residual_model_type: (str) what type of residual model to it
                types include 'linear' and 'local'
            smooth_radius: (optional List[int]) smoother to pass to the Local residual smoother
            num_smooths: (optional int) number of times to run the smoother
        """
        if residual_model_type == 'linear':
            self.residual_model = LinearRM(
                data=residual_data,
                outcome=col,
                covariates=covariates
            )
        elif residual_model_type == 'local':
            if smooth_radius is None:
                raise RuntimeError("Need a value for smooth radius if you're doing local smoothing.")
            if num_smooths is None:
                raise RuntimeError("Need a value for the number of smooths you want "
                                   "to do if you're doing local smoothing.")
            self.residual_model = LocalSmoothSimpleExtrapolateRM(
                data=residual_data,
                outcome=col,
                covariates=covariates,
                radius=smooth_radius,
                num_smooths=num_smooths
            )
        else:
            raise ValueError(f"Unknown residual model type {residual_model_type}.")

        self.residual_model.fit()

    def simulate(self, mp, num_simulations, prediction_times, group, epsilon=1e-2, theta=1):
        """
        Simulate the residuals based on the mean and standard deviation of predicting
        into the future.

        Args:
            mp: (curvefit.model_generator.ModelPipeline) model pipeline
            prediction_times: (np.array) times to create predictions at
            num_simulations: number of simulations
            group: (str) the group to make the simulations for
            epsilon: (epsilon) the floor for standard deviation moving out into the future
            theta: (theta) scaling of residuals to do relative to prediction magnitude

        Returns:
            List[pd.DataFrame] list of data frames for each simulation
        """
        data = mp.all_data.loc[mp.all_data[mp.col_group] == group].copy()
        max_t = int(np.round(data[mp.col_t].max()))
        num_obs = data.loc[~data[mp.col_obs_compare].isnull()][mp.col_group].count()

        predictions = mp.mean_predictions[group]

        add_noise = prediction_times > max_t
        no_noise = prediction_times <= max_t

        forecast_out_times = prediction_times[add_noise] - max_t

        error = self.residual_model.sample(
            num_samples=num_simulations,
            forecast_out_times=forecast_out_times,
            num_data=num_obs,
            epsilon=epsilon
        )
        no_error = np.zeros(shape=(num_simulations, sum(no_noise)))
        all_error = np.hstack([no_error, error])

        noisy_forecast = predictions - (predictions ** theta) * all_error
        noisy_forecast = data_translator(
            data=noisy_forecast, input_space=mp.predict_space, output_space=mp.predict_space
        )
        return noisy_forecast
