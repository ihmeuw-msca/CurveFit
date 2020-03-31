"""
The Forecaster class is meant to fit regression models to the residuals
coming from evaluating predictive validity. We want to predict the residuals
forward with respect to how much data is currently in the model and how far out into the future.
"""

import numpy as np
import pandas as pd
import itertools
from curvefit.utils import data_translator


class ResidualModel:
    def __init__(self, data, outcome, covariates):
        """
        Base class for a residual model. Can fit and predict out.

        Args:
            data: (pd.DataFrame) data to use
            outcome: (str) outcome column name
            covariates: List[str] covariates to predict
        """
        self.data = data
        self.outcome = outcome
        self.covariates = covariates

        assert type(self.outcome) == str
        assert type(self.covariates) == list

        self.coef = None

    def fit(self):
        pass

    def predict(self, df):
        pass


class LinearResidualModel(ResidualModel):
    def __init__(self, **kwargs):
        """
        A basic linear regression for the residuals.

        Args:
            **kwargs: keyword arguments to ResidualModel base class
        """
        super().__init__(**kwargs)

    def fit(self):
        df = self.data.copy()
        df['intercept'] = 1
        pred = np.asarray(df[['intercept'] + self.covariates])
        out = np.asarray(df[[self.outcome]])
        self.coef = np.linalg.inv(pred.T.dot(pred)).dot(pred.T).dot(out)

    def predict(self, df):
        df['intercept'] = 1
        pred = np.asarray(df[['intercept'] + self.covariates])
        return pred.dot(self.coef)


class Forecaster:
    def __init__(self):
        """
        A Forecaster will generate forecasts of residuals to create
        new, potential future datasets that can then be fit by the ModelPipeline
        """

        self.mean_residual_model = None
        self.std_residual_model = None

    def fit_residuals(self, residual_data, mean_col, std_col,
                      residual_covariates, residual_model_type):
        """
        Run a regression for the mean and standard deviation
        of the scaled residuals.

        Args:
            residual_data: (pd.DataFrame) data frame of residuals
                that has the columns listed in the covariate
            mean_col: (str) the name of the column that has mean
                of the residuals
            std_col: (str) the name of the column that has the std
                of the residuals
            residual_covariates: (str) the covariates to include in the regression
            residual_model_type: (str) what type of residual model to it
                types include 'linear'

        """
        residual_data[f'log_{std_col}'] = np.log(residual_data[std_col])
        if residual_model_type == 'linear':
            self.mean_residual_model = LinearResidualModel(
                data=residual_data, outcome=mean_col, covariates=residual_covariates
            )
            self.std_residual_model = LinearResidualModel(
                data=residual_data, outcome=f'log_{std_col}', covariates=residual_covariates
            )
        else:
            raise ValueError(f"Unknown residual model type {residual_model_type}.")

        self.mean_residual_model.fit()
        self.std_residual_model.fit()

    def predict(self, far_out, num_data):
        """
        Predict out the residuals for all combinations of far_out and num_data
        for both the mean residual and the standard deviation of the residuals.

        Args:
            far_out: (np.array) of how far out to predict
            num_data: (np.array) of numbers of data points

        Returns:

        """
        data_dict = {'far_out': far_out, 'num_data': num_data}
        rows = itertools.product(*data_dict.values())
        new_data = pd.DataFrame.from_records(rows, columns=data_dict.keys())

        new_data['residual_mean'] = self.mean_residual_model.predict(df=new_data)
        new_data['log_residual_std'] = self.std_residual_model.predict(df=new_data)
        new_data['residual_std'] = np.exp(new_data['log_residual_std'])

        return new_data

    def simulate(self, mp, far_out, num_simulations, group, epsilon=1, theta=1e-2):
        """
        Simulate the residuals based on the mean and standard deviation of predicting
        into the future.

        Args:
            mp: (curvefit.model_generator.ModelPipeline) model pipeline
            far_out: (int) how far out into the future to predict
            num_simulations: number of simulations
            group: (str) the group to make the simulations for
            epsilon: (epsilon) the floor for standard deviation moving out into the future

        Returns:
            List[pd.DataFrame] list of data frames for each simulation
        """
        data = mp.all_data.loc[mp.all_data[mp.col_group] == group].copy()
        max_t = data[mp.col_t].max()
        num_obs = data.loc[~data[mp.col_obs_compare].isnull()][mp.col_group].count()

        num_out = np.array(range(far_out)) + 1
        forecast_times = max_t + num_out

        observations = np.asarray(data[mp.col_obs_compare])
        obs_times = np.asarray(data[mp.col_t])
        all_times = np.append(obs_times, forecast_times)

        mean_pred = mp.predict(
            times=forecast_times, predict_space=mp.predict_space, predict_group=group
        )
        residuals = self.predict(
            far_out=num_out,
            num_data=np.array([num_obs])
        )
        mean_residual = residuals['residual_mean'].values
        std_residual = residuals['residual_std'].apply(lambda x: max(x, epsilon)).values

        error = np.random.normal(
            loc=mean_residual, scale=std_residual, size=(num_simulations, far_out)
        )
        forecast_data = mean_pred + (mean_pred ** theta) * error
        simulated_flag = np.append(
            np.repeat(0, len(observations)),
            np.repeat(1, far_out)
        )
        cov_dict = {}
        for cov in mp.all_cov_names:
            covariate = data[cov].unique()
            assert len(covariate) == 1, f"There is not a unique covariate value for {cov} group {group}"
            cov_dict[cov] = covariate[0]

        dfs = []
        for i in range(num_simulations):
            new_observations = np.append(observations, forecast_data[i, :])
            # translate into new space with data translator
            fit_space_new_observations = data_translator(
                data=new_observations, input_space=mp.predict_space, output_space=mp.fun
            )
            df = pd.DataFrame({
                mp.col_t: all_times,
                mp.col_obs: fit_space_new_observations,
                mp.col_obs_compare: new_observations,
                mp.col_group: group,
                'simulated': simulated_flag,
                'intercept': 1
            })
            for k, v in cov_dict.items():
                df[k] = v
            if mp.obs_se_func is not None:
                df[mp.col_obs_se] = df[mp.col_t].apply(mp.obs_se_func)
            dfs.append(df)

        return dfs
