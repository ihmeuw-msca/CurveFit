"""
The Forecaster class is meant to fit regression models to the residuals
coming from evaluating predictive validity. We want to predict the residuals
forward with respect to how much data is currently in the model and how far out into the future.
"""

import numpy as np
import pandas as pd
import itertools


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
    def __init__(self, data, col_t, col_obs, col_group, all_cov_names):
        """
        A Forecaster will generate forecasts of residuals to create
        new, potential future datasets that can then be fit by the ModelPipeline

        Args:
            data: (pd.DataFrame) the model data
            col_t: (str) column of data that indicates time
            col_obs: (str) column of data that's in the same space
                as the forecast (linear space)
            col_group: (str) column of data that indicates group membership
            all_cov_names: List[str] list of all the covariate names that need
                to be copied forward
            model_pipeline
        """
        self.data = data
        self.col_t = col_t
        self.col_obs = col_obs
        self.col_group = col_group
        self.all_cov_names = all_cov_names

        assert type(self.all_cov_names) == list
        for l in self.all_cov_names:
            assert type(l) == str

        self.num_obs_per_group = self.get_num_obs_per_group()
        self.max_t_per_group = self.get_max_t_per_group()
        self.covariates_by_group = self.get_covariates_by_group()

        self.mean_residual_model = None
        self.std_residual_model = None

    def get_num_obs_per_group(self):
        """
        Get the number of observations per group that will inform
        the amount of data going forwards.

        Returns:
            (dict) dictionary keyed by group with value num obs
        """
        non_nulls = self.data.loc[~self.data[self.col_obs].isnull()].copy()
        return non_nulls.groupby(self.col_group)[self.col_group].count().to_dict()

    def get_max_t_per_group(self):
        """
        Get the maximum t per group.

        Returns:
            (dict) dictionary keyed by group with value max t
        """
        non_nulls = self.data.loc[~self.data[self.col_obs].isnull()].copy()
        return non_nulls.groupby(self.col_group)[self.col_t].max().to_dict()

    def get_covariates_by_group(self):
        """
        Get the covariate entries for each group to fill in the data frame.

        Returns:
            (dict[dict]) dictionary keyed by covariate then keyed by group with value as covariate value
        """
        cov_dict = {}
        for cov in self.all_cov_names:
            cov_dict[cov] = self.data.groupby(self.col_group)[cov].unique().to_dict()
            for k, v in cov_dict[cov].items():
                assert len(v) == 1, f"There is not a unique covariate value for group {k}"
                cov_dict[cov][k] = v[0]
        return cov_dict

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
        if residual_model_type == 'linear':
            self.mean_residual_model = LinearResidualModel(
                data=residual_data, outcome=mean_col, covariates=residual_covariates
            )
            self.std_residual_model = LinearResidualModel(
                data=residual_data, outcome=std_col, covariates=residual_covariates
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
        new_data['residual_std'] = self.std_residual_model.predict(df=new_data)

        return new_data

    def simulate(self, far_out, num_simulations, predictions, group,
                 model_pipeline):
        """
        Simulate the residuals based on the mean and standard deviation of predicting
        into the future.

        Args:
            far_out: (int)
            num_simulations: number of simulations to take
            predictions:

        Returns:
            List[pd.DataFrame] list of data frames for each simulation
        """
        # TODO: MAIN TO-DO!!! FINISH THIS!!! below is old
        # TODO: NEEDS TO FORECAST AND THEN TRANSLATE BETWEEN FIT SPACE AND FORECASTING SPACE
        # data = self.data.copy()
        # data['max_obs'] = data[self.col_group].map(self.num_obs_per_group)
        # data['max_t'] = data[self.col_group].map(self.max_t_per_group)
        # for cov in self.all_cov_names:
        #     data[cov] = data[self.col_group].map(self.covariates_by_group[cov])


