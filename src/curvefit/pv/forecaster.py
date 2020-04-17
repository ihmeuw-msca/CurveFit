"""
The Forecaster class is meant to fit regression models to the residuals
coming from evaluating predictive validity. We want to predict the residuals
forward with respect to how much data is currently in the model and how far out into the future.
"""

import numpy as np
import pandas as pd
import itertools
from curvefit.core.utils import data_translator
from curvefit.core.utils import neighbor_mean_std


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

    def fit(self):
        pass

    def predict(self, df):
        pass


class LinearRM(ResidualModel):
    def __init__(self, **kwargs):
        """
        A basic linear regression for the residuals.

        Args:
            **kwargs: keyword arguments to ResidualModel base class
        """
        super().__init__(**kwargs)
        self.coef = None

    def fit(self):
        df = self.data.copy()
        df['intercept'] = 1
        df['inv_num_data'] = 1 / df['num_data']
        df['num_data_transformed'] = 1 / (1 + df['num_data'])
        df['ln_num_data_transformed'] = np.log(df['num_data_transformed'])
        pred = np.asarray(df[self.covariates])
        out = np.asarray(df[[self.outcome]])
        self.coef = np.linalg.inv(pred.T.dot(pred)).dot(pred.T).dot(out)

    def predict(self, df):
        df['intercept'] = 1
        df['inv_num_data'] = 1 / df['num_data']
        df['num_data_transformed'] = 1 / (1 + df['num_data'])
        df['ln_num_data_transformed'] = np.log(df['num_data_transformed'])
        pred = np.asarray(df[self.covariates])
        return pred.dot(self.coef)


class LocalSmoothDistanceExtrapolateRM(ResidualModel):
    def __init__(self, radius, **kwargs):
        """
        An n-dimensional smoother for the covariates that are passed. Extrapolates
        for unobserved values of the covariates based on weighted average of all observed
        where weights are the inverse of the distance in n-dimensional covariate space.

        Args:
            radius: List[int] radius of smoother in each direction
            **kwargs: keyword arguments to ResidualModel base class
        """
        super().__init__(**kwargs)
        self.radius = radius
        self.smoothed = None

    def fit(self):
        df = self.data.copy()
        df['Smooth_Group'] = 'All'
        # smooth
        self.smoothed = neighbor_mean_std(
            df=df, col_val=self.outcome,
            col_group='Smooth_Group',
            col_axis=self.covariates,
            radius=self.radius
        )

    def predict(self, df):
        data = df.copy()
        outcome = 'residual_std'

        smooth = self.smoothed.copy()

        index = data.index
        data = data.merge(smooth, on=self.covariates, how='left', sort=False)
        data = data.iloc[index]
        nans = np.isnan(data[outcome])

        fill_in = data.loc[nans]
        for i, row in fill_in.iterrows():
            distance = 0
            for cov in ['far_out', 'num_data']:
                distance += (row[cov] - smooth[cov]) ** 2
            distance = distance ** 0.5
            result = sum(distance ** (-1) * smooth['residual_std']) / sum(distance)
            data.at[i, 'residual_std'] = result

        return data['residual_std'].values


class LocalSmoothSimpleExtrapolateRM(ResidualModel):
    def __init__(self, radius, num_smooths, **kwargs):
        """
        An n-dimensional smoother for the covariates that are passed.
        Args:
            radius: List[int] radius of smoother in each direction
            num_smooths: (int) number of times to go through the smoothing process
            **kwargs: keyword arguments to ResidualModel base class
        """
        super().__init__(**kwargs)
        self.radius = radius
        self.num_smooths = num_smooths
        self.smoothed = None

    def fit(self):
        df = self.data.copy()
        df['group'] = 'All'
        # smooth
        self.smoothed = neighbor_mean_std(
            df=df, col_val=self.outcome,
            col_group='group',
            col_axis=self.covariates,
            radius=self.radius
        )
        if self.num_smooths > 1:
            i = 1
            data = self.smoothed.copy()
            data.rename(columns={'residual_std': 'residual'}, inplace=True)
            data.drop(columns={'residual_mean'}, axis=1, inplace=True)

            while i < self.num_smooths:
                self.smoothed = neighbor_mean_std(
                    df=data, col_val='residual',
                    col_group='group',
                    col_axis=self.covariates,
                    radius=self.radius
                )
                data = self.smoothed.copy()
                data.rename(columns={'residual_mean': 'residual'}, inplace=True)
                data.drop(['residual_std'], inplace=True, axis=1)
                i += 1

            self.smoothed.drop('residual_std', inplace=True, axis=1)
            self.smoothed.rename(columns={'residual_mean': 'residual_std'}, inplace=True)

    def predict(self, df):
        data = df.copy()
        outcome = 'residual_std'
        smooth = self.smoothed.copy()

        index = data.index
        data = data.merge(smooth, on=self.covariates, how='left', sort=False)
        data = data.iloc[index]

        corner_value = smooth[smooth['num_data'] == smooth['num_data'].max()]['residual_std'].mean()

        for i, row in data.iterrows():
            if np.isnan(row[outcome]):
                df_sub = smooth[smooth['num_data'] == row['num_data']].copy()
                if df_sub.empty:
                    new_val = corner_value
                else:
                    max_far_out = df_sub[~df_sub['residual_std'].isnull()]['far_out'].max()
                    new_val = np.nanmean(df_sub[df_sub['far_out'] == max_far_out]['residual_std'][-1:])
                data.at[i, 'residual_std'] = new_val

        return data['residual_std'].values


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
                data=residual_data, outcome=col, covariates=covariates
            )
        elif residual_model_type == 'local':
            if smooth_radius is None:
                raise RuntimeError("Need a value for smooth radius if you're doing local smoothing.")
            if num_smooths is None:
                raise RuntimeError("Need a value for the number of smooths you want "
                                   "to do if you're doing local smoothing.")
            self.residual_model = LocalSmoothSimpleExtrapolateRM(
                data=residual_data, outcome=col, covariates=covariates, radius=smooth_radius, num_smooths=num_smooths
            )
        else:
            raise ValueError(f"Unknown residual model type {residual_model_type}.")

        self.residual_model.fit()

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
        new_data['data_index'] = new_data['far_out'] + new_data['num_data']

        new_data['residual_mean'] = 0
        new_data['residual_std'] = self.residual_model.predict(df=new_data)

        return new_data

    def create_residual_samples(self, num_simulations, forecast_out_times, num_data, std_floor, std_ceiling=np.inf):
        """

        Args:
            num_simulations: (int) number of draws
            forecast_out_times: (np.array) of times
            num_data: (int) number of existing data points
            std_floor: (float) std floor
            std_ceiling: (float) std ceiling

        Returns:
            (np.ndarray) with shape (num_simulations, forecast_out_times)
        """
        residuals = self.predict(
            far_out=forecast_out_times, num_data=np.array([num_data])
        )
        std_residual = residuals['residual_std'].apply(lambda x: min(max(x, std_floor), std_ceiling)).values
        standard_noise = np.random.randn(num_simulations)
        error = np.outer(standard_noise, std_residual)
        return error

    def simulate(self, mp, num_simulations, prediction_times, group, std_floor=1e-2, theta=1,
                 std_ceiling=np.inf):
        """
        Simulate the residuals based on the mean and standard deviation of predicting
        into the future.

        Args:
            mp: (curvefit.model_generator.ModelPipeline) model pipeline
            prediction_times: (np.array) times to create predictions at
            num_simulations: number of simulations
            group: (str) the group to make the simulations for
            std_floor: (float) std floor
            std_ceiling: (float) std ceiling
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
        error = self.create_residual_samples(
            num_simulations=num_simulations,
            forecast_out_times=forecast_out_times,
            num_data=num_obs,
            std_floor=std_floor,
            std_ceiling=std_ceiling
        )
        no_error = np.zeros(shape=(num_simulations, sum(no_noise)))
        all_error = np.hstack([no_error, error])

        noisy_forecast = predictions - (predictions ** theta) * all_error
        noisy_forecast = data_translator(
            data=noisy_forecast, input_space=mp.predict_space, output_space=mp.predict_space
        )
        return noisy_forecast
