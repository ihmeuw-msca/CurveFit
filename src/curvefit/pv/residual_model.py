import numpy as np
import pandas as pd
import itertools
from curvefit.core.utils import neighbor_mean_std


class ResidualModel:
    def __init__(self, data, outcome, covariates):
        """
        Base class for a residual model. Can fit, predict out, and sample residuals.

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

    @staticmethod
    def predict_frame(far_out, num_data):
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
        return new_data

    def sample(self, num_samples, forecast_out_times, num_data, epsilon):
        """
        Args:
            num_samples: (int) number of draws
            forecast_out_times: (np.array) of times
            num_data: (int) number of existing data points
            epsilon: (float) cv floor

        Returns:
            (np.ndarray) with shape (num_samples, forecast_out_times)
        """
        pass


class SimpleResidualModel(ResidualModel):
    """
    A simple residual model that uses standard deviation of residuals.
    """
    def sample(self, num_samples, forecast_out_times, num_data, epsilon):
        residuals = self.predict_frame(
            far_out=forecast_out_times, num_data=np.array([num_data])
        )
        residuals['residual_mean'] = 0
        residuals['residual_std'] = self.predict(df=residuals)

        std_residual = residuals['residual_std'].apply(lambda x: max(x, epsilon)).values
        standard_noise = np.random.randn(num_samples)
        error = np.outer(standard_noise, std_residual)
        return error


class LinearRM(SimpleResidualModel):
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
        df['log_num_data_transformed'] = np.log(df['num_data_transformed'])
        pred = np.asarray(df[self.covariates])
        out = np.asarray(df[[self.outcome]])
        self.coef = np.linalg.inv(pred.T.dot(pred)).dot(pred.T).dot(out)

    def predict(self, df):
        df['intercept'] = 1
        df['inv_num_data'] = 1 / df['num_data']
        df['num_data_transformed'] = 1 / (1 + df['num_data'])
        df['log_num_data_transformed'] = np.log(df['num_data_transformed'])
        pred = np.asarray(df[self.covariates])
        return pred.dot(self.coef)


class LocalSmoothDistanceExtrapolateRM(SimpleResidualModel):
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
            distance = np.zeros(len(smooth))
            # TODO: Fix this distance function
            for cov in ['far_out', 'num_data']:
                distance += (row[cov] - smooth[cov]) ** 2
            distance = distance ** 0.5
            result = sum(distance ** (-1) * smooth['residual_std']) / sum(distance)
            data.at[i, 'residual_std'] = result

        return data['residual_std'].values


class LocalSmoothSimpleExtrapolateRM(SimpleResidualModel):
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
