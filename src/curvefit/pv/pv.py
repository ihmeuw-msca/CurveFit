import numpy as np
import pandas as pd
from curvefit.diagnostics.plot_diagnostics import plot_residuals, plot_predictions, plot_residuals_1d, plot_es
from curvefit.core.utils import neighbor_mean_std


class PVGroup:
    def __init__(self, data, col_t, col_obs, col_grp, col_obs_compare,
                 model_generator, predict_space, predict_group):
        """
        Gets forward out of sample predictive validity for a model based on the function
        fit_model that takes arguments df and times and returns predictions at times.

        Args:
            data: (pd.DataFrame)
            col_t: (str) column indicating the time
            col_obs: (str) column indicating the observation
            col_grp: (str) column indicating the group membership
            col_obs_compare: (str) column indicating the observation that represents the space we will
                be calculating predictive validity in (can be different from col_obs, but your fit_model
                function for predict should match that)
            model_generator: object of class model_generator.ModelGenerator
            predict_space: a function from curvefit.model that gives the prediction space to calculate PV in
                (needs to be the same as col_obs_compare)
            predict_group: name of group to predict for

        Attributes:
            self.grp_df: (pd.DataFrame) the data frame for this group only
            self.times: (np.array) the available observation times
            self.difference: (np.array) the differences in time between consecutive observations
            self.compare_observations: (np.array) the observations that we want our predictions to be compared to
            self.amount_data: (np.array) the amount of data that each sequential model
            self.num_times: (int) the number of available times -- the number of models we can run
            self.models: (list) list of models created by the model generator

            self.prediction_matrix: (np.ndarray) array of shape self.times x self.times with predictions
                where the rows are each model and columns are each observation
            self.residual_matrix: (np.ndarray) difference between prediction matrix and observed data
            self.residuals: (np.ndarray) 2d array of shape self.times x 3 where the columns are:
                0: how far out predicting
                1: how many data points to do prediction
                2: the corresponding residual

        """
        self.df = data.copy()
        self.predict_group = predict_group
        self.col_t = col_t
        self.col_obs = col_obs
        self.col_grp = col_grp
        self.col_obs_compare = col_obs_compare
        self.predict_space = predict_space
        self.model_generator = model_generator

        assert type(self.col_t) == str
        assert type(self.col_obs) == str
        assert self.col_t in self.df.columns
        assert self.col_obs in self.df.columns
        assert callable(self.model_generator.fit)
        assert callable(self.model_generator.predict)

        self.grp_df = self.df.loc[self.df[self.col_grp] == self.predict_group].copy()
        self.times = np.unique(self.grp_df[self.col_t].values)
        self.num_times = len(self.times)

        # get the differences between the available times
        # these need to all be integers. the cumulative differences
        # tells us how big of a step we took to the next data point
        difference = np.diff(self.times)
        assert np.isclose(np.array([round(x) for x in difference]), difference, atol=1e-14).all()
        self.difference = np.array([int(round(x)) for x in difference])

        # which observations are we comparing the predictions to? and how much data do we have?
        self.compare_observations = self.grp_df[self.col_obs_compare].values
        self.amount_data = np.array(range(len(self.compare_observations))) + 1
        self.models = [model_generator.generate() for i in range(self.num_times)]
        self.prediction_matrix = None
        self.residual_matrix = None
        self.residuals = None

    @staticmethod
    def condense_residual_matrix(matrix, sequential_diffs, data_density):
        """
        Condense the residuals from a residual matrix to three columns
        that represent how far out the prediction was, the number of data points,
        and the observed residual.

        Args:
            matrix: (np.ndarray)
            sequential_diffs:
            data_density:

        Returns:

        """
        far_out = np.array([])
        num_data = np.array([])
        robs = np.array([])

        diagonals = np.array(range(matrix.shape[0]))[1:]

        # get the diagonal of the residual matrix and figure out
        # how many data points out we were predicting (convolve)
        # plus the amount of data that we had to do the prediction
        for i in diagonals:
            diagonal = np.diag(matrix, k=i)
            obs = len(diagonal)
            out = np.convolve(sequential_diffs, np.ones(i, dtype=int), mode='valid')

            far_out = np.append(far_out, out[-obs:])
            num_data = np.append(num_data, data_density[:obs])
            robs = np.append(robs, diagonal)

        # return the results for the residual matrix as a (len(available_times), 3) shaped matrix
        r_matrix = np.vstack([far_out, num_data, robs]).T
        return r_matrix

    def run_pv(self, theta):
        """
        Run predictive validity for all observation sequences in the available data for this group.
        """
        print(f"Running PV for {self.predict_group}")
        predictions = []

        for i, time in enumerate(self.times):
            print(f"Fitting model for end time {time}", end='\r')
            # remove the rows for this group that are greater than the available times
            remove_rows = (self.df[self.col_t] > time) & (self.df[self.col_grp] == self.predict_group)
            df = self.df[~remove_rows].copy()
            self.models[i].fit(df=df, group=self.predict_group)
            predictions.append(
                self.models[i].predict(
                    times=self.times,
                    predict_space=self.predict_space,
                    predict_group=self.predict_group
                )
            )
        self.prediction_matrix = np.vstack([predictions])
        self.compute_residuals(theta=theta)

        return self

    def compute_residuals(self, theta):
        """
        Compute the residual matrix and condense the residual matrix.
        Args:
            theta: power scaling for the predictions matrix, a theta of 0 means no scaling
                larger theta --> more scaling relative to prediction magnitude
        """
        assert 0 <= theta <= 1
        self.residual_matrix = (self.prediction_matrix - self.compare_observations) / (self.prediction_matrix ** theta)
        self.residuals = self.condense_residual_matrix(
            matrix=self.residual_matrix,
            sequential_diffs=self.difference,
            data_density=self.amount_data
        )

    def residual_df(self):
        return pd.DataFrame({
            self.col_grp: self.predict_group,
            'far_out': self.residuals[:, 0],
            'num_data': self.residuals[:, 1],
            'data_index': self.residuals[:, 0] + self.residuals[:, 1],
            'residual': self.residuals[:, 2]
        })

    def get_exponential_weights(self, exp_smoothing, max_last):
        """
        Get what the exponential weighting function result is based on the parameter.

        Args:
            exp_smoothing: (float) exponential smoothing parameter -->
                larger value will give more weight to more "recent" models
            max_last: (optional int) number of previous times to consider

        Returns:

        """
        times = self.times[-max_last:]

        weights = np.exp(-exp_smoothing * (np.max(times) - times))
        weights = weights / sum(weights)
        return weights

    def exp_smooth_preds(self, exp_smoothing, prediction_times, max_last=None):
        """
        Create a smoothed set of predictions with exponentially decreasing weights to further back
        forecasts.

        Args:
            exp_smoothing: (float) exponential smoothing parameter -->
                larger value will give more weight to more "recent" models
            prediction_times: (np.array) times to predict at
            max_last: (optional int) number of previous models to consider

        Returns:
            (np.array) of length prediction times
        """
        if max_last is None:
            max_last = len(self.times)

        weights = self.get_exponential_weights(
            exp_smoothing=exp_smoothing,
            max_last=max_last
        )
        weights = weights.reshape((len(weights), 1))
        predictions = []
        for i, t in enumerate(self.times[-max_last:]):
            predictions.append(self.models[-max_last:][i].predict(
                times=prediction_times,
                predict_space=self.predict_space,
                predict_group=self.predict_group
            ))

        weighted_predictions = np.vstack(predictions) * weights
        smooth_predictions = weighted_predictions.sum(axis=0)

        return smooth_predictions

    def plot_exponential_smoothing(self, exp_smoothing, prediction_times, max_last=None):
        """
        Plot exponential smoothing results.

        Args:
            exp_smoothing: (np.array) exponential smoothing parameter
            prediction_times: (np.array) prediction times
            max_last: (int) number of previous models to consider

        Returns:

        """
        plot_es(pv_group=self, exp_smoothing=exp_smoothing,
                prediction_times=prediction_times, max_last=max_last)


class PVModel:
    def __init__(self, data, col_group, col_t, col_obs, col_obs_compare, model_generator, predict_space):
        """
        Runs and stores predictive validity for a whole model and all groups in the model.

        Args:
            data: (pd.DataFrame)
            col_group: (str) grouping column string
            col_t: (str) column indicating the time
            col_obs: (str) column indicating the observation for fitting
            col_obs_compare: (str) column indicating the observation that represents the space we will
                be calculating predictive validity in (can be different from col_obs, but your fit_model
                function for predict should match that)
            model_generator: object of class model_generator.ModelGenerator
            predict_space: a function from curvefit.model that gives the prediction space to calculate PV in
                (needs to be the same as col_obs_compare)

        Attributes:
            self.groups: the groups in this model
            self.pv_groups: a dictionary keyed by group with PVGroup for that group
            self.all_residuals: 2d array of stacked residuals from each PVGroup
            self.r_mean: averaged residuals over far out and data density
            self.r_std: standard deviation of residuals over far out and data density
            self.r_mad: MAD of residuals over far out and data density
        """
        self.df = data.copy()
        self.df.sort_values([col_group, col_t], inplace=True)

        self.col_group = col_group
        self.col_t = col_t
        self.col_obs = col_obs
        self.col_obs_compare = col_obs_compare
        self.model_generator = model_generator
        self.predict_space = predict_space

        assert type(self.col_group) == str
        assert self.col_group in self.df.columns

        self.groups = sorted(self.df[self.col_group].unique())

        self.model_generator.run_init_model()

        self.pv_groups = {
            grp: PVGroup(
                data=self.df, col_t=self.col_t, col_obs=self.col_obs, col_grp=self.col_group,
                col_obs_compare=self.col_obs_compare, model_generator=self.model_generator.generate(),
                predict_space=self.predict_space, predict_group=grp
            ) for grp in self.groups
        }

        self.all_residuals = None
        self.all_smoothed_residuals = None

    def get_all_residuals(self):
        """
        Grab all of the residuals from the group-specific models into
        a data frame.
        """
        self.all_residuals = pd.concat([
            grp.residual_df() for grp in self.pv_groups.values()
        ])

    def run_pv(self, theta):
        """
        Run predictive validity for all of the groups.

        Args:
            theta: (float) from 0 to 1 indicating how much scaling to
                do of the residuals relative to the prediction magnitude
                theta of 0 means no scaling, theta of 1 means max scaling
        """
        for group in self.groups:
            self.pv_groups[group].run_pv(theta=theta)
        self.get_all_residuals()

    def recompute_residuals(self, theta):
        """
        Recompute what the residuals would be with a new theta for scaling.
        Args:
            theta: (float)
        """
        for group in self.groups:
            self.pv_groups[group].compute_residuals(theta=theta)
        self.get_all_residuals()

    def get_smoothed_residuals(self, radius):
        """
        Smooth residuals for all of the data across some radius
        Args:
            radius: List[int] with two elements indicating the width and height
                of the square that we want to smooth over.

        Returns:
            smoothed_residuals: (pd.DataFrame)
        """
        smoothed_residuals = neighbor_mean_std(
            df=self.all_residuals,
            col_val='residual',
            col_group=self.col_group,
            col_axis=['far_out', 'num_data'],
            radius=radius
        )
        return smoothed_residuals

    def plot_simple_residuals(self, x_axis, y_axis, radius, color=None, exclude_groups=None):
        """
        Make simple residual scatter plots by location.
        Args:
            x_axis: (str)
            y_axis: (str)
            radius: List[int] radius for smoothing
            color: (str)

        Returns:

        """
        smooth = self.get_smoothed_residuals(radius=radius)
        if exclude_groups is not None:
            smooth = smooth.loc[~smooth[self.col_group].isin(exclude_groups)].copy()
        plot_residuals_1d(residual_df=smooth, x_axis=x_axis, y_axis=y_axis, group_col=self.col_group, color=color)
        for g in self.groups:
            plot_residuals_1d(residual_df=smooth, x_axis=x_axis, y_axis=y_axis,
                              group_col=self.col_group, color=color, group=g)

    def triangle_residual_plots(self, radius, x_axis='far_out', y_axis='num_data', exclude=0, absolute=False):
        """
        Plot all of the residuals based on some exclusion criteria for
        number of data points that were used in the fitting and some radius
        for smoothing over a window of neighboring data density / number predicting out.

        Args:
            radius: List[int]
            absolute: (bool) plot mean residuals by value or absolute value
            exclude: (int) exclude model fits with under this many data points
            x_axis: (str) the x-axis variable to plot (one of far_out, num_data, or data_index)
            y_axis: (str) the y-axis variable to plot (one of far_out, num_data, or data_index)
        """
        smoothed_residuals = self.get_smoothed_residuals(radius=radius)
        smoothed_residuals = smoothed_residuals.loc[smoothed_residuals['num_data'] > exclude].copy()

        for i, (k, v) in enumerate(self.pv_groups.items()):

            residuals = self.all_residuals.loc[self.all_residuals[self.col_group] == k].copy()
            plot_residuals(residual_array=np.asarray(residuals[[x_axis, y_axis, 'residual']]),
                           group_name=k, absolute=absolute,
                           x_label=x_axis, y_label=y_axis)
            smooth = smoothed_residuals.loc[smoothed_residuals[self.col_group] == k]
            smooth_mean = np.asarray(smooth[[x_axis, y_axis, 'residual_mean']])
            smooth_std = np.asarray(smooth[[x_axis, y_axis, 'residual_std']])
            plot_residuals(residual_array=smooth_mean,
                           group_name=f'{k} smooth mean radius {radius}',
                           absolute=absolute,
                           x_label=x_axis, y_label=y_axis)
            plot_residuals(residual_array=smooth_std,
                           group_name=f'{k} smooth std radius {radius}',
                           absolute=True,
                           x_label=x_axis, y_label=y_axis)

    def plot_predictions(self, group_name):
        """
        Plot the predictions for one location for each model that was fit deleting the ith data point.
        Args:
            group_name: (str) location to plot
        """
        times = self.pv_groups[group_name].times
        observations = self.pv_groups[group_name].compare_observations
        predictions = self.pv_groups[group_name].prediction_matrix
        plot_predictions(prediction_array=predictions, group_name=group_name,
                         times=times, observations=observations)
