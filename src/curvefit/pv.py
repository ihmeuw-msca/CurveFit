import numpy as np
from curvefit.diagnostics import plot_residuals


def get_residual_std(matrix, n, window):
    """
    Gets the residual standard deviation from a model
    residual matrix predicting n time points out from its last data point
    and using a maximum time window to average.

    NOTE: masks nans in case there was a missing observation.

    Args:
        matrix: (np.ndarray) square 2 dimensional array with
            each time points observation across the columns and
            each model fit with the last time point across rows
        n: (int) number of time points predicting out (n=0) would be
            the entry for the model where n^{th} observation was the
            last observation in the model
        window: (int) number of time + n points to use from the most
            recent one

    Returns:
        (float)
    """
    assert len(matrix.shape) == 2
    assert matrix.shape[0] == matrix.shape[1]
    assert type(window) == int
    assert type(n) == int

    return np.nanstd(
        np.diag(matrix, k=n)[-window:]
    )


def get_full_data(grp_df, col_t, col_obs, prediction_times):
    """
    Fill out an observation vector for all one increment time
    points, filling in missing values with nan.

    Args:
        grp_df: (pd.DataFrame) data frame with time and observation
            column
        col_t: (str) the time column
        col_obs: (str) the observation column
        prediction_times: (np.array) the time vector for which predictions are needed

    Returns:
        (np.array) observations filled out to the max time point
        in the loc_df
    """
    assert type(col_t) == str
    assert type(col_obs) == str
    assert col_t in grp_df.columns
    assert col_obs in grp_df.columns

    data_dict = grp_df[[col_t, col_obs]].set_index(col_t).to_dict(orient='index')
    data_dict = {k: v[col_obs] for k, v in data_dict.items()}
    full_data = np.array([data_dict[x] if x in data_dict else np.nan for x in prediction_times])
    return full_data


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

    def run_pv(self):
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
            self.models[i].fit(df=df)
            predictions.append(
                self.models[i].predict(
                    times=self.times,
                    predict_space=self.predict_space,
                    predict_group=self.predict_group
                )
            )
            self.prediction_matrix = np.vstack([predictions])
            self.residual_matrix = self.prediction_matrix - self.compare_observations

        self.residuals = self.condense_residual_matrix(
            matrix=self.residual_matrix,
            sequential_diffs=self.difference,
            data_density=self.amount_data
        )
        return self


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

        self.pv_groups = {
            grp: PVGroup(
                data=self.df, col_t=self.col_t, col_obs=self.col_obs, col_grp=self.col_group,
                col_obs_compare=self.col_obs_compare, model_generator=self.model_generator,
                predict_space=self.predict_space, predict_group=grp
            ) for grp in self.groups
        }

    def run_pv(self):
        """
        Run predictive validity for all of the groups.
        """
        for group in self.groups:
            self.pv_groups[group].run_pv()

    def plot_diagnostics(self, absolute=False):
        for k, v in self.pv_groups.items():
            plot_residuals(residual_array=v.residuals, group_name=k, absolute=absolute)

