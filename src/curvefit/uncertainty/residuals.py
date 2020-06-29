from dataclasses import dataclass, fields, field, InitVar
from typing import List
import numpy as np
import pandas as pd


@dataclass
class ResidualInfo:
    """
    {begin_markdown ResidualInfo}
    {spell_markdown metadata}

    # `curvefit.uncertainty.predictive_validity.ResidualInfo`
    ## Keeps track of metadata about the residuals

    ## Arguments

    - `group_name (str)`: name of the group
    - `times (np.array)`: times that have data
    - `obs (np.array)`: observations at `times`

    ## Attributes

    - `num_times (int)`: number of times
    - `difference (np.array)`: array of length `num_times - 1` that represents
        the sequential differences in each time to the next
    - `amount_data (np.array)`: the amount of data at each time point

    {end_markdown ResidualInfo}
    """
    group_name: str
    times: np.array
    obs: np.array

    num_times: int = field(init=False)
    difference: np.array = field(init=False)
    amount_data: np.array = field(init=False)

    def __post_init__(self):
        self.num_times = len(self.times)
        self.difference = np.array([int(round(x)) for x in np.diff(self.times)])
        self.amount_data = np.array(range(len(self.obs))) + 1


class Residuals:
    """
    {begin_markdown Residuals}
    {spell_markdown metadata}

    # `curvefit.uncertainty.predictive_validity.residuals`
    ## Data storage and manipulation for residual matrices

    The `Residuals` class keeps track of a prediction matrix and the associated
    residual matrix at each time point.

    ## Arguments

    - `residual_info (ResidualInfo)`: metadata about residuals
    - `data_specs (curvefit.core.data.DataSpecs)`: specifications about what data
        was passed in in order to generate these residuals

    ## Attributes

    - `prediction_matrix (np.ndarray)`: square matrix of size total number of time points
        for a group. The rows of the matrix are predictions from models fit on progressively
        *more* data, and the columns of the matrix are the predictions for each point
        in the time series. **Everything above the diagonal is an out of sample prediction.**
    - `residual_matrix (np.ndarray)`: square matrix of the same size as prediction
        matrix but has had observations subtracted off of it and (potentially) scaled
        by the prediction value

    ## Methods

    ### `_record_predictions`
    Records a set of predictions into the prediction matrix.

    - `i (int)`: the ith set of predictions (the whole time series) to record
    - `predictions (np.array)`: 1d numpy array of predictions across the time series

    ### `_compute_residuals`
    Given some observed data and an amount of scaling (theta), compute the residuals.

    - `obs (np.array)`: 1d numpy array of observed data in the same space as the predictions
    - `theta (float)`: amount of scaling. A `theta = 1` means that they are relative residuals
        (relative to the prediction magnitude) and a `theta = 0` means that they are
        absolute residuals

    ### `_condense_matrix`
    Takes a square matrix of predictions or residuals and condenses this to
    a smaller matrix that only has out of sample predictions or residuals, and matches
    it to metadata about those residuals or predictions including how much data was used
    to predict (data_density --> "num_data") and how far out was this
    prediction time point from the last observed time point (sequential diffs --> "far_out").

    - `matrix (np.ndarray)`: the square matrix to condense
    - `sequential_diffs (np.array)`: 1d array of sequential differences in time
        between observations (e.g. observations 3 time points apart would have
        sequential_diffs = 3) (results in the "far_out" column)
    - `data_density (np.array)`: the amount of data at each time point dropping
        all observations beyond this time point (results in the "num_data" column)

    {end_markdown Residuals}
    """

    def __init__(self, residual_info, data_specs):
        self.residual_info = residual_info
        self.data_specs = data_specs

        self.prediction_matrix = np.empty((residual_info.num_times, residual_info.num_times))
        self.prediction_matrix[:] = np.nan

        self.residual_matrix = np.empty(self.prediction_matrix.shape)
        self.residual_matrix[:] = np.nan

    def _record_predictions(self, i, predictions):
        self.prediction_matrix[i, :] = predictions

    def _compute_residuals(self, obs, theta):
        for i in range(self.residual_matrix.shape[0]):
            self.residual_matrix[i, :] = (self.prediction_matrix[i, :] - obs) / (self.prediction_matrix[i, :] ** theta)

    @staticmethod
    def _condense_matrix(matrix, sequential_diffs, data_density):
        row_idx, col_idx = np.triu_indices(matrix.shape[0], 1)
        map1 = np.cumsum(np.insert(sequential_diffs, 0, 0))
        map2 = data_density

        far_out = map1[col_idx] - map1[row_idx]
        num_data = map2[row_idx]
        robs = matrix[row_idx, col_idx]

        # return the results for the residual matrix
        # as a (len(available_times), 3) shaped matrix
        r_matrix = np.vstack([far_out, num_data, robs]).T
        return r_matrix

    def _residual_df(self):
        residual_matrix = self._condense_matrix(
            matrix=self.residual_matrix,
            sequential_diffs=self.residual_info.difference,
            data_density=self.residual_info.amount_data
        )
        return pd.DataFrame({
            self.data_specs.col_group: self.residual_info.group_name,
            'far_out': residual_matrix[:, 0],
            'num_data': residual_matrix[:, 1],
            'data_index': residual_matrix[:, 0] + residual_matrix[:, 1],
            'residual': residual_matrix[:, 2]
        })
