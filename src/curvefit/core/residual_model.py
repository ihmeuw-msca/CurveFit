import itertools
import pandas as pd


class _ResidualModel:
    """
    {begin_markdown _ResidualModel}

    {spell_markdown subclassed}

    # `curvefit.core.residual_model._ResidualModel`
    ## A model for describing the out of sample residuals

    This is a model that describes how the out of sample residuals vary
    with a set of covariates. The goal is to be able to understand what the residuals
    moving into the future will look like based on key covariates (e.g. how much data
    do we currently have, how far out are we trying to predict, etc.).

    Ultimately this class simulates residuals into the future. It does this in *coefficient of variation*
    space (if predicting in log space --> absolute residuals; if predicting in linear space --> relative residuals).

    This is the **BASE** class and should not be used directly. It should be
    subclassed and the methods overwritten based on the type of residual model.

    See the subclasses descriptions for each of their methods including
    [SmoothResidualModel](#extract_md/SmoothResidualModel.md).

    ## Arguments

    - `cv_bounds (List[float])`: a 2-element list of bounds on the coefficient of variation.
        The first element is the lower bound and the second is the upper bound
    - `covariates (Dict[str: None, str])`: a dictionary of covariates to use in the model. The keys of the dictionary
        are the names of covariates to include in the residual model fitting and the optional values for each key
        are the subsets of the covariates to use
        (e.g. only use the subset of data where covariate1 > 10. in the fitting).
    - `exclude_groups (List[str])`: a list of groups to exclude from the fitting process (not excluded from
        making predictions)

    ## Methods

    `fit_residuals`
    Fits the residual model to the residual data frame passed in.

    - `residual_df (pd.DataFrame)`: a data frame that contains all of the covariates and a residual observation
        that will be used for fitting the model (the design matrix)

    `simulate_residuals`
    Simulates residuals from the fitted residual model for particular covariate values.

    - `covariate_specs (Dict[str: np.array])`: a dictionary of covariate values to create
        residual simulations for
    - `num_simulations (int)`: number of residual simulations to produce

    {end_markdown _ResidualModel}
    """
    def __init__(self, cv_bounds, covariates, exclude_groups):

        self.cv_bounds = cv_bounds
        self.covariates = covariates
        self.exclude_groups = exclude_groups

        assert type(self.covariates) == dict
        for k, v in self.covariates.items():
            assert type(k) == str
            if v is not None:
                assert type(v) == str

        assert type(self.cv_bounds) == list
        assert len(self.cv_bounds) == 2
        for i in self.cv_bounds:
            assert type(i) == float
            assert i > 0.

        if self.exclude_groups is not None:
            assert type(self.exclude_groups) == list
            for i in self.exclude_groups:
                assert type(i) == str

    def fit_residuals(self, residual_df):
        pass

    @staticmethod
    def _expand_grid(covariate_specs):
        rows = itertools.product(*covariate_specs.values())
        new_data = pd.DataFrame.from_records(rows, columns=covariate_specs.keys())
        return new_data

    def _predict_residuals(self, covariate_specs):
        pass

    def simulate_residuals(self, covariate_specs, num_simulations):
        pass


class SmoothResidualModel(_ResidualModel):
    """
    {begin_markdown SmoothResidualModel}

    {spell_markdown bool prioritizes}

    # `curvefit.core.residual_model.SmoothResidualModel`
    ## A local smoother for the coefficient of variation in forecasts

    This is a residual model (see [_ResidualModel](#extract_md/_ResidualModel.md for a description).
    This particular residual model creates a smoothed standard deviation over the residual data.
    It calculates the standard deviation of the residuals with a moving window
    over neighboring covariate values.

    The specific covariates for this residual model are num_data and far_out: how much data did the
    prediction model have and how far out into the future was it predicting.
    To extrapolate to unobserved values of the covariates in order to predict the
    residuals for those observations, it prioritizes num_data and then far_out in
    a simple "carry forward" extrapolation.

    ## Syntax
    ```python
    d = SmoothResidualModel(
        cv_bounds, covariates, exclude_groups,
        num_smooth_iterations, smooth_radius, robust
    )
    ```

    ## Arguments

    - `cv_bounds (List[float])`: a 2-element list of bounds on the coefficient of variation.
        The first element is the lower bound and the second is the upper bound
    - `covariates (Dict[str: None, str])`: a dictionary of covariates to use in the model. The keys of the dictionary
        are the names of covariates to include in the residual model fitting and the optional values for each key
        are the subsets of the covariates to use
        (e.g. only use the subset of data where covariate1 > 10. in the fitting). **NOTE**: this residual model
        requires that only two covariates, `"far_out"` and `"num_data"` are used.
    - `exclude_groups (List[str])`: a list of groups to exclude from the fitting process (not excluded from
        making predictions)
    - `num_smooth_iterations (int)`: how many times should the model smooth over the residual matrix. If 1, then
        only calculates the standard deviation over a moving window. If > 1, then runs a local smoother over the
        standard deviation surface `num_smooth_iterations - 1` times.
    - `smooth_radius (List[int])`: the size of the moving window in each of the covariate directions. Since
        for this residual model only two covariates are used, this needs to be a 2-element list of integers
        specifying how many units of each covariate to consider for the neighborhood. For example, if
        `covariates` is a dictionary with ['far_out', 'num_data'] as the keys, and `smooth_radius=[2, 2]`,
        then the standard deviation for the residual with `far_out == 4` and `num_data == 3` will be calculated
        over the window `2 < far_out < 6` and `1 < num_data < 5`.
    - `robust (bool)`: whether or not to use a robust estimator for the
        standard deviation (1.4826 * median absolute deviation).

    ## Attributes

    - `self.residual_std_matrix (np.array)`: a smooth surface of residual standard deviations across the covariate
        axes, only populated after `fit_residuals()` is called

    ## Methods

    See [_ResidualModel](#extract_md/_ResidualModel.md for descriptions of the class methods
    for a _ResidualModel.

    ## Usage

    ```python
    SmoothResidualModel(
        cv_bounds=[0., np.inf],
        covariates={'far_out': 'far_out >= 10', 'num_data': None},
        exclude_groups=None,
        num_smooth_iterations=1,
        smooth_radius=[1, 1],
        robust=True
    )
    ```

    {end_markdown SmoothResidualModel}
        """
    def __init__(self, num_smooth_iterations, smooth_radius, robust, **kwargs):

        super().__init__(**kwargs)

        assert 'far_out' in self.covariates, "The SmoothResidualModel must have far_out as a covariate."
        assert 'num_data' in self.covariates,  "The SmoothResidualModel must have num_data as a covariate."

        self.num_smooth_iterations = num_smooth_iterations
        self.smooth_radius = smooth_radius
        self.robust = robust

        assert type(self.num_smooth_iterations) == int
        assert self.num_smooth_iterations > 0

        assert len(self.smooth_radius) == 2
        for i in self.smooth_radius:
            assert type(i) == int
            assert i >= 0

        assert type(self.robust) == bool

        self.residual_std_matrix = None

    def fit_residuals(self, residual_df):
        pass

    def _predict_residuals(self, covariate_specs):
        pass

    def simulate_residuals(self, covariate_specs, num_simulations):
        pass
