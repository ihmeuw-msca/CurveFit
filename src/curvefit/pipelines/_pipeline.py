from copy import deepcopy
import numpy as np

from curvefit.pv.forecaster import Forecaster
from curvefit.pv.pv import PVModel
from curvefit.diagnostics.plot_diagnostics import plot_fits


class ModelPipeline:
    """
    Base class for a model generator.
    If a model needs to have initial parameters started for the predictive validity,
    put that in run_init_model
    """
    def __init__(self, all_data, col_t, col_obs, col_group,
                 col_obs_compare, all_cov_names, fun, predict_space, obs_se_func=None):
        """
        Base class for a model pipeline. At minimum needs the following arguments for a
        model pipeline.

        Args:
            all_data: (pd.DataFrame) of *all* the data that will go into this modeling pipeline
            col_t: (str) name of the column with time
            col_group: (str) name of the column with the group in it
            col_obs: (str) the name of the column with observations for fitting the model
            col_obs_compare: (str) the name of the column that will be used for predictive validity comparison
            all_cov_names: List[str] list of name(s) of covariate(s). Not the same as the covariate specifications
                that are required by CurveModel in order of parameters. You should exclude intercept from this list.
            fun: (callable) the space to fit in, one of curvefit.functions
            predict_space: (callable) the space to do predictive validity in, one of curvefit.functions
            obs_se_func: (optional) function to get observation standard error from col_t

        Attributes:
            self.pv: (curvefit.pv.PVModel) predictive validity model
            self.forecaster: (curvefit.forecaster.Forecaster) residual forecasting tool
            self.mean_predictions: (dict) dictionary of mean predictions keyed by group
            self.simulated_data: (dict) dictionary of simulated datasets keyed by group
            self.draws: (dict) dictionary of resulting keyed by group
        """
        self.all_data = all_data
        self.col_t = col_t
        self.col_group = col_group
        self.col_obs = col_obs
        self.col_obs_compare = col_obs_compare
        self.all_cov_names = all_cov_names
        self.fun = fun
        self.predict_space = predict_space
        self.obs_se_func = obs_se_func

        # If we're doing predictive validity in log space, we want absolute residuals, which
        # corresponds to theta = 0. If we're doing predictive validity in linear space, we want
        # relative residuals, which corresponds to theta = 1.
        if self.predict_space.__name__.startswith('ln'):
            self.theta = 0
        else:
            self.theta = 1

        # If we're fitting in log space we ultimately want everything in
        # normal space. But it's simulated normally in log space
        # and we need mean to be unbiased in linear space.
        #
        # This flag will de-bias the draws
        # mean in linear space to match exactly the exp mean in log space.
        self.de_bias_draws = self.predict_space.__name__.startswith('ln')

        if self.obs_se_func is not None:
            self.col_obs_se = 'obs_se'
            self.all_data[self.col_obs_se] = self.all_data[self.col_t].apply(self.obs_se_func)
        else:
            self.col_obs_se = None

        # these are the attributes that can't be used to initialize a
        # CurveModel but are needed to initialize the ModelPipeline
        self.pop_cols = [
            'all_data', 'all_cov_names', 'col_obs_compare', 'predict_space', 'obs_se_func'
        ]

        self.all_data.sort_values([col_group, col_t], inplace=True)
        self.groups = sorted(self.all_data[self.col_group].unique())

        self.pv = None
        self.forecaster = None

        self.mean_predictions = None
        self.simulated_data = None
        self.draws = None
        self.draw_models = None

    def run(self, n_draws, prediction_times, cv_lower_threshold,
            smoothed_radius, num_smooths, exclude_groups, exclude_below=0, cv_upper_threshold=np.inf,
            max_last=None, exp_smoothing=None):
        """
        Runs the whole model with PV and forecasting residuals and creating draws.

        Args:
            n_draws: (int) number of draws to produce
            prediction_times: (np.array) array of times to make predictions at
            cv_lower_threshold: (float) lower bound on the coefficient of variation
                for the residuals simulation
            cv_upper_threshold: (optional float) upper bound on the coefficient of variation
                for the residuals simulation
            smoothed_radius: List[int] residual smoothing before running the
                residual forecast -- how many neighbors to look at, e.g. [3, 3]
                would smooth over a radius of 3
            num_smooths: (int) number of iterations to run through for smoothing residuals
            exclude_groups: List[str] list of your group names that you want to exclude
                from the residual analysis. should be Wuhan
            exclude_below: (int) exclude results from the predictive validity analysis
                that had less than this many data points -- just for going into the regression
                to predict the coefficient of variation (low numbers of data points makes this unstable)
            exp_smoothing: (optional float) exponential smoothing parameter for combining time series predictions
            max_last: (optional int) number of models from previous observations to use since the maximum time
        Returns:
        """
        assert type(n_draws) == int
        assert type(cv_lower_threshold) == float
        assert type(cv_upper_threshold) == float
        assert type(smoothed_radius) == list
        assert type(num_smooths) == int
        assert type(exclude_below) == int
        assert type(exclude_groups) == list
        if exp_smoothing is not None:
            assert type(exp_smoothing) == float
        if max_last is not None:
            assert type(max_last) == int

        # Setup the initial model (optional for some subclasses)
        self.run_init_model()

        # Run predictive validity with a theta = 1, means everything is in relative space
        # -- relative mean bias, relative standard deviation (coefficient of variation)
        self.run_predictive_validity(theta=self.theta)

        # Excludes Wuhan from the residual fitting.
        # Right now only std_covariates are used.
        self.fit_residuals(
            smoothed_radius=smoothed_radius,
            num_smooths=num_smooths,
            exclude_below=exclude_below,
            covariates=['far_out', 'num_data'],
            exclude_groups=exclude_groups
        )

        # Create draws. Access them in self.draws by location.
        self.create_draws(
            num_draws=n_draws,
            std_lower_threshold=cv_lower_threshold,
            std_upper_threshold=cv_upper_threshold,
            prediction_times=prediction_times,
            theta=self.theta,
            exp_smoothing=exp_smoothing,
            max_last=max_last
        )

    def setup_pipeline(self):
        """
        Sets up the pipeline for running predictive validity and forecasting data out.
        Should be run at the end of the inheriting class' init so that the self.generate()
        gets the model settings to be run for all models.
        """
        self.pv = PVModel(
            data=self.all_data,
            col_t=self.col_t,
            col_group=self.col_group,
            col_obs=self.col_obs,
            col_obs_compare=self.col_obs_compare,
            predict_space=self.predict_space,
            model_generator=self.generate()
        )
        self.forecaster = Forecaster()

    def run_init_model(self):
        """
        Runs the model that doesn't need to be run multiple times.
        """
        self.refresh()

    def refresh(self):
        """
        Clear the current model results.
        """
        pass

    def generate(self):
        """
        Generate a copy of this class.
        """
        return deepcopy(self)

    def fit(self, df, group=None):
        """
        Function to fit the model with a given data frame.
        Args:
            df: (pd.DataFrame)
            group: (str) optional group to use in whatever capacity is needed for calling this function
        """
        pass

    def predict(self, times, predict_space, predict_group):
        """
        Function to create predictions based on the model fit.
        Args:
            times: (np.array) of times to predict at
            predict_space: (callable) curvefit.functions function to predict in that space
            predict_group: which group to make predictions for
        """
        pass

    def run_predictive_validity(self, theta):
        """
        Run predictive validity for the full model.

        Args:
            theta: amount of scaling for residuals relative to prediction.
        """
        self.pv.run_pv(theta=theta)

    def fit_residuals(self, smoothed_radius, num_smooths, covariates,
                      exclude_below, exclude_groups):
        """
        Fits residuals given a smoothed radius, and some models to exclude.
        Exclude below excludes models with less than that many data points.
        Exclude groups excludes all models from the list of groups regardless of the data points.

        Args:
            smoothed_radius: List[int] 2-element list of amount of smoothing for the residuals
            num_smooths: (int) how many smoothing iterations to go through
            covariates: List[str] which covariates to use to predict the residuals
            exclude_groups: List[str] which groups to exclude from the residual analysis
            exclude_below: (int) observations with less than exclude_below
                will be excluded from the analysis
        """
        residual_data = self.pv.all_residuals.copy()
        residual_data = residual_data.loc[residual_data['num_data'] > exclude_below].copy()
        residual_data = residual_data.loc[~residual_data[self.col_group].isin(exclude_groups)].copy()

        self.forecaster.fit_residuals(
            smooth_radius=smoothed_radius,
            residual_data=residual_data,
            col='residual',
            covariates=covariates,
            residual_model_type='local',
            num_smooths=num_smooths
        )

    def create_draws(self, num_draws, prediction_times,
                     theta=1, std_lower_threshold=1e-2, std_upper_threshold=np.inf,
                     exp_smoothing=None, max_last=None):
        """
        Generate draws for a model pipeline, smoothing over a neighbor radius of residuals
        for far out and num data points.

        Args:
            num_draws: (int) the number of draws to take
            prediction_times: (int) which times to produce final predictions (draws) at
            std_lower_threshold: (float) floor for standard deviation
            std_upper_threshold: (float) ceiling for standard deviation
            theta: (float) between 0 and 1, how much scaling of the residuals to do relative to the prediction mean
            exp_smoothing: (optional float) amount of exponential smoothing --> higher value means more weight
                given to the more recent models
            max_last: (optional int) number of models from previous observations to use since the maximum time
        """
        if self.pv.all_residuals is None:
            raise RuntimeError("Need to first run predictive validity with self.run_predictive_validity.")

        if max_last is not None and exp_smoothing is None:
            raise RuntimeError("If you pass a max last, you must pass an exponential smoothing parameter.")

        # Get the best fit we can
        self.fit(df=self.all_data)

        self.mean_predictions = {}
        self.draws = {}

        for group in self.groups:
            # Get the mean prediction for each group
            if exp_smoothing is None:
                self.mean_predictions[group] = self.predict(
                    times=prediction_times, predict_space=self.predict_space, predict_group=group
                )
            else:
                # will average the predictions across the last max_last forecast models
                # based on dropping data up to the max_last[i] time point
                self.mean_predictions[group] = self.pv.pv_groups[group].exp_smooth_preds(
                    prediction_times=prediction_times, exp_smoothing=exp_smoothing, max_last=max_last
                )

        # Loop through each group, forecasting the residuals and making draws
        for group in self.groups:
            draws = self.forecaster.simulate(
                mp=self,
                num_simulations=num_draws,
                prediction_times=prediction_times,
                group=group,
                theta=theta,
                std_floor=std_lower_threshold,
                std_ceiling=std_upper_threshold
            )
            if self.de_bias_draws:
                draws = draws - draws.var(axis=0) / 2

            self.draws[group] = draws

        return self

    def get_cv_matrices(self):
        """
        Get matrices of coefficient of variation by number of data points in the model
        by how far out the model is predicting from the last data point.

        Returns:
            (pd.DataFrame)
        """
        return self.forecaster.residual_model.smoothed.pivot('num_data', 'far_out', 'residual_std')

    def plot_results(self, prediction_times, sharex=True, sharey=False, draw_space=None,
                     plot_obs=None, plot_uncertainty=True):
        """
        Plot the draws resulting from a model in any space. Does it for each group in the model.

        Args:
            prediction_times: (np.array) of prediction times for the model
            sharex: (bool) fix x-axis across plots
            sharey: (bool) fix y-axis across plots
            draw_space: (callable) curvefit.functions what space to plot draws in
            plot_obs: (optional str) name of column that represents data in draw_space
            plot_uncertainty: (optional bool) whether to plot uncertainty intervals
        """
        if draw_space is None:
            draw_space = self.predict_space
        if plot_obs is None:
            plot_obs = self.col_obs_compare

        plot_fits(generator=self, sharex=sharex, sharey=sharey, prediction_times=prediction_times,
                  draw_space=draw_space, plot_obs=plot_obs, plot_uncertainty=plot_uncertainty)
