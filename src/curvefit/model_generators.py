"""
All classes of model generators should have a model_function that takes arguments
df and times and returns predictions at those times.

**NOTE**: This is useful for the predictive validity functions that need a fit_model
function that takes those arguments. That callable will be generated with the model_function in these classes.
"""

from copy import deepcopy
import pandas as pd

from curvefit.model import CurveModel
from curvefit.forecaster import Forecaster
from curvefit.pv import PVModel
from curvefit.utils import convex_combination, model_average
from curvefit.utils import get_initial_params
from curvefit.utils import compute_starting_params
from curvefit.diagnostics import plot_draws


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
        print("Generating.")
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

    def fit_residuals(self, smoothed_radius, covariates, exclude_below, exclude_groups):
        """
        Fits residuals given a smoothed radius, and some models to exclude.
        Exclude below excludes models with less than that many data points.
        Exclude groups excludes all models from the list of groups regardless of the data points.

        Args:
            smoothed_radius: List[int] 2-element list of amount of smoothing for the residuals
            covariates: List[str] which covariates to use to predict the residuals
                choices of num_data, far_out, and data_index (where data_index = far_out + num_data)
            exclude_groups: List[str] which groups to exclude from the residual analysis
            exclude_below: (int) observations with less than exclude_below
                will be excluded from the analysis

        Returns:

        """
        residual_data = self.pv.get_smoothed_residuals(radius=smoothed_radius)
        residual_data = residual_data.loc[residual_data['num_data'] > exclude_below].copy()
        residual_data = residual_data.loc[~residual_data[self.col_group].isin(exclude_groups)].copy()

        self.forecaster.fit_residuals(
            residual_data=residual_data,
            mean_col='residual_mean',
            std_col='residual_std',
            residual_covariates=covariates,
            residual_model_type='linear'
        )

    def create_draws(self, num_draws, num_forecast_out, prediction_times,
                     theta=1, std_threshold=1e-2):
        """
        Generate draws for a model pipeline, smoothing over a neighbor radius of residuals
        for far out and num data points.

        Args:

            num_draws: (int) the number of draws to take
            num_forecast_out: (int) how far out into the future should residual simulations be taken
            prediction_times: (int) which times to produce final predictions at
            std_threshold: (float) floor for standard deviation
            theta: (float) between 0 and 1, how much scaling of the residuals to do relative to the prediction mean
        """
        if self.pv.all_residuals is None:
            raise RuntimeError("Need to first run predictive validity with self.run_predictive_validity.")

        generator = self.generate()

        self.mean_predictions = {}
        self.simulated_data = {}
        self.draws = {}

        self.fit(df=self.all_data)

        for group in self.groups:
            sims = self.forecaster.simulate(
                mp=self,
                far_out=num_forecast_out,
                num_simulations=num_draws,
                group=group,
                theta=theta,
                epsilon=std_threshold
            )
            self.simulated_data[group] = sims
            self.mean_predictions[group] = self.predict(
                times=prediction_times, predict_space=self.predict_space, predict_group=group
            )

        for group in self.groups:
            self.draws[group] = []

        for i in range(num_draws):
            new_data = []

            for group in self.groups:
                new_data.append(self.simulated_data[group][i])
            new_data = pd.concat(new_data)

            print(f"Creating {i}th draw.", end='\r')
            generator.fit(df=new_data)

            for group in self.groups:
                predictions = generator.predict(
                    times=prediction_times,
                    predict_space=self.predict_space,
                    predict_group=group
                )
                self.draws[group].append(predictions)

        return self


class BasicModel(ModelPipeline):
    def __init__(self, fit_dict, basic_model_dict, **pipeline_kwargs):
        """
        Generic class for a function to produce predictions from a model
        with the following attributes.

        Args:
            **pipeline_kwargs: keyword arguments for the base class of ModelPipeline
            predict_group: (str) which group to make predictions for
            fit_dict: keyword arguments to CurveModel.fit_params()
            basic_model_dict: additional keyword arguments to the CurveModel class
                col_obs_se: (str) of observation standard error
                col_covs: List[str] list of names of covariates to put on the parameters
            param_names (list{str}):
                Names of the parameters in the specific functional form.
            link_fun (list{function}):
                List of link functions for each parameter.
            var_link_fun (list{function}):
                List of link functions for the variables including fixed effects
                and random effects.
        """
        super().__init__(**pipeline_kwargs)
        self.fit_dict = fit_dict
        self.basic_model_dict = basic_model_dict
        self.basic_model_dict.update({'col_obs_se': self.col_obs_se})

        generator_kwargs = pipeline_kwargs
        for arg in self.pop_cols:
            generator_kwargs.pop(arg)

        self.basic_model_dict.update(**generator_kwargs)
        self.mod = None

        self.setup_pipeline()

    def refresh(self):
        self.mod = None

    def fit(self, df, group=None):
        self.mod = CurveModel(df=df, **self.basic_model_dict)
        self.mod.fit_params(**self.fit_dict)

    def predict(self, times, predict_space, predict_group):
        predictions = self.mod.predict(
            t=times, group_name=predict_group,
            prediction_functional_form=predict_space
        )
        return predictions


class BasicModelWithInit(BasicModel):
    def __init__(self, smart_init_options=None, **kwargs):
        if smart_init_options is None:
            smart_init_options = {}
        self.smart_init_options = smart_init_options

        super().__init__(**kwargs)

        if self.fit_dict['options']:
            self.smart_init_options = {**self.fit_dict['options'],
                                       **self.smart_init_options}

        self.init_dict = None
        self.mod = None

    def run_init_model(self):
        self.init_dict = self.get_init_dict(df=self.all_data,
                                            groups=self.groups)

    def update_init_model(self, df, group):
        """
        Update the initial model with a re-fit model
        from the specified group. Returns a new copy of the init dict

        Args:
            df: (pd.DataFrame) data used to update the init model
            group: (str) the group to update

        Returns:

        """
        new_init_dict = deepcopy(self.init_dict)
        new_init_dict.update(self.get_init_dict(df=df, groups=[group]))
        return new_init_dict

    def get_init_dict(self, df, groups):
        """
        Run the init model for each location.

        Args:
            df: (pd.DataFrame) data frame to fit the model that will
                be subset by group
            groups: (str) groups to get in the dict

        Returns:
            (dict) dictionary of fixed effects keyed by group
        """
        model = CurveModel(df=df,
                           **self.basic_model_dict)

        init_fit_dict = deepcopy(self.fit_dict)
        init_fit_dict.update(options=self.smart_init_options)

        init_dict = get_initial_params(
            groups=groups,
            model=model,
            fit_arg_dict=init_fit_dict
        )
        return init_dict

    def fit(self, df, group=None):
        """
        Fits a loose, tight, beta, and p combinations model. If you pass in
        update group it will override the initial parameters with new
        initial parameters based on the df you pass.

        Args:
            df:
            group: (str) passing in the group will update the initialization
                dictionary (not replacing the old one) for this particular fit.

        Returns:

        """
        if group is not None:
            init_dict = self.update_init_model(df=df, group=group)
        else:
            init_dict = deepcopy(self.init_dict)

        fit_dict = deepcopy(self.fit_dict)
        fe_init, re_init = compute_starting_params(init_dict)
        fit_dict.update(fe_init=fe_init, re_init=re_init)

        self.mod = CurveModel(df=df, **self.basic_model_dict)
        self.mod.fit_params(**fit_dict)

    def refresh(self):
        self.mod = None


class TightLooseBetaPModel(ModelPipeline):
    def __init__(self, basic_fit_dict,
                 basic_model_dict, model_specific_dict,
                 loose_beta_fit=None, tight_beta_fit=None,
                 loose_p_fit=None, tight_p_fit=None,
                 beta_model_extras=None, p_model_extras=None,
                 **pipeline_kwargs):
        """
        Produces two tight-loose models as a convex combination between the two of them,
        and then averages

        Args:
            **pipeline_kwargs: keyword arguments for the base class of ModelPipeline

            basic_fit_dict: dictionary of keyword arguments to CurveModel.fit_params()
            loose_beta_fit: dictionary of keyword arguments to override basic_fit_dict for the loose beta model
            tight_beta_fit: dictionary of keyword arguments to override basic_fit_dict for the tight beta model
            loose_p_fit: dictionary of keyword arguments to override basic_fit_dict for the loose p model
            tight_p_fit: dictionary of keyword arguments to override basic_fit_dict fro the tight p model

            basic_model_dict: additional keyword arguments to the CurveModel class
                col_obs_se: (str) of observation standard error
                col_covs: List[str] list of names of covariates to put on the parameters
            model_specific_dict: additional keyword arguments specific to the TightLooseBetaPModel combo
                beta_weight: (float) weight for the beta model
                p_weight: (float) weight for the p model
                blend_start_t: (int) the time to start blending tight and loose
                blend_end_t: (int) the time to stop blending tight and loose
                smart_init_options: (dict) options for the smart initialization
            param_names (list{str}):
                Names of the parameters in the specific functional form.
            link_fun (list{function}):
                List of link functions for each parameter.
            var_link_fun (list{function}):
                List of link functions for the variables including fixed effects
                and random effects.
            beta_model_extras: (optional) dictionary of keyword arguments to
                override the basic_model_kwargs for the beta model
            p_model_extras: (optional) dictionary of keyword arguments to
                override the basic_model_kwargs for the p model
        """
        super().__init__(**pipeline_kwargs)
        generator_kwargs = pipeline_kwargs
        for arg in self.pop_cols:
            generator_kwargs.pop(arg)

        self.beta_weight = None
        self.p_weight = None
        self.blend_start_t = None
        self.blend_end_t = None
        self.smart_init_options = None

        for k, v in model_specific_dict.items():
            setattr(self, k, v)

        assert self.beta_weight + self.p_weight == 1

        self.basic_model_dict = basic_model_dict
        self.basic_model_dict.update(**generator_kwargs)
        self.basic_model_dict.update({'col_obs_se': self.col_obs_se})

        self.beta_model_kwargs = deepcopy(self.basic_model_dict)
        self.p_model_kwargs = deepcopy(self.basic_model_dict)

        if beta_model_extras is not None:
            self.beta_model_kwargs.update(beta_model_extras)

        if p_model_extras is not None:
            self.p_model_kwargs.update(p_model_extras)

        self.loose_beta_fit_dict = deepcopy(basic_fit_dict)
        self.tight_beta_fit_dict = deepcopy(basic_fit_dict)
        self.loose_p_fit_dict = deepcopy(basic_fit_dict)
        self.tight_p_fit_dict = deepcopy(basic_fit_dict)

        if loose_beta_fit is not None:
            self.loose_beta_fit_dict.update(loose_beta_fit)
        if tight_beta_fit is not None:
            self.tight_beta_fit_dict.update(tight_beta_fit)
        if loose_p_fit is not None:
            self.loose_p_fit_dict.update(loose_p_fit)
        if tight_p_fit is not None:
            self.tight_p_fit_dict.update(tight_p_fit)

        self.loose_beta_model = None
        self.tight_beta_model = None
        self.loose_p_model = None
        self.tight_p_model = None

        self.init_dict = dict()

        self.setup_pipeline()

    def run_init_model(self):
        self.init_dict = self.get_init_dict(df=self.all_data, groups=self.groups)

    def update_init_model(self, df, group):
        """
        Update the initial model with a re-fit model
        from the specified group. Returns a new copy of the init dict

        Args:
            df: (pd.DataFrame) data used to update the init model
            group: (str) the group to update

        Returns:

        """
        new_init_dict = deepcopy(self.init_dict)
        updated_group = self.get_init_dict(df=df, groups=[group])
        for param in ['beta', 'p']:
            for fit_type in ['loose', 'tight']:
                new_init_dict[param][fit_type].update(updated_group[param][fit_type])
        return new_init_dict

    def get_init_dict(self, df, groups):
        """
        Run the init model for each location.

        Args:
            df: (pd.DataFrame) data frame to fit the model that will
                be subset by group
            groups: (str) groups to get in the dict

        Returns:
            (dict) dictionary of fixed effects keyed by group
        """
        init_dict = {}
        for param in ['beta', 'p']:
            init_dict[param] = {}

            for fit_type in ['loose', 'tight']:
                model_arg_dict = deepcopy(getattr(self, f'{param}_model_kwargs'))
                model = CurveModel(df=df, **model_arg_dict)

                fit_arg_dict = deepcopy(getattr(self, f'{fit_type}_{param}_fit_dict'))
                fit_arg_dict.update(options=self.smart_init_options)

                init_dict[param][fit_type] = get_initial_params(
                    groups=groups,
                    model=model,
                    fit_arg_dict=fit_arg_dict
                )
        return init_dict

    def refresh(self):
        self.loose_beta_model = None
        self.tight_beta_model = None
        self.loose_p_model = None
        self.tight_p_model = None

    def fit(self, df, group=None):
        """
        Fits a loose, tight, beta, and p combinations model. If you pass in
        update group it will override the initial parameters with new
        initial parameters based on the df you pass.

        Args:
            df:
            group: (str) passing in the group will update the initialization
                dictionary (not replacing the old one) for this particular fit.

        Returns:

        """
        if group is not None:
            init_dict = self.update_init_model(df=df, group=group)
        else:
            init_dict = deepcopy(self.init_dict)

        for param in ['beta', 'p']:
            if getattr(self, f'{param}_weight') == 0:
                continue
            for fit_type in ['loose', 'tight']:
                model_arg_dict = deepcopy(getattr(self, f'{param}_model_kwargs'))
                fit_arg_dict = deepcopy(getattr(self, f'{fit_type}_{param}_fit_dict'))
                model = CurveModel(df=df, **model_arg_dict)

                fe_init, re_init = compute_starting_params(
                    init_dict[param][fit_type]
                )

                fit_arg_dict.update(fe_init=fe_init, re_init=re_init)
                model.fit_params(**fit_arg_dict)

                setattr(self, f'{fit_type}_{param}_model', model)

    def predict(self, times, predict_space, predict_group):
        beta_predictions = None
        p_predictions = None

        if self.beta_weight > 0:
            loose_beta_predictions = self.loose_beta_model.predict(
                t=times, group_name=predict_group,
                prediction_functional_form=predict_space
            )
            tight_beta_predictions = self.tight_beta_model.predict(
                t=times, group_name=predict_group,
                prediction_functional_form=predict_space
            )
            beta_predictions = convex_combination(
                t=times, pred1=tight_beta_predictions, pred2=loose_beta_predictions,
                pred_fun=predict_space, start_day=self.blend_start_t, end_day=self.blend_end_t
            )
        if self.p_weight > 0:
            loose_p_predictions = self.loose_p_model.predict(
                t=times, group_name=predict_group,
                prediction_functional_form=predict_space
            )
            tight_p_predictions = self.tight_p_model.predict(
                t=times, group_name=predict_group,
                prediction_functional_form=predict_space
            )
            p_predictions = convex_combination(
                t=times, pred1=tight_p_predictions, pred2=loose_p_predictions,
                pred_fun=predict_space, start_day=self.blend_start_t, end_day=self.blend_end_t
            )

        if (self.beta_weight > 0) & (self.p_weight > 0):
            averaged_predictions = model_average(
                pred1=beta_predictions, pred2=p_predictions,
                w1=self.beta_weight, w2=self.p_weight, pred_fun=predict_space
            )
        elif (self.beta_weight > 0) & (self.p_weight == 0):
            averaged_predictions = beta_predictions
        elif (self.beta_weight == 0) & (self.p_weight > 0):
            averaged_predictions = p_predictions
        else:
            raise RuntimeError
        return averaged_predictions

    def plot_draws(self, prediction_times, sharex, sharey):
        plot_draws(generator=self, sharex=sharex, sharey=sharey, prediction_times=prediction_times)

