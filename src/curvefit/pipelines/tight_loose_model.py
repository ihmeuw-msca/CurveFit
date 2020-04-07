"""
A weighted combination of a beta-model and p-model, then convex combination between tight and loose predictions.
"""

from copy import deepcopy

from curvefit.core.model import CurveModel
from curvefit.pipelines._pipeline import ModelPipeline
from curvefit.core.utils import convex_combination, model_average
from curvefit.core.utils import get_initial_params
from curvefit.core.utils import compute_starting_params


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
