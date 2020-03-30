"""
All classes of model generators should have a model_function that takes arguments
df and times and returns predictions at those times.

**NOTE**: This is useful for the predictive validity functions that need a fit_model
function that takes those arguments. That callable will be generated with the model_function in these classes.
"""

from copy import deepcopy
from curvefit.model import CurveModel
from curvefit.utils import convex_combination, model_average


class ModelPipeline:
    """
    Base class for a model generator.
    If a model needs to have initial parameters started for the predictive validity,
    put that in run_init_model
    """
    def __init__(self):
        pass

    def run_init_model(self):
        """
        Runs the model that doesn't need to be run multiple times.
        """
        self.refresh()
        pass

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

    def fit(self, df):
        """
        Function to fit the model with a given data frame.
        Args:
            df: (pd.DataFrame)
        """
        pass

    def predict(self, times, predict_space, predict_group='all'):
        """
        Function to create predictions based on the model fit.
        Args:
            times: (np.array) of times to predict at
            predict_space: (callable) curvefit.functions function to predict in that space
            predict_group: which group to make predictions for
        """
        pass


class BasicModel(ModelPipeline):
    def __init__(self, fit_dict, **basic_model_kwargs):
        """
        Generic class for a function to produce predictions from a model
        with the following attributes.

        Args:
            predict_group: (str) which group to make predictions for
            fit_dict: keyword arguments to CurveModel.fit_params()
            **basic_model_kwargs: keyword arguments to CurveModel.__init__()
        """
        super().__init__()
        self.fit_dict = fit_dict
        self.basic_model_kwargs = basic_model_kwargs
        self.mod = None

    def refresh(self):
        self.mod = None

    def fit(self, df):
        self.mod = CurveModel(df=df, **self.basic_model_kwargs)
        self.mod.fit_params(**self.fit_dict)

    def predict(self, times, predict_space, predict_group='all'):
        predictions = self.mod.predict(
            t=times, group_name=predict_group,
            prediction_functional_form=predict_space
        )
        return predictions


class TightLooseBetaPModel(ModelPipeline):
    def __init__(self, loose_beta_fit_dict, tight_beta_fit_dict,
                 loose_p_fit_dict, tight_p_fit_dict,
                 beta_model_extras=None, p_model_extras=None,
                 blend_start_t=2, blend_end_t=30,
                 **basic_model_kwargs):
        """
        Produces two tight-loose models as a convex combination between the two of them,
        and then averages

        Args:
            loose_beta_fit_dict: dictionary of keyword arguments to CurveModel.fit_params() for the loose beta model
            tight_beta_fit_dict: dictionary of keyword arguments to CurveModel.fit_params() fro the tight beta model
            loose_p_fit_dict: dictionary of keyword arguments to CurveModel.fit_params() for the loose p model
            tight_p_fit_dict: dictionary of keyword arguments to CurveModel.fit_params() fro the tight p model
            beta_model_extras: (optional) dictionary of keyword arguments to
                override the basic_model_kwargs for the beta model
            p_model_extras: (optional) dictionary of keyword arguments to
                override the basic_model_kwargs for the p model
            blend_start_t: (int) the time to start blending tight and loose
            blend_end_t: (int) the time to stop blending tight and loose

            predict_group: (str) which group to make predictions for
            **basic_model_kwargs: keyword arguments to the basic model
        """
        super().__init__()
        self.beta_model_kwargs = basic_model_kwargs
        self.p_model_kwargs = basic_model_kwargs

        if beta_model_extras is not None:
            self.beta_model_kwargs = self.beta_model_kwargs.update(beta_model_extras)

        if p_model_extras is not None:
            self.p_model_kwargs = self.p_model_kwargs.update(p_model_extras)

        self.loose_beta_fit_dict = loose_beta_fit_dict
        self.tight_beta_fit_dict = tight_beta_fit_dict
        self.loose_p_fit_dict = loose_p_fit_dict
        self.tight_p_fit_dict = tight_p_fit_dict

        self.blend_start_t = blend_start_t
        self.blend_end_t = blend_end_t

        self.loose_beta_model = None
        self.tight_beta_model = None
        self.loose_p_model = None
        self.tight_p_model = None

    def refresh(self):
        self.loose_beta_model = None
        self.tight_beta_model = None
        self.loose_p_model = None
        self.tight_p_model = None

    def fit(self, df):
        self.loose_beta_model = CurveModel(df=df, **self.beta_model_kwargs)
        self.tight_beta_model = CurveModel(df=df, **self.beta_model_kwargs)
        self.loose_p_model = CurveModel(df=df, **self.p_model_kwargs)
        self.tight_p_model = CurveModel(df=df, **self.p_model_kwargs)

        self.loose_beta_model.fit_params(**self.loose_beta_fit_dict)
        self.tight_beta_model.fit_params(**self.tight_beta_fit_dict)
        self.loose_p_model.fit_params(**self.loose_p_fit_dict)
        self.tight_p_model.fit_params(**self.tight_p_fit_dict)

    def predict(self, times, predict_space, predict_group='all'):
        loose_beta_predictions = self.loose_beta_model.predict(
            t=times, group_name=predict_group,
            prediction_functional_form=predict_space
        )
        tight_beta_predictions = self.tight_beta_model.predict(
            t=times, group_name=predict_group,
            prediction_functional_form=predict_space
        )
        loose_p_predictions = self.loose_p_model.predict(
            t=times, group_name=predict_group,
            prediction_functional_form=predict_space
        )
        tight_p_predictions = self.tight_p_model.predict(
            t=times, group_name=predict_group,
            prediction_functional_form=predict_space
        )
        beta_predictions = convex_combination(
            t=times, pred1=tight_beta_predictions, pred2=loose_beta_predictions,
            pred_fun=predict_space, start_day=self.blend_start_t, end_day=self.blend_end_t
        )
        p_predictions = convex_combination(
            t=times, pred1=tight_p_predictions, pred2=loose_p_predictions,
            pred_fun=predict_space, start_day=self.blend_start_t, end_day=self.blend_end_t
        )
        averaged_predictions = model_average(
            pred1=beta_predictions, pred2=p_predictions,
            w1=0.5, w2=0.5, pred_fun=predict_space
        )
        return averaged_predictions
