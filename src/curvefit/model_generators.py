"""
All classes of model generators should have a model_function that takes arguments
df and times and returns predictions at those times.

**NOTE**: This is useful for the predictive validity functions that need a fit_model
function that takes those arguments. That callable will be generated with the model_function in these classes.
"""

from copy import deepcopy
from curvefit.model import CurveModel


class ModelGenerator:
    """
    Base class for a model generator.
    """
    def __init__(self):
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


class BasicModel(ModelGenerator):
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

    def fit(self, df):
        self.mod = CurveModel(df=df, **self.basic_model_kwargs)
        self.mod.fit_params(**self.fit_dict)

    def predict(self, times, predict_space, predict_group='all'):
        predictions = self.mod.predict(
            t=times, group_name=predict_group,
            prediction_functional_form=predict_space
        )
        return predictions


class TightLooseModel(ModelGenerator):
    def __init__(self, loose_fit_dict, tight_fit_dict,
                 **basic_model_kwargs):
        """
        Produces a tight-loose model as a convex combination between the two of them.

        Args:
            loose_fit_kwargs: dictionary of keyword arguments to CurveModel.fit_params() for the loose model
            tight_fit_kwargs: dictionary of keyword arguments to CurveModel.fit_params() fro the tight model

            predict_group: (str) which group to make predictions for
            **basic_model_kwargs: keyword arguments to the basic model
        """
        super().__init__()
        self.basic_model_kwargs = basic_model_kwargs
        self.loose_fit_dict = loose_fit_dict
        self.tight_fit_dict = tight_fit_dict

        self.tight_mod = None
        self.loose_mod = None

    def fit(self, df):
        self.tight_mod = CurveModel(df=df, **self.basic_model_kwargs)
        self.loose_mod = CurveModel(df=df, **self.basic_model_kwargs)

        self.tight_mod.fit_params(**self.tight_fit_dict)
        self.loose_mod.fit_params(**self.loose_fit_dict)

    def predict(self, times, predict_space, predict_group='all'):
        tight_predictions = self.tight_mod.predict(
            t=times, group_name=predict_group,
            prediction_functional_form=predict_space
        )
        loose_predictions = self.loose_mod.predict(
            t=times, group_name=predict_group,
            prediction_functional_form=predict_space
        )
        predictions = 0.5 * tight_predictions + 0.5 * loose_predictions
        return predictions
