"""
A Basic Model, and a Basic Model with initialization of parameters.
"""

from copy import deepcopy
from curvefit.core.model import CurveModel
from curvefit.pipelines._pipeline import ModelPipeline
from curvefit.core.utils import get_initial_params
from curvefit.core.utils import compute_starting_params


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
