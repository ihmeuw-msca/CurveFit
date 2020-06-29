import pandas as pd
from tqdm import tqdm

from curvefit.uncertainty.residuals import Residuals, ResidualInfo


class PredictiveValidity:
    """
    {begin_markdown PredictiveValidity}
    {spell_markdown metadata debug}

    # `curvefit.uncertainty.predictive_validity`
    ## Out of sample predictive validity

    Runs out of sample predictive validity for all groups in the data
    with the specified model and solver. Uses a [`Residuals`](Residuals.md)
    object for each group to store predictions and compute out of sample residuals at each time point.

    ## Arguments

    - `evaluation_space (Callable)`: the space in which to do predictive validity,
        can be different than the space that you're fitting the model in
    - `debug_mode (bool)`: whether or not to store all of the model data and results
        used for every single group-specific model (saves a lot of memory when `debug_mode = False`)

    ## Attributes

    - `theta (int)`: the amount of scaling for the residuals relative to the predictions.
        If the evaluation space is in log space, then `theta` is set to `0.` so that they
        are absolute residuals. If the evaluation space is in linear space, then `theta`
        is set to `1.` so that they are relative residuals to the prediction magnitude.
    - `group_residuals (Dict[str: curvefit.uncertainty.residuals.Residual]): residual objects
        for each group in the dataset
    - `group_records (Dict[str: List[curvefit.solvers.solver.Solver]]): a list of solvers
        at each time point in the predictive validity analysis for a particular group

    ## Methods

    ### `_make_group_info`
    Creates a [ResidualInfo](ResidualInfo.md) object based on the data passed in and
    for a particular group.

    - `data (curvefit.core.data.DataSpecs)`: data specifications
    - `group_name (str)`: the name of the group

    ### `_run_group_pv`
    Runs the predictive validity analysis for only one group using the `data`, `model`
    and `solver` from the `run_predictive_validity()` function.

    ### `run_predictive_validity`
    Runs predictive validity for all groups in the data.

    - `data (curvefit.core.data.Data)`: data and specifications for the whole analysis
    - `group (str)`: name of the group
    - `model (curvefit.models.base.Model)`: a model object that may be copied depending
        on whether or not in debug mode
    - `solver (curvefit.solvers.solver.Solver)`: a solver used to fit the model

    ### `get_residual_data`
    Return the out of sample residuals for all groups from the data argument to
    `run_predictive_validity`. Important input to the [`_ResidualModel`](_ResidualModel.md) for
    eventually creating uncertainty.

    {end_markdown PredictiveValidity}
    """
    def __init__(self, evaluation_space, debug_mode=False):

        assert callable(evaluation_space)
        assert type(debug_mode) == bool

        self.evaluation_space = evaluation_space
        self.debug_mode = debug_mode

        if self.evaluation_space.__name__.startswith('ln'):
            self.theta = 0.
        else:
            self.theta = 1.

        self.group_residuals = dict()
        self.group_records = dict()  # not filled if not in debug_mode

    @staticmethod
    def _make_group_info(data, group_name):
        df, specs = data._get_df(group=group_name, copy=False, return_specs=True)

        return ResidualInfo(
            group_name=group_name,
            times=df[specs.col_t].values,
            obs=df[specs.col_obs].values
        )

    def _run_group_pv(self, data, group, model, solver):

        if self.debug_mode:
            self.group_records[group] = []

        df, specs = data._get_df(group=group, return_specs=True)
        for i, t in tqdm(enumerate(self.group_residuals[group].residual_info.times)):

            if self.debug_mode:
                model = model.clone()
                solver = solver.clone()
                solver.set_model_instance(model)

            df_i = df.query(f"{data.data_specs.col_t} <= {t}")
            solver.fit(data=(df_i, specs))

            if self.debug_mode:
                self.group_records[group].append(solver)

            self.group_residuals[group]._record_predictions(
                i, solver.predict(
                    t=self.group_residuals[group].residual_info.times,
                    predict_fun=self.evaluation_space
                )
            )
        self.group_residuals[group]._compute_residuals(
            obs=data._get_translated_observations(group=group, space=self.evaluation_space),
            theta=self.theta
        )

    def run_predictive_validity(self, data, model_prototype, solver_prototype):
        print("Running predictive validity.")

        model = model_prototype.clone()
        solver = solver_prototype.clone()
        solver.set_model_instance(model)

        for group in data.groups:
            self.group_residuals[group] = Residuals(
                residual_info=self._make_group_info(data=data, group_name=group),
                data_specs=data.data_specs
            )

            self._run_group_pv(data=data, group=group,
                               model=model, solver=solver)

    def get_residual_data(self):
        dfs = []
        for group, residuals in self.group_residuals.items():
            dfs.append(residuals._residual_df())
        return pd.concat(dfs).reset_index()
