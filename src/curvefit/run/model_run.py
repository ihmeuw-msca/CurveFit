import time


class ModelRunner:
    """
    {begin_markdown ModelRunner}
    {spell_markdown }

    # `curvefit.run.model_run.ModelRunner`
    ## Runs a full model from start to finish for all groups

    Runs a full model, including initialization for model priors,
    predictive validity, and creating draws.
    
    ## Syntax

    ## Arguments

    - `data (curvefit.core.data.Data)`: data object with all groups
    - `model (curvefit.models.base.Model)`: a model instance with an attached
        ParameterSet
    - `solver (curvefit.solvers.solver.Solver)`: a solver instance to fit the model
    - `predictive_validity (curvefit.uncertainty.predictive_validity.PredictiveValidity)`: a predictive validity object
    - `residual_model (curvefit.uncertainty.residual_models._ResidualModel)`: a residual model
    - `draws (curvefit.uncertainty.draws.Draws)`: a draws object
    - `prior_initializer (optional, curvefit.initializer.initializers.PriorInitializer)`: prior initializer
    - `initializer_data (optional, curvefit.core.data.Data)`: data for the initializer only,
        might be the same or different from `data` (could overlap, could be completely separate)


    ## Methods

    ### `run`
    The main run function for a `ModelRunner`. This function initializes the priors,
    it runs predictive validity, uses the results of predictive validity
    to estimate residuals, and creates draws by simulating them forwards in time.

    ## Examples


    {end_markdown ModelRunner}
    """
    def __init__(self, data, model, solver,
                 predictive_validity, residual_model, draws,
                 prior_initializer=None, initializer_data=None):

        self.data = data
        self.model = model

        self.solver = solver
        if not hasattr(self.model, 'param_set'):
            raise RuntimeError("The model instance must have a parameter set.")
        self.model.erase_data()
        self.solver.detach_model_instance()

        self.predictive_validity = predictive_validity
        self.residual_model = residual_model
        self.draws = draws

        self.prior_initializer = prior_initializer
        self.initializer_data = initializer_data

    def run(self):
        start = time.time()

        # Initialize the priors
        if self.prior_initializer is not None:
            if self.initializer_data is None:
                self.initializer_data = self.data
            self.model.parameter_set = self.prior_initializer.initialize(
                data=self.initializer_data,
                model_prototype=self.model,
                solver_prototype=self.solver
            )

        # Delete random effects from this point forward because all final models
        # are individual rather than joint models
        self.model.param_set = self.model.param_set.delete_random_effects()

        # Run predictive validity
        self.predictive_validity.run_predictive_validity(
            data=self.data,
            model_prototype=self.model,
            solver_prototype=self.solver
        )
        # Use the results of the predictive validity analysis to estimate residuals
        self.residual_model.fit_residuals(
            residual_df=self.predictive_validity.get_residual_data()
        )
        # Create draws by simulating forward residuals
        self.draws.create_draws(
            data=self.data,
            model_prototype=self.model,
            solver_prototype=self.solver,
            residual_model=self.residual_model,
            evaluation_space=self.predictive_validity.evaluation_space,
            theta=self.predictive_validity.theta
        )
        end = time.time()
        print(f"Elapsed time: {end - start}.")
