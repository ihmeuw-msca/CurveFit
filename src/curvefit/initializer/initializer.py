from curvefit.initializer.initializer_component import PriorInitializerComponent


class PriorInitializer:
    """
    {begin_markdown PriorInitializer}
    {spell_markdown initialization, initializer}

    # `curvefit.initializer.initializer.PriorInitializer`
    ## A prior initializer for a parameter set

    For a given solver, model and parameter set there are a number of strategies to get
    information to inform a prior for a later model fit. A `PriorInitializer` uses one or more
    `PriorInitializerComponent`s to inform what priors should be for a new parameter set. Typically,
    the data passed to `PriorInitializer` is only a subset of the data, for example a subset that
    has a substantial number of data points that result in well-informed parameter estimates for all
    parameters in a model. The updated parameter set that results from PriorInitializer.initialize()
    has priors that are informed by whatever `PriorInitializerComponent`s were passed in.

    These PriorInitializerComponents are **model-specification-specific**. You need to be aware of what
    component you are passing in and that it matches your model_prototype. If it does not, an error will be thrown.
    The requirements are listed in the description
    sections of each [`PriorInitializerComponent`](PriorInitializerComponent.md).

    ## Arguments

    - `self.prior_initializer_components (List[PriorInitializerComponent])` a list of prior initializer
        components (instantiated) to use in updating the parameter set

    ## Attributes

    - `self.component_types (List[str])`: list of the types of initializer components
    - `self.joint_solver (curvefit.solvers.solver.Solver)`: a solver/model run on all of the data
    - `self.individual_solvers (List[curvefit.solvers.solver.Solver]): a list of solver/models run on each
        group in the data individually

    ## Methods

    ### `initialize`
    For a given data, model prototype and solver prototype, run the prior initialization for all
    prior initializer components and return an updated parameter set.

    - `data (curvefit.core.data.Data)`: a Data object that represents all of the data that will be
        used in the initialization (this will often be a subset of all available data)
    - `model_prototype (curvefit.core.model.CoreModel)`: a model that will be used as the prototype
        and cloned for the joint and individual fits
    - `solver_prototype (curvefit.solvers.solver.Solver): a solver that will be used as the prototype
        and cloned for the joint and individual fits

    ## Usage

    ```python
    prior_initializer = PriorInitializer([LnAlphaBetaPrior(), BetaPrior()])
    new_parameter_set = prior_initializer.initialize(
        data=data, model_prototype=model_prototype,
        solver_prototype=solver_prototype, parameter_set_prototype
    )
    ```

    {end_markdown PriorInitializer}
    """
    def __init__(self, prior_initializer_components):

        self.prior_initializer_components = prior_initializer_components
        self.component_types = [c.component_type for c in self.prior_initializer_components]

        self.joint_solver = None
        self.individual_solvers = None

    @staticmethod
    def _run_joint(data, model_prototype, solver_prototype):
        model = model_prototype.clone()
        solver = solver_prototype.clone()

        solver.set_model_instance(model)
        solver.fit(data=data._get_df(return_specs=True))
        return solver

    @staticmethod
    def _run_individual(data, model_prototype, solver_prototype):
        solvers = list()

        for group in data.groups:
            model = model_prototype.clone()
            solver = solver_prototype.clone()

            group_data = data._get_df(group=group, copy=False, return_specs=True)

            solver.set_model_instance(model)
            solver.fit(data=group_data)

            solvers.append(solver)

        return solvers

    def initialize(self, data, model_prototype, solver_prototype):
        print("Running prior initializer.")

        if 'individual' in self.component_types:
            self.individual_solvers = self._run_individual(
                data, model_prototype=model_prototype,
                solver_prototype=solver_prototype
            )
        if 'joint' in self.component_types:
            self.joint_solver = self._run_joint(
                data, model_prototype=model_prototype,
                solver_prototype=solver_prototype
            )

        param_set = model_prototype.param_set.clone()

        for component in self.prior_initializer_components:
            if component.component_type == 'joint':
                param_set = component._update_parameter_set(
                    solver=self.joint_solver,
                    parameter_set_prototype=param_set
                )
            elif component.component_type == 'individual':
                param_set = component._update_parameter_set(
                    solver=self.individual_solvers,
                    parameter_set_prototype=param_set
                )
            else:
                raise RuntimeError(f"Cannot find component type {component.component_type}")

        return param_set
