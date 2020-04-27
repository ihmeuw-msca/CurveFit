import numpy as np
from typing import List

from curvefit.solvers.solvers import Solver
from curvefit.core.effects2params import unzip_x


class PriorInitializerComponent:
    """
    {begin_markdown PriorInitializerComponent}

    # `curvefit.initializer.initializer.PriorInitializerComponent`
    ## A component of a prior initializer for a model

    For a given solver, model and parameter set there are a number of strategies to get
    information to inform a prior for a later model fit. This base class is used to build
    prior initializer components, that is, objects that extract information from models that have already been run
    to get well-informed priors for later runs.

    Prior initializer components can have two types: priors that are extracted from the analysis of a joint model run
    (one model run with > 1 group), and priors that are extracted from the analysis of a bunch of
    individual models runs (many models runs with 1 group each). See the documentation on
    [`JointPriorInitializerComponent`](JointPriorInitializerComponent.md) and
    [`IndividualPriorInitializerComponent`](IndividualPriorInitializerComponent.md).

    ## Attributes

    - `self.component_type (str, None)`: type of component, overridden in subclasses ("joint" or "individual")

    ## Methods

    ### `_extract_prior`
    A method to extract prior information from a solver or a list of solvers. Overridden in subclasses.

    - `solver (List[Solver], curvefit.core.solver.Solver)`: solver or list of
        solvers that have fit models to data using a parameter set

    ### `_update_parameter_set`
    A method to update a parameter set given information that was extracted about the priors from the solvers.

    - `solver (List[Solver], curvefit.core.solver.Solver)`: see above
    - `parameter_set_prototype (curvefit.core.parameter.ParameterSet)`: a parameter set that will be updated
        with new prior information

    {end_markdown PriorInitializerComponent}
    """
    def __init__(self):
        self.component_type = None

    def _extract_prior(self, solver):
        pass

    def _update_parameter_set(self, parameter_set_prototype, solver):
        pass


class JointPriorInitializerComponent(PriorInitializerComponent):
    """
    {begin_markdown JointPriorInitializerComponent}

    # `curvefit.initializer.initializer.JointPriorInitializerComponent`
    ## A joint prior initializer component for an initializer for a model

    See [`PriorInitializerComponent`](PriorInitializerComponent.md).

    ## Attributes

    - `self.component_type (str)`: "joint", which is the type of component

    ## Methods

    See methods for [`PriorInitializerComponent`](PriorInitializerComponent.md).

    {end_markdown PriorInitializerComponent}
    """
    def __init__(self):
        super().__init__()
        self.component_type = 'joint'


class IndividualPriorInitializerComponent(PriorInitializerComponent):
    """
    {begin_markdown IndividualPriorInitializerComponent}

    # `curvefit.initializer.initializer.IndividualPriorInitializerComponent`
    ## An individual prior initializer component for an initializer for a model

    See [`PriorInitializerComponent`](PriorInitializerComponent.md).

    ## Attributes

    - `self.component_type (str)`: "individual", which is the type of component

    ## Methods

    See methods for [`PriorInitializerComponent`](PriorInitializerComponent.md).

    {end_markdown IndividualPriorInitializerComponent}
    """
    def __init__(self):
        super().__init__()
        self.component_type = 'individual'


class LnAlphaBetaPrior(IndividualPriorInitializerComponent):
    """
    {begin_markdown LnAlphaBetaPrior}

    # `curvefit.initializer.initializer.LnAlphaBetaPrior`
    ## Gets information about a ln alpha-beta prior for model

    See [`PriorInitializerComponent`](PriorInitializerComponent.md) for a description of a prior initializer.
    This prior initializer component uses group-specific individual
    model fits with alpha and beta parameters to get information
    about what the mean and standard deviation for a functional prior on log(alpha * beta) should be. It uses
    the empirical mean and standard deviation of log(alpha * beta), typically fit on only a subset of groups
    that have more data and information about what alpha and beta should be.

    **NOTE: Requires 'alpha' and 'beta' parameters in a ParameterSet along with a functional prior
    called 'ln-alpha-beta'.**

    ## Attributes

    See attributes for [`IndividualPriorInitializerComponent`](IndividualPriorInitializerComponent.md).

    ## Methods

    See methods for [`IndividualPriorInitializerComponent`](IndividualPriorInitializerComponent.md).

    {end_markdown LnAlphaBetaPrior}
    """
    def __init__(self):
        super().__init__()

    def _extract_prior(self, solver):
        assert isinstance(solver, List)

        alphas = np.array([])
        betas = np.array([])

        for sol in solver:

            alpha_idx = sol.model.param_set.get_param_index('alpha')
            beta_idx = sol.model.param_set.get_param_index('beta')

            params = sol.model.get_params(x=sol.x_opt)

            alphas = alphas.append(params[alpha_idx, 0])
            betas = betas.append(params[beta_idx, 0])

        prior_mean = np.log(alphas * betas).mean()
        prior_std = np.log(alphas * betas).std()

        return [prior_mean, prior_std]

    def _update_parameter_set(self, parameter_set_prototype, solver):
        param_set = parameter_set_prototype.clone()

        # Check that the alpha-beta parameter exists
        param_function_index = param_set.get_param_function_index('ln-alpha-beta')

        # Extract the ln-alpha-beta prior
        prior = self._extract_prior(solver=solver)
        param_set.param_function_fe_gprior[param_function_index] = prior
        param_set.__post__init()

        return param_set


class BetaPrior(JointPriorInitializerComponent):
    """
    {begin_markdown BetaPrior}

    # `curvefit.initializer.initializer.BetaPrior`
    ## Gets information about a beta prior for a model

    See [`PriorInitializerComponent`](PriorInitializerComponent.md) for a description of a prior initializer.
    This prior initializer component uses one joint model fit that has beta as a parameter. Typically this
    prior is used only on a subset of groups that have more information about what beta should be.
    This prior initializer component uses the fixed effect mean as the new prior mean and the standard
    deviation of the random effects as the prior standard deviation for beta.

    **NOTE: Requires a 'beta' parameter in a ParameterSet.**

    ## Attributes

    See attributes for [`IndividualPriorInitializerComponent`](IndividualPriorInitializerComponent.md).

    ## Methods

    See methods for [`IndividualPriorInitializerComponent`](IndividualPriorInitializerComponent.md).

    {end_markdown BetaPrior}
    """
    def __init__(self):
        super().__init__()

    def _extract_prior(self, solver):
        assert isinstance(solver, Solver)

        fe, re = unzip_x(
            x=solver.x_opt,
            num_groups=solver.model.data_inputs.group_sizes,
            num_fe=solver.model.param_set.num_fe
        )

        beta_idx = solver.model.param_set.get_param_index('beta')

        beta_fe_mean = fe[beta_idx]
        beta_fe_std = np.std(re[:, beta_idx])

        return [beta_fe_mean, beta_fe_std]

    def _update_parameter_set(self, parameter_set_prototype, solvers):
        param_set = parameter_set_prototype.clone()

        # Check that the beta parameter exists and only has one variable
        param_index = param_set.get_param_index(param_name='beta')
        assert len(param_set.fe_gprior[param_index]) == 1

        # Extract the beta prior from the individual solvers
        prior = self._extract_prior(solver=solvers)
        param_set.fe_gprior[param_index] = prior
        param_set.__post__init()

        return param_set


class PriorInitializer:
    """
    {begin_markdown PriorInitializer}

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
    component you are passing in and that it matches your model_prototype. If it doesn't, an error will be thrown.
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
        solver.fit(data=data)
        return solver

    @staticmethod
    def _run_individual(data, model_prototype, solver_prototype):
        solvers = list()

        for group in data.groups:
            model = model_prototype.clone()
            solver = solver_prototype.clone()

            group_data = data.get_df(group=group, copy=False, return_specs=True)

            solver.set_model_instance(model)
            solver.fit(data=group_data)

            solvers.append(solver)

        return solvers

    def initialize(self, data, model_prototype, solver_prototype):

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

        return param_set





