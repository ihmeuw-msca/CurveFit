import numpy as np
from typing import List

from curvefit.solvers.solvers import Solver
from curvefit.core.effects2params import unzip_x


class PriorInitializerComponent:
    """
    {begin_markdown PriorInitializerComponent}
    {spell_markdown initializer}

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
    {spell_markdown initializer}

    # `curvefit.initializer.initializer.JointPriorInitializerComponent`
    ## A joint prior initializer component for an initializer for a model

    See [`PriorInitializerComponent`](PriorInitializerComponent.md).

    ## Attributes

    - `self.component_type (str)`: "joint", which is the type of component

    ## Methods

    See methods for [`PriorInitializerComponent`](PriorInitializerComponent.md).

    {end_markdown JointPriorInitializerComponent}
    """
    def __init__(self):
        super().__init__()
        self.component_type = 'joint'


class IndividualPriorInitializerComponent(PriorInitializerComponent):
    """
    {begin_markdown IndividualPriorInitializerComponent}
    {spell_markdown initializer}

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
    {spell_markdown initializer}

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

            model = sol.get_model_instance()
            params = model.get_params(x=sol.x_opt)

            alphas = np.append(alphas, params[alpha_idx, 0])
            betas = np.append(betas, params[beta_idx, 0])

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

        return param_set


class BetaPrior(JointPriorInitializerComponent):
    """
    {begin_markdown BetaPrior}
    {spell_markdown initializer}

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
            num_groups=solver.model.data_inputs.num_groups,
            num_fe=solver.model.param_set.num_fe
        )

        model = solver.get_model_instance()
        beta_idx = model.param_set.get_param_index('beta')

        beta_fe_mean = fe[beta_idx]
        beta_fe_std = np.std(re[:, beta_idx])

        return [beta_fe_mean, beta_fe_std]

    def _update_parameter_set(self, parameter_set_prototype, solver):
        param_set = parameter_set_prototype.clone()

        # Check that the beta parameter exists and only has one variable
        param_index = param_set.get_param_index(param_name='beta')
        assert len(param_set.fe_gprior[param_index]) == 1

        # Extract the beta prior from the individual solvers
        prior = self._extract_prior(solver=solver)
        param_set.fe_gprior[param_index] = prior

        return param_set

