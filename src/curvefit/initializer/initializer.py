import numpy as np
from typing import List

from curvefit.solvers.solvers import Solver
from curvefit.core.effects2params import effects2params


class PriorInitializerComponent:
    def __init__(self):
        self.initializer_type = None

    def _extract_prior(self, solver):
        pass

    def _update_parameter_set(self, parameter_set_prototype, solvers):
        pass


class JointPriorInitializerComponent(PriorInitializerComponent):
    def __init__(self):
        super().__init__()
        self.initializer_type = 'joint'


class IndividualPriorInitializerComponent(PriorInitializerComponent):
    def __init__(self):
        super().__init__()
        self.initializer_type = 'individual'


class LnAlphaBetaPrior(IndividualPriorInitializerComponent):
    def __init__(self):
        super().__init__()

    def _extract_prior(self, solver):
        assert isinstance(solver, List)
        pass

    def _update_parameter_set(self, parameter_set_prototype, solvers):
        param_set = parameter_set_prototype.clone()

        # Check that the alpha-beta parameter exists
        param_function_index = param_set.get_param_function_index('ln-alpha-beta')

        # Extract the ln-alpha-beta prior
        prior = self._extract_prior(solver=solvers)
        param_set.param_function_fe_gprior[param_function_index] = prior

        return param_set


class BetaPrior(JointPriorInitializerComponent):
    def __init__(self):
        super().__init__()

    def _extract_prior(self, solver):
        assert isinstance(solver, Solver)
        pass

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
    def __init__(self, prior_initializer_instances):

        self.prior_initializer_instances = prior_initializer_instances
        self.initializer_types = [c.initializer_type for c in self.prior_initializer_instances]

        self.joint_solver = None
        self.individual_solvers = None

    @staticmethod
    def run_joint(data, model_prototype, solver_prototype):
        model = model_prototype.clone()
        solver = solver_prototype.clone()

        solver.set_model_instance(model)
        solver.fit(data=data)
        return solver

    @staticmethod
    def run_individual(data, model_prototype, solver_prototype):
        solvers = {}

        for group in data.groups:
            model = model_prototype.clone()
            solver = solver_prototype.cone()

            group_data = data.get_df(group=group, copy=False, return_specs=True)

            solver.set_model_instance(model)
            solver.fit(data=group_data)

            solvers[group] = solver

        return solvers

    def initialize(self, data, model_prototype, solver_prototype, parameter_set_prototype):

        if 'individual' in self.initializer_types:
            self.individual_solvers = self.run_individual(
                data, model_prototype=model_prototype,
                solver_prototype=solver_prototype
            )
        if 'joint' in self.initializer_types:
            self.joint_solver = self.run_joint(
                data, model_prototype=model_prototype,
                solver_prototype=solver_prototype
            )

        param_set = parameter_set_prototype.clone()

        for initializer in self.prior_initializer_instances:
            if initializer.initializer_type == 'joint':
                param_set = initializer._update_parameter_set(
                    solvers=self.joint_solver,
                    parameter_set_prototype=param_set
                )
            elif initializer.initializer_type == 'individual':
                param_set = initializer._update_parameter_set(
                    solvers=self.individual_solvers,
                    parameter_set_prototype=param_set
                )

        return param_set





