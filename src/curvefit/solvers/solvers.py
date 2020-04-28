import numpy as np
from copy import deepcopy
import scipy.optimize as sciopt
from dataclasses import dataclass

from curvefit.core.effects2params import effects2params
from curvefit.core.prototype import Prototype


class ModelNotDefinedError(Exception):
    pass


class SolverNotDefinedError(Exception):
    pass


class Solver(Prototype):

    def __init__(self, model_instance=None):
        self.model = model_instance
        self.x_opt = None
        self.fun_val_opt = None
        self.options = None

    def set_model_instance(self, model_instance):
        self.model = model_instance

    def detach_model_instance(self):
        self.model = None

    def get_model_instance(self):
        if self.model is not None:
            return self.model
        else:
            raise ModelNotDefinedError()

    def set_options(self, options: dict):
        self.options = options

    def fit(self, data, x_init=None, options=None):
        raise NotImplementedError()

    def predict(self, **kwargs):
        return self.model.predict(self.x_opt, **kwargs)


class ScipyOpt(Solver):

    def fit(self, data, x_init=None, options=None):
        self.model.convert_inputs(data)
        if x_init is None:
            x_init = self.model.x_init

        result = sciopt.minimize(
            fun=lambda x: self.model.objective(x, data),
            x0=x_init,
            jac=lambda x: self.model.gradient(x, data),
            bounds=self.model.bounds,
            options=options if options is not None else self.options,
        )

        self.x_opt = result.x
        self.fun_val_opt = result.fun


class CompositeSolver(Solver):

    def __init__(self, solver=None):
        super().__init__(model_instance=None)
        self.solver = ScipyOpt() if solver is None else solver

    def set_solver(self, solver):
        self.solver = solver

    def set_options(self, options: dict):
        if self.assert_solver_defined():
            self.solver.set_options(options)

    def set_model_instance(self, model_instance):
        if self.assert_solver_defined() is True:
            self.solver.set_model_instance(model_instance)

    def detach_model_instance(self):
        if self.solver is not None:
            self.solver.detach_model_instance()

    def get_model_instance(self):
        if self.assert_solver_defined() is True:
            return self.solver.get_model_instance()

    def assert_solver_defined(self):
        if self.solver is not None:
            return True
        else:
            raise SolverNotDefinedError()

    def predict(self, **kwargs):
        return self.solver.predict(**kwargs)


class MultipleInitializations(CompositeSolver):

    def __init__(self, sample_fun, solver=None):
        super().__init__()
        self.sample_fun = sample_fun

    def fit(self, data, x_init=None, options=None):
        if self.assert_solver_defined() is True:
            fun_vals = []
            xs_opt = []
            xs_init = self.sample_fun(x_init)
            for x in xs_init:
                self.solver.fit(data, x, options=options)
                fun_vals.append(self.solver.fun_val_opt)
                xs_opt.append(self.solver.x_opt)

            self.x_opt = xs_opt[np.argmin(fun_vals)]
            self.fun_val_opt = np.min(fun_vals)


class GaussianMixturesIntegration(CompositeSolver):

    def __init__(self, gm_model, solver=None):
        super().__init__()
        self.gm_model = gm_model

    def fit(self, data, x_init=None, options=None):
        if self.assert_solver_defined() is True:
            self.solver.fit(data, x_init, options)
            model = self.get_model_instance()
            params = effects2params(
                self.solver.x_opt,
                model.data_inputs.group_sizes,
                model.data_inputs.covariates_matrices,
                model.param_set.link_fun,
                model.data_inputs.var_link_fun,
            )
            self.gm_model.set_params(params[:, 0])
            gm_solver = ScipyOpt(self.gm_model)
            gm_solver.fit(data)
            self.x_opt = gm_solver.x_opt
            self.fun_val_opt = gm_solver.fun_val_opt

    def predict(self, t):
        return self.gm_model.predict(self.x_opt, t)


class SmartInitialization(CompositeSolver):

    def __init__(self, solver=None):
        super().__init__()

        self.x_mean = None

    def fit(self, data, x_init=None, options=None):
        if self.assert_solver_defined() is True:
            df = data[0]
            data_specs = data[1]
            group_names = df[data_specs.col_group].unique()
            if len(group_names) == 1:
                raise RuntimeError('SmartInitialization is only for multiple groups.')

            model = self.get_model_instance()
            re_bounds = deepcopy(model.param_set.re_bounds)
            model.param_set.re_bounds = self._set_bounds_zeros(model.param_set.re_bounds)
            xs = []
            for group in group_names:
                data_sub = (df[df[data_specs.col_group] == group], data_specs)
                self.solver.fit(data_sub, x_init, options)
                xs.append(self.solver.x_opt)
                model.erase_data()
            xs = np.array(xs)
            self.x_mean = np.mean(xs, axis=0)[:model.param_set.num_fe]
            x_init = np.concatenate((self.x_mean, np.reshape(xs[:, :model.param_set.num_fe] - self.x_mean, (-1,))))
            model.param_set.re_bounds = re_bounds
            self.solver.fit(data, x_init, options)
            self.x_opt = self.solver.x_opt
            self.fun_val_opt = self.solver.fun_val_opt

    def _set_bounds_zeros(self, bounds):
        if len(bounds) == 2 and isinstance(bounds[0], float):
            return [0.0, 0.0]
        else:
            bds = []
            for b in bounds:
                bds.append(self._set_bounds_zeros(b))
            return bds
