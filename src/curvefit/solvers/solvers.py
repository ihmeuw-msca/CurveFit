import numpy as np
from copy import deepcopy
import scipy.optimize as sciopt


class ModelNotDefinedError(Exception):
    pass


class SolverNotDefinedError(Exception):
    pass


class Solver:

    def __init__(self, model_instance=None):
        self.model = model_instance
        self.x_opt = None
        self.fun_val_opt = None

    def set_model_instance(self, model_instance):
        self.model = model_instance

    def detach_model_instance(self):
        self.model = None

    def get_model_instance(self):
        if self.model is not None:
            return self.model
        else:
            raise ModelNotDefinedError()

    def fit(self, data, x_init=None, options=None):
        raise NotImplementedError()

    def clone(self):
        return deepcopy(self)


class ScipyOpt(Solver):

    def fit(self, data, x_init=None, options=None):
        if x_init is None:
            x_init = self.model.x_init
        
        result = sciopt.minimize(
            fun=lambda x: self.model.objective(x, data), 
            x0=x_init, 
            jac=lambda x: self.model.gradient(x, data),
            bounds=self.model.bounds,
            options=options,
        )
        
        self.x_opt = result.x
        self.fun_val_opt = result.fun


class CompositeSolver(Solver):

    def __init__(self):
        super().__init__(model_instance=None)
        self.solver = None
    
    def set_solver(self, solver):
        self.solver = solver

    def set_model_instance(self, model_instance):
        if self.is_solver_defined():
            self.solver.set_model_instance(model_instance)

    def detach_model_instance(self):
        if self.solver is not None:
            self.solver.detach_model_instance()

    def get_model_instance(self):
        if self.is_solver_defined():
            return self.solver.get_model_instance()
        
    def is_solver_defined(self):
        if self.solver is not None:
            return True 
        else:
            raise SolverNotDefinedError()


class MultipleInitializations(CompositeSolver):

    def __init__(self, num_init, sample_fun):
        super().__init__()
        self.num_init = num_init
        self.sample_fun = lambda x: sample_fun(x, self.num_init)

    def fit(self, data, x_init=None, options=None):
        if self.is_solver_defined():
            if x_init is None:
                x_init = self.get_model_instance().x_init        
            fun_vals = []
            xs_opt = []
            xs_init = self.sample_fun(x_init)
            for i in range(self.num_init):
                self.solver.fit(data, xs_init[i], options=options)
                fun_vals.append(self.solver.fun_val_opt)
                xs_opt.append(self.solver.x_opt)

            self.x_opt = xs_opt[np.argmin(fun_vals)]
            self.fun_val_opt = np.min(fun_vals)
