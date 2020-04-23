import numpy as np
import scipy.optimize as sciopt

from curvefit.models.gaussian_mixture import GaussianMixtures
from curvefit.core.effects2params import effects2params

class SolverNotDefinedError(Exception):
    pass


class Base:

    def __init__(self, model_instance=None):
        self.model = model_instance
        self.x_opt = None
        self.fun_val_opt = None
    
    def set_model_instance(self, model_instance):
        self.model = model_instance

    def fit(self, data, x_init=None, options=None):
        raise NotImplementedError()
    
    def predict(self, data):
        return self.model.forward(self.x_opt, data)


class ScipyOpt(Base):

    def fit(self, data, x_init=None, options=None):
        if x_init is None:
            x_init = self.model.x_init
        
        result = sciopt.minimize(
            fun=lambda x: self.model.objective(x, data), 
            x0=x_init, 
            jac=self.model.gradient,
            bounds=self.model.bounds,
            options=options,
        )
        
        self.x_opt = result.x
        self.fun_val_opt = result.fun


class Composite(Base):

    def __init__(self, model_instance=None):
        super().__init__(model_instance)
        self.solver = None
    
    def update_solver(self, solver):
        self.solver = solver
        if self.model is not None:
            self.solver.set_model_instance(self.model)

    def set_model_instance(self, model_instance):
        self.model = model_instance
        if self.solver is not None:
            self.solver.set_model_instance(self.model)
        
    def is_solver_defined(self):
        if self.solver is not None:
            return True 
        else:
            raise SolverNotDefinedError()


class MultipleInitializations(Composite):

    def __init__(self, num_init, sample_fun, model_instance=None):
        super().__init__(model_instance)
        self.num_init = num_init
        self.sample_fun = sample_fun

    def fit(self, data, x_init=None, options=None):
        if self.is_solver_defined():
            fun_vals = []
            xs_opt = []
            xs_init = self.sample_fun(self.num_init)
            for i in range(self.num_init):
                self.solver.fit(data, xs_init[i], options=options)
                fun_vals.append(self.solver.fun_val_opt)
                xs_opt.append(self.solver.x_opt)
            
            self.x_opt = xs_opt[np.argmin(fun_vals)]
            self.fun_val = np.min(fun_vals)

class GaussianMixturesIntegration(Composite):

    def __init__(self, gm_model, model_instance=None):
        super().__init__(model_instance)
        self.gm_model = gm_model 

    def fit(self, data, x_init=None, options=None):
        if self.is_solver_defined():
            self.solver.fit(data, x_init, options)
            params = effects2params(
                self.solver.x_opt, 
                self.model.group_size, 
                self.model.covs,
                self.model.link_fun,
                self.model.var_link_fun,
            )
            self.gm_model.set_params(params)
            gm_solver = ScipyOpt(self.gm_model)
            w_init = np.zeros(self.gm_model.size)
            w_init[self.gm_model.size // 2] = 1.0
            gm_solver.fit(data, w_init)


    



            
            

