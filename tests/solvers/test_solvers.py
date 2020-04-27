import pytest 
import numpy as np
from scipy.optimize import rosen, rosen_der

from curvefit.solvers.solvers import ScipyOpt, MultipleInitializations
from curvefit.models.base import Model
from curvefit.models.core_model import CoreModel
from curvefit.core.functions import gaussian_cdf, gaussian_pdf, ln_gaussian_cdf, ln_gaussian_pdf, normal_loss, st_loss
from data_and_param_simulator import simulate_params, simulate_data


class Rosenbrock(Model):

    def __init__(self, n_dim=2):
        super().__init__()
        self.n_dim = n_dim
        self.bounds = np.array([[-2.0, 2.0]] * n_dim)
        self.x_init = np.array([-1.0] * n_dim)

    @staticmethod
    def objective(x, data):
        return rosen(x)

    @staticmethod
    def gradient(x, data):
        return rosen_der(x)

    def convert_inputs(self, data):
        pass


@pytest.fixture(scope='module')
def rb():
    return Rosenbrock()


class TestBaseSolvers:

    def test_scipyopt(self, rb):
        solver = ScipyOpt(rb)
        solver.fit(data=None, options={'maxiter': 20})
        assert np.abs(solver.fun_val_opt) < 1e-5

    @pytest.mark.parametrize('curve_fun', [ln_gaussian_pdf, ln_gaussian_cdf, gaussian_pdf, gaussian_cdf])
    def test_scipyopt_core_mdoel(self, curve_fun):
        params_set, params_true, _ = simulate_params(1)
        data = simulate_data(curve_fun, params_true)
        core_model = CoreModel(params_set, curve_fun, normal_loss)
        solver = ScipyOpt(core_model)
        solver.fit(data=data, options={'maxiter': 50})
        assert solver.fun_val_opt < 1e-5
        y_pred = solver.predict(t=data[0]['t'].to_numpy())
        y_true = data[0]['obs'].to_numpy()
        assert np.linalg.norm(y_pred - y_true) / np.linalg.norm(y_true) < 1e-2
        

class TestCompositeSolvers:

    def test_multi_init(self, rb):
        num_init = 3
        xs_init = np.random.uniform(
            low=[b[0] for b in rb.bounds], 
            high=[b[1] for b in rb.bounds],
            size=(num_init, rb.n_dim),
        )
        sample_fun = lambda x: xs_init
        solver = MultipleInitializations(sample_fun)
        base_solver = ScipyOpt()
        solver.set_solver(base_solver)
        solver.set_model_instance(rb)
        assert solver.model is None
        assert isinstance(solver.get_model_instance(), Rosenbrock)
        solver.fit(data=None, options={'maxiter': 15})
        
        for x in xs_init:
            assert rb.objective(x, None) >= solver.fun_val_opt








