import pytest 
import numpy as np
from scipy.optimize import rosen, rosen_der

from curvefit.solvers.solvers import ScipyOpt, MultipleInitializations


class Rosenbrock:

    def __init__(self, n_dim=2):
        self.n_dim = n_dim
        self.bounds = np.array([[-2.0, 2.0]] * n_dim)
        self.x_init = np.array([-1.0] * n_dim)

    @staticmethod
    def objective(x, data):
        return rosen(x)

    @staticmethod
    def gradient(x, data):
        return rosen_der(x)


@pytest.fixture(scope='module')
def rb():
    return Rosenbrock()


class TestBaseSolvers:

    def test_scipyopt(self, rb):
        solver = ScipyOpt(rb)
        solver.fit(data=None, options={'maxiter': 20})
        assert np.abs(solver.fun_val_opt) < 1e-5


class TestCompositeSolvers:

    def test_multi_init(self, rb):
        num_init = 3
        xs_init = np.random.uniform(
            low=[b[0] for b in rb.bounds], 
            high=[b[1] for b in rb.bounds],
            size=(num_init, rb.n_dim),
        )
        sample_fun = lambda x, n: xs_init
        solver = MultipleInitializations(num_init, sample_fun)
        base_solver = ScipyOpt()
        solver.set_solver(base_solver)
        solver.set_model_instance(rb)
        assert solver.model is None
        assert isinstance(solver.get_model_instance(), Rosenbrock)
        solver.fit(data=None, options={'maxiter': 15})
        
        for x in xs_init:
            assert rb.objective(x, None) >= solver.fun_val_opt








