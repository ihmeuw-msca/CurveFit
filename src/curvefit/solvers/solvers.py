import numpy as np
from copy import deepcopy
import scipy.optimize as sciopt

from curvefit.core.effects2params import effects2params
from curvefit.core.prototype import Prototype
from curvefit.utils.data import data_translator
from curvefit.core.functions import gaussian_pdf
from curvefit.models.base import DataInputs


class ModelNotDefinedError(Exception):
    pass


class SolverNotDefinedError(Exception):
    pass


class Solver(Prototype):
    """
    {begin_markdown Solver}
    {spell_markdown kwargs}

    # `curvefit.solvers.solvers.Solver`
    ## Solver base class that fits a Model

    In order to fit a `curvefit.models.base.Model`, you must
    first define a `Solver` and assign the model to the solver.
    The reason for this is that there might be multiple ways that
    you could solve a particular model.

    ## Arguments

    - `model_instance (curvefit.models.base.Model)`: the model instance
        that will be solved

    ## Methods

    ### `fit`
    Fit the solver to some data using `self.model_instance`.

    - `data (Tuple[pd.DataFrame, DataSpecs])`: the input data frame to be fit,
        and data specifications object
    - `options (None | Options)`: an optional Options object that has
        fit specifications for the underlying solver; overrides
        the options that have already been set

    ### `predict`
    Create predictions based on the optimal values estimated by the solver.

    - `**kwargs`: keyword arguments passed to `self.model_instance.predict()`

    ### `set_options`
    Set a dictionary of options that will be used in the optimization.

    ### `set_model_instance`
    Attach a new model instance.

    ### `detach_model_instance`
    Detach the current model instance.

    ### `get_model_instance`
    Get the current model instance.

    {end_markdown Solver}
    """

    def __init__(self, model_instance=None):
        self.model = model_instance
        self.success = None
        self.x_opt = None
        self.fun_val_opt = None
        self.options = None
        self.status = None

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

    def get_fit_status(self):
        return self.status

    def fit(self, data, x_init=None, options=None):
        raise NotImplementedError()

    def predict(self, **kwargs):
        return self.model.predict(self.x_opt, **kwargs)


class ScipyOpt(Solver):
    """
    {begin_markdown ScipyOpt}

    # `curvefit.solvers.solvers.ScipyOpt`
    ## Scipy Optimize Solver
    Use `scipy` optimize to fit the model using the L-BFGS-B method.

    See [`Solver`](Solver.md) for arguments and methods.

    {end_markdown ScipyOpt}
    """

    def fit(self, data, x_init=None, options=None):
        import cppad_py
        #
        self.model.convert_inputs(data)
        if x_init is None:
            x_init = self.model.x_init
        else:
            x_init = x_init[:len(self.model.x_init)]
        #
        # create a cppad_py.d_fun version of the objective
        ax    = cppad_py.independent(x_init)
        aobj  = self.model.objective(ax, data)
        ay    = np.array( [ aobj ] )
        d_fun = cppad_py.d_fun(ax, ay);
        #
        # evaluate objective using d_fun
        def d_fun_objective(x) :
            y = d_fun.forward(0, x)
            return y[0]
        #
        # evaluate gradient of objective using d_fun
        def d_fun_gradient(x) :
            J = d_fun.jacobian(x)
            return J.flatten()
        #
        result = sciopt.minimize(
            fun=d_fun_objective,
            x0=x_init,
            jac=d_fun_gradient,
            bounds=self.model.bounds,
            options=options if options is not None else self.options,
        )
        #
        self.success = result.success
        self.x_opt = result.x
        self.fun_val_opt = result.fun
        self.status = result.message


class CompositeSolver(Solver):
    """
    {begin_markdown CompositeSolver}

    # `curvefit.solvers.solvers.CompositeSolver`
    ## Composite Solver
    General base class for a solver with multiple elements. Used to
    create composites of solvers.

    ## Arguments

    - `solver (curvefit.solvers.solvers.Solver)`: a base solver
        to build the composite off of

    See [`Solver`](Solver.md) for more arguments and methods.

    {end_markdown CompositeSolver}
    """

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

    def get_fit_status(self):
        if self.assert_solver_defined() is True:
            return self.solver.get_fit_status()

    def assert_solver_defined(self):
        if self.solver is not None:
            return True
        else:
            raise SolverNotDefinedError()

    def predict(self, **kwargs):
        return self.solver.predict(**kwargs)


class MultipleInitializations(CompositeSolver):
    """
    {begin_markdown MultipleInitialization}
    {spell_markdown Initialization}

    # `curvefit.solvers.solvers.MultipleInitialization`
    ## MultipleInitialization Solver
    Uses a sampling function to sample initial values around
    the initial values specified in the model instance, and picks
    the initial values which attain lowest objective function value.

    ## Arguments

    - `sample_fun (Callable)`: some function to use to sample
        initial points around the initial points specified in the model
        instance

    See [`CompositeSolver`](CompositeSolver.md) for more arguments and methods.

    {end_markdown MultipleInitialization}
    """

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
    """
    {begin_markdown GaussianMixturesIntegration}

    # `curvefit.solvers.solvers.GaussianMixturesIntegration`
    ## GaussianMixturesIntegration Solver
    The solver that is used to find the linear combination of Gaussian atoms
    built on top of one core model instance.

    ## Arguments

    - `gm_model (curvefit.models.gaussian_mixtures.GaussianMixtures)`: some
        a model instance of a gaussian mixture model

    See [`CompositeSolver`](CompositeSolver.md) for more arguments and methods.

    {end_markdown GaussianMixturesIntegration}
    """

    def __init__(self, gm_model, solver=None):
        super().__init__()
        self.gm_model = gm_model

    def fit(self, data, x_init=None, options=None):
        if self.assert_solver_defined() is True:
            self.solver.fit(data, x_init, options)
            model = self.get_model_instance()
            self.input_curve_fun = model.curve_fun
            params = effects2params(
                self.solver.x_opt,
                model.data_inputs.group_sizes,
                model.data_inputs.covariates_matrices,
                model.param_set.link_fun,
                model.data_inputs.var_link_fun,
                expand=False,
            )
            self.gm_model.set_params(params[:, 0])
            gm_solver = ScipyOpt(self.gm_model)
            data_inputs_gm = DataInputs(
                t=model.data_inputs.t,
                obs=model.data_inputs.obs,
                obs_se=model.data_inputs.obs_se,
            )
            obs_gau_pdf = data_translator(data_inputs_gm.obs, model.curve_fun, gaussian_pdf)
            data_inputs_gm.obs = obs_gau_pdf
            gm_solver.fit(data_inputs_gm)
            self.x_opt = gm_solver.x_opt
            self.fun_val_opt = gm_solver.fun_val_opt

    def predict(self, t, predict_fun=None):
        pred_gau_pdf = self.gm_model.predict(self.x_opt, t)
        if predict_fun is None:
            return data_translator(pred_gau_pdf, gaussian_pdf, self.input_curve_fun)
        return data_translator(pred_gau_pdf, gaussian_pdf, predict_fun)


class SmartInitialization(CompositeSolver):
    """
    {begin_markdown SmartInitialization}
    {spell_markdown Initialization}

    # `curvefit.solvers.solvers.SmartInitialization`
    ## SmartInitialization Solver
    Built on top of any solver; used when there are many groups that will be
    fit to using random effects. First fits individual models for each group,
    and uses their fixed effect values minus the fixed effects mean across all groups as the
    random effects initial values (finding a "smart" initial value for better
    convergence).

    See [`CompositeSolver`](CompositeSolver.md) for more arguments and methods.

    {end_markdown SmartInitialization}
    """

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
                assert model.data_inputs is None
                self.solver.fit(data_sub, None, options)
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
