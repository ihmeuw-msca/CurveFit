import numpy as np
from . import utils
from scipy.optimize import minimize


class GaussianMixture:
    def __init__(self, df, col_t, col_obs, params, beta_stride, mixture_size,
                 col_obs_se=None):
        """
        Fits a Gaussian Mixture model.

        Args:
            df:
            col_t:
            col_obs:
            params:
            beta_stride:
            mixture_size:
            col_obs_se:
        """

        self.df = df.copy()
        self.col_t = col_t
        self.col_obs = col_obs
        self.col_obs_se = col_obs_se
        self.params = params
        self.beta_stride = beta_stride
        self.mixture_size = mixture_size

        self.df.sort_values(col_t, inplace=True)
        self.t = self.df[self.col_t].values
        self.obs = self.df[self.col_obs].values
        self.num_obs = self.obs.size
        if self.col_obs_se is None:
            self.obs_se = np.ones(self.num_obs)*np.abs(self.obs).mean()
        else:
            self.obs_se = self.df[self.col_obs_se].values*np.abs(
                self.obs).mean()

        self.mat, _ = utils.compute_gaussian_mixture_matrix(self.t,
                                                            self.params,
                                                            self.beta_stride,
                                                            self.mixture_size)
        self.nu = 1e-2
        self.weights = None
        self.result = None
        self.sum_weights_bounds = None

    def objective(self, w):
        residual = (self.obs - self.mat.dot(w[:-1]))/self.obs_se
        sum_residual = (np.sum(w[:-1]) - w[-1])/self.nu
        val = 0.5*np.sum(residual**2) + 0.5*np.sum(sum_residual**2)
        return val

    def gradient(self, w):
        residual = (self.obs - self.mat.dot(w[:-1]))/self.obs_se
        sum_residual = (np.sum(w[:-1]) - w[-1])/self.nu
        grad = -(self.mat.T/self.obs_se).dot(residual) + sum_residual/self.nu
        grad = np.hstack((grad, -sum_residual/self.nu))
        return grad

    def fit_mixture_weights(self, w0=None, bounds=None, options=None):
        if w0 is None:
            w0 = np.zeros(self.mixture_size)
            w0[self.mixture_size//2] = 1.0
            w0 = np.hstack((w0, 1.0))

        if bounds is None:
            bounds = np.array([[0.0, np.inf]]*self.mixture_size)

        result = minimize(
            fun=self.objective,
            x0=w0,
            jac=self.gradient,
            method='L-BFGS-B',
            bounds=bounds,
            options=options
        )
        self.result = result
        self.weights = self.result.x[:-1]

    def predict(self, t):
        mat, _ = utils.compute_gaussian_mixture_matrix(t,
                                                       self.params,
                                                       self.beta_stride,
                                                       self.mixture_size)
        return mat.dot(self.weights)
