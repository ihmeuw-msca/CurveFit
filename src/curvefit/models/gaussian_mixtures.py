import numpy as np
from curvefit.core.functions import gaussian_pdf

class GaussianMixtures:

    def __init__(self, stride, size, params=None):
        self.params = params
        self.stride = stride 
        self.size = size

    def set_params(self, params):
        self.params = params

    def compute_design_matrix(self, x):
        half_size = self.size // 2
        self.size = half_size * 2 + 1  # making sure it's odd
        betas = np.linspace(
            self.params[1] - half_size * self.stride, 
            self.params[1] + half_size * self.stride, 
            num=self.size,
        )
        assert np.abs(betas[half_size] - self.params[1]) / np.abs(self.params[1]) < 1e-2
        X = []
        for beta in betas:
            X.append(gaussian_pdf(x, [self.params[0], beta, self.params[2]]))
        X = np.asarray(X).T
        assert X.shape == (len(x), self.size)
        return X

    def _objective_and_gradient(self, x, data):
        df = data[0]
        data_specs = data[1]
        obs = df[data_specs.col_obs]
        obs_se = df[data_specs.col_obs_se]
        t = df[data_specs.col_t]
        self.matrix = self.compute_design_matrix(t)
        residuals = (obs - np.dot(self.matrix, x)) / obs_se
        return 0.5 * np.sum(residuals**2), -(self.matrix.T / obs_se).dot(residuals)

    def objective(self, x, data):
        return self._objective_and_gradient(x, data)[0]

    def gradient(self, x, data):
        return self._objective_and_gradient(x, data)[1]

    @property
    def bounds(self):
        return np.array([[0.0, np.inf]] * self.size)

    @property
    def x_init(self):
        x = np.zeros(self.size)
        x[self.size // 2] = 1.0
        return x

    def predict(self, x, data):
        df = data[0]
        data_specs = data[1]
        t = df[data_specs.col_t]
        matrix = self.compute_design_matrix(t)
        return np.dot(matrix, x)