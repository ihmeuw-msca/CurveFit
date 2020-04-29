import numpy as np
from curvefit.core.functions import gaussian_pdf
from curvefit.models.base import Model, DataInputs


class GaussianMixtures(Model):
    """
    {begin_markdown GaussianMixtures}
    {spell_markdown init}

    # `curvefit.models.gaussian_mixtures.GaussianMixtures`
    ## Fit a linear combination of some Gaussian distributions

    The `GaussianMixtures` model allows you to fit weights for a linear combination
    of Gaussian distributions using least squares. It is typically used
    on top of a [`CoreModel`](CoreModel.md).

    ## Arguments

    - `stride (int)`: the distance between the Gaussian atoms; how far to space them out
        over the independent variable
    - `size (int)`: number of Gaussian atoms to include, replicated over different
        locations determined by `stride`
    - `params (np.array)`: parameters to build the Gaussian distributions; must
        be parameters from a Gaussian family function in `curvefit.core.functions`

    ## Methods

    ### `compute_design_matrix`
    Calculates the design matrix from the Gaussian atoms that will be used in order
    to get their optimal linear combination.

    - `t (np.array)`: how far in time should the predictions that will go into
        the design matrix extend

    See additional methods in [`CoreModel`](CoreModel.md).

    {end_markdown GaussianMixtures}
    """

    def __init__(self, stride, size, params=None):
        super().__init__()
        self.params = params
        self.stride = stride 
        self.size = size

    def set_params(self, params):
        self.params = params

    def compute_design_matrix(self, t):
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
            X.append(gaussian_pdf(t, [self.params[0], beta, self.params[2]]))
        X = np.asarray(X).T
        assert X.shape == (len(t), self.size)
        return X, betas

    def _objective_and_gradient(self, x, t, obs, obs_se):
        self.matrix = self.compute_design_matrix(t)[0]
        residuals = (obs - np.dot(self.matrix, x)) / obs_se
        return 0.5 * np.sum(residuals**2), -(self.matrix.T / obs_se).dot(residuals)

    def objective(self, x, data):
        if self.data_inputs is None:
            self.convert_inputs(data)
        return self._objective_and_gradient(x, self.data_inputs.t, self.data_inputs.obs, self.data_inputs.obs_se)[0]

    def gradient(self, x, data):
        if self.data_inputs is None:
            self.convert_inputs(data)
        return self._objective_and_gradient(x, self.data_inputs.t, self.data_inputs.obs, self.data_inputs.obs_se)[1]

    @property
    def bounds(self):
        return np.array([[0.0, np.inf]] * self.size)

    @property
    def x_init(self):
        x = np.zeros(self.size)
        x[self.size // 2] = 1.0
        return x

    def predict(self, x, t):
        matrix = self.compute_design_matrix(t)[0]
        return np.dot(matrix, x)

    def convert_inputs(self, data):
        if isinstance(data, DataInputs):
            self.data_inputs = data
            return 
        
        df = data[0]
        data_specs = data[1]

        t = df[data_specs.col_t].to_numpy()
        obs = df[data_specs.col_obs].to_numpy()
        obs_se = df[data_specs.col_obs_se].to_numpy()
        
        self.data_inputs = DataInputs(t=t, obs=obs, obs_se=obs_se)