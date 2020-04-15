import numpy as np


class GaussianMixture:
    def __init__(self, num_gaussians):
        self.num_gaussians = num_gaussians
        self.weights = np.empty(self.num_gaussians)
        self.weights[:] = np.nan

    @staticmethod
    def get_gauss_mix_design_matrix(times, num_gaussians, params, peak_width):
        return np.empty(shape=(times, num_gaussians))

    def fit_gaussian_mixture(self, times, outcome_var, num_gaussians, params, peak_width):
        dm = self.get_gauss_mix_design_matrix(
            times=times, num_gaussians=num_gaussians,
            params=params, peak_width=peak_width
        )
        # perform model fitting with outcome_var and dm with constraints on the betas
        # this will not actually return the outcome var

        # fit and replace self.weights
        # ...
        return outcome_var
