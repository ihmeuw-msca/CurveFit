import numpy as np
from scipy.stats import norm

from curvefit.models.gaussian_mixtures import GaussianMixtures

def test_compute_design_matrix():
    np.random.seed(123)
    n_run = 5
    n_t = 10

    betas = np.random.uniform(-2, 2, size=n_run)
    stds = np.random.rand(n_run)
    alphas = 1 / (stds * np.sqrt(2))
    params = np.asarray([alphas, betas, [1.0] * n_run]).T
    beta_stride = np.random.rand(n_run)
    mixture_size = np.random.choice(np.arange(1, 10), size=n_run)
    ts = np.random.randn(n_run, n_t)
    
    for i in range(n_run):
        gm = GaussianMixtures(beta_stride[i], mixture_size[i], params[i])
        X, beta_vec = gm.compute_design_matrix(ts[i])
        for j in range(len(beta_vec)):
            assert np.linalg.norm(X[:, j] - norm.pdf(ts[i], loc=beta_vec[j], scale=stds[i])) < 1e-5

    