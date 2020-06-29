import numpy as np
from scipy.stats import norm

from curvefit.uncertainty.draws import Draws


def test_mean_and_quantiles_for_draws():
    # The test may spontaneously fail if you change the seed because there is always
    # a non-zero probability of a sample quantile to diverge from its expectation (Hoeffding's inequality).
    seed = 42
    # the next three are related through CLT: you need to increase the number of draws
    # if you want to set smaller tolerance.
    rtol = 0.05
    atol = 0.1
    num_draws = 2000
    prediction_times = np.arange(0, 31, 1)
    np.random.seed(seed)
    draws = np.random.randn(num_draws, len(prediction_times))
    mean, lower_quantile, higher_quantile = Draws._get_mean_and_quantiles_for_draws(draws, quantiles=0.05)
    assert np.allclose(mean, np.zeros(len(prediction_times)), rtol=rtol, atol=atol)
    assert np.allclose(lower_quantile, norm.ppf(0.05), rtol=rtol, atol=atol)
    assert np.allclose(higher_quantile, norm.ppf(0.95), rtol=rtol, atol=atol)
