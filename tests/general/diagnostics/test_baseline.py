import pytest
import numpy as np
from general.diagnostics.baselines import LinearRegressionBaseline
from data_simulator import simulate_linear_data_multigroups

class TestBaselines:

    # Note (Jize) -- we probably need more tests. This is just a sanity check.
    @pytest.mark.parametrize('n_groups, max_n_data, n_features, noisy', [[5, 30, 1, False], [5, 30, 3, True]])
    def test_linear_regression_baseline(self, n_groups, max_n_data, n_features, noisy):
        ys, Xs, groups, ytrues = simulate_linear_data_multigroups(n_groups, max_n_data, n_features)
        baseline = LinearRegressionBaseline(ys, groups, Xs)
        baseline.fit()
        for grp, X, ytrue in zip(groups, Xs, ytrues):
            assert np.linalg.norm(baseline.baseline_est[grp] - ytrue) < 1e-5
        values = baseline.compare(ys, groups, metric_fun=lambda x, y: np.linalg.norm(x-y))
        for k, vs in values.items():
            assert [abs(v) <1e-16 for v in vs]
            assert k in groups
    
