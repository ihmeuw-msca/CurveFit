import pytest
import numpy as np
from general.diagnostics.baselines import LinearRegressionBaseline
from data_simulator import simulate_linear_data_multigroups

class TestBaselines:

    @pytest.mark.parametrize('n_groups, max_n_data, n_features', [[10, 30, 1]])
    def test_linear_regression_baseline(self, n_groups, max_n_data, n_features):
        ys, Xs, groups, coefs = simulate_linear_data_multigroups(n_groups, max_n_data, n_features)
        baseline = LinearRegressionBaseline(ys, groups, Xs)
        baseline.fit()
        for grp, X, coef in zip(groups, Xs, coefs):
            assert np.linalg.norm(baseline.baseline_est[grp] - np.dot(X, coef)) < 1e-7
        values = baseline.compare(ys, groups, metric_fun=lambda x, y: np.linalg.norm(x-y))
        for _, vs in values.items():
            assert [abs(v) <1e-16 for v in vs]
    
