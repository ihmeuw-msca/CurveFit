import pytest
import numpy as np
from general.diagnostics.peak_detectors import PieceWiseLinearPeakDetector
from data_simulator import simulate_random_data_with_labels

class TestPeakDetectors:

    # Note (jize) -- this just checks everything runs, not a correctness check
    @pytest.mark.parametrize('n_groups, max_n_data, n_features', [[10, 10, 1]])
    def test_piecewise_linear_peak_detector(self, n_groups, max_n_data, n_features):
        ys, Xs, groups, labels = simulate_random_data_with_labels(n_groups, max_n_data, n_features)
        peak_detector = PieceWiseLinearPeakDetector(ys, groups, Xs, labels)
        peak_detector.train_peak_classifier()
        new_y = np.random.randn(10)
        group = 'grp'
        new_X = np.random.randn(10, n_features)
        peak_detector.has_peaked(new_y, group, new_X)