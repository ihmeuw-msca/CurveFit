# -*- coding: utf-8 -*-
"""
    test pv
    ~~~~~~~

    Test pv module
"""
import numpy as np
import pytest
from curvefit.pv import PVGroup

@pytest.mark.parametrize('mat', [np.arange(16).reshape(4, 4)])
@pytest.mark.parametrize('diff', [np.array([1]*3),
                                  np.arange(3)])
@pytest.mark.parametrize('num_points', [np.arange(4) + 1])
def test_condense_residual_matrix(mat, diff, num_points):
    row_idx, col_idx = np.triu_indices(mat.shape[0], 1)
    map1 = np.cumsum(np.insert(diff, 0, 0))
    map2 = num_points

    far_out = map1[col_idx] - map1[row_idx]
    points = map2[row_idx]
    residuals = mat[row_idx, col_idx]

    result = PVGroup.condense_residual_matrix(mat, diff, num_points)

    print(np.allclose(np.sort(far_out), np.sort(result[:, 0])))

    assert np.allclose(np.sort(far_out), np.sort(result[:, 0]))
    assert np.allclose(np.sort(points), np.sort(result[:, 1]))
    assert np.allclose(np.sort(residuals), np.sort(result[:, 2]))
