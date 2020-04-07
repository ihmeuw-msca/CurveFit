# -*- coding: utf-8 -*-
"""
    test pv
    ~~~~~~~

    Test pv module
"""
import numpy as np
import pytest
from curvefit.pv.pv import PVGroup
from curvefit.core.utils import condense_residual_matrix


@pytest.mark.parametrize('mat', [np.arange(16).reshape(4, 4)])
@pytest.mark.parametrize('diff', [np.array([1]*3),
                                  np.arange(3)])
@pytest.mark.parametrize('num_points', [np.arange(4) + 1])
def test_condense_residual_matrix(mat, diff, num_points):
    my_result = condense_residual_matrix(mat, diff, num_points)
    result = PVGroup.condense_residual_matrix(mat, diff, num_points)

    assert np.allclose(np.sort(my_result[:, 0]),
                       np.sort(result[:, 0]))
    assert np.allclose(np.sort(my_result[:, 1]),
                       np.sort(result[:, 1]))
    assert np.allclose(np.sort(my_result[:, 2]),
                       np.sort(result[:, 2]))
