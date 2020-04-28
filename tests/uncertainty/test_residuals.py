import numpy as np
import pytest
from curvefit.uncertainty.residuals import Residuals


def condense_residual_matrix_by_hand(matrix, sequential_diffs, data_density):
    """
    Condense the residuals from a residual matrix to three columns
    that represent how far out the prediction was, the number of data points,
    and the observed residual.

    Args:
        matrix: (np.ndarray)
        sequential_diffs:
        data_density:

    Returns:

    """
    far_out = np.array([])
    num_data = np.array([])
    robs = np.array([])

    diagonals = np.array(range(matrix.shape[0]))[1:]

    # get the diagonal of the residual matrix and figure out
    # how many data points out we were predicting (convolve)
    # plus the amount of data that we had to do the prediction
    for i in diagonals:
        diagonal = np.diag(matrix, k=i)
        obs = len(diagonal)
        out = np.convolve(sequential_diffs, np.ones(i, dtype=int), mode='valid')

        far_out = np.append(far_out, out[-obs:])
        num_data = np.append(num_data, data_density[:obs])
        robs = np.append(robs, diagonal)

    # return the results for the residual matrix as a (len(available_times), 3) shaped matrix
    r_matrix = np.vstack([far_out, num_data, robs]).T
    return r_matrix


@pytest.mark.parametrize('mat', [np.arange(16).reshape(4, 4)])
@pytest.mark.parametrize('diff', [np.array([1]*3),
                                  np.arange(3)])
@pytest.mark.parametrize('num_points', [np.arange(4) + 1])
def test_condense_residual_matrix(mat, diff, num_points):
    by_hand = condense_residual_matrix_by_hand(mat, diff, num_points)
    result = Residuals._condense_matrix(mat, diff, num_points)

    assert np.allclose(np.sort(by_hand[:, 0]),
                       np.sort(result[:, 0]))
    assert np.allclose(np.sort(by_hand[:, 1]),
                       np.sort(result[:, 1]))
    assert np.allclose(np.sort(by_hand[:, 2]),
                       np.sort(result[:, 2]))
