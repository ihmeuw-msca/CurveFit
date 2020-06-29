import pytest
import pandas as pd
import numpy as np

from curvefit.utils.smoothing import local_deviations, convolve_sum, df_to_mat
from curvefit.utils.smoothing import local_smoother


def test_local_deviations():
    np.random.seed(10)
    residual_data = pd.DataFrame({
        'num_data': np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]),
        'far_out': np.array([1, 2, 3, 1, 2, 3, 1, 2, 3]),
        'residual': np.random.randn(9)
    })
    smoothed = local_deviations(
        df=residual_data,
        col_val='residual',
        col_axis=['num_data', 'far_out'],
        radius=[1, 1]
    )
    assert np.isclose(
        smoothed.residual_std[
            (smoothed.num_data == 1) & (smoothed.far_out == 1)
        ].iloc[0], residual_data.residual[
            (residual_data.num_data.isin([1, 2])) &
            (residual_data.far_out.isin([1, 2]))
        ].mad() * 1.4826
    )
    assert np.isclose(
        smoothed.residual_std[
            (smoothed.num_data == 2) & (smoothed.far_out == 2)
        ].iloc[0], residual_data.residual[
            (residual_data.num_data.isin([1, 2, 3])) &
            (residual_data.far_out.isin([1, 2, 3]))
        ].mad() * 1.4826
    )
    assert np.isclose(
        smoothed.residual_std[
            (smoothed.num_data == 3) & (smoothed.far_out == 3)
        ].iloc[0], residual_data.residual[
            (residual_data.num_data.isin([2, 3])) &
            (residual_data.far_out.isin([2, 3]))
        ].mad() * 1.4826
    )


@pytest.mark.parametrize('mat', [np.arange(9).reshape(3, 3)])
@pytest.mark.parametrize(('radius', 'result'),
                         [((0, 0), np.arange(9).reshape(3, 3)),
                          ((1, 1), np.array([[8, 15, 12], [21, 36, 27], [20, 33, 24]]))])
def test_convolve_sum(mat, radius, result):
    my_result = convolve_sum(mat, radius=radius)
    assert np.allclose(result, my_result)


def test_df_to_mat():
    df = pd.DataFrame({
        'val': np.ones(5),
        'axis0': np.arange(5, dtype=float),
        'axis1': np.arange(5, dtype=float)
    })

    my_result, indices, axis = df_to_mat(df, 'val', ['axis0', 'axis1'], return_indices=True)
    assert np.allclose(my_result[indices[:, 0], indices[:, 1]], 1.0)


@pytest.mark.parametrize('radius', [[1, 1]])
def test_local_smoother(radius):
    data = pd.DataFrame({
        'val': np.arange(5),
        'axis0': np.arange(5),
        'axis1': np.arange(5)
    })
    result = local_smoother(data, 'val', ['axis0', 'axis1'], radius=radius)
    assert np.allclose(result['val_' + 'mean'].values,
                       np.array([0.5, 1.0, 2.0, 3.0, 3.5]))
