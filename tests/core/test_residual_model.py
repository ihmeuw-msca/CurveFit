import pytest
import numpy as np
import pandas as pd

from curvefit.core.residual_model import _ResidualModel, SmoothResidualModel
from curvefit.utils.smoothing import local_deviations, local_smoother


def test_residual_model():
    rm = _ResidualModel(
        cv_bounds=[1e-4, np.inf],
        covariates={'far_out': 'far_out >= 10', 'num_data': None},
        exclude_groups=None
    )
    assert rm.cv_bounds == [1e-4, np.inf]
    assert 'far_out' in rm.covariates
    assert 'num_data' in rm.covariates
    assert rm.exclude_groups is None


@pytest.fixture
def smooth_rm():
    return SmoothResidualModel(
        cv_bounds=[1e-4, np.inf],
        covariates={'far_out': None, 'num_data': 'num_data >= 2'},
        exclude_groups=None,
        num_smooth_iterations=1,
        smooth_radius=[1, 1],
        robust=True
    )


def test_smooth_residual_model_error():
    with pytest.raises(AssertionError):
        SmoothResidualModel(
            cv_bounds=[1e-4, np.inf],
            covariates={'log_dr': None, 'num_data': None},
            exclude_groups=None,
            num_smooth_iterations=1,
            smooth_radius=[1, 1],
            robust=True
        )


def test_smooth_residual_model(smooth_rm):
    assert smooth_rm.cv_bounds == [1e-4, np.inf]
    assert 'far_out' in smooth_rm.covariates
    assert 'num_data' in smooth_rm.covariates
    assert smooth_rm.exclude_groups is None

    assert smooth_rm.smooth_radius == [1, 1]
    assert smooth_rm.num_smooth_iterations == 1
    assert smooth_rm.robust


@pytest.fixture(scope='module')
def residual_data():
    np.random.seed(10)
    return pd.DataFrame({
        'num_data': np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]),
        'far_out': np.array([1, 2, 3, 1, 2, 3, 1, 2, 3]),
        'residual': np.random.randn(9)
    })


def test_smooth_residual_model_smooth(smooth_rm, residual_data):
    smooth_rm.fit_residuals(residual_data)
    by_hand = local_deviations(
        df=residual_data[residual_data.num_data >= 2],
        col_val='residual',
        col_axis=['far_out', 'num_data'],
        radius=[1, 1]
    )
    pd.testing.assert_frame_equal(
        smooth_rm.smoothed_residual_data.sort_values(['far_out', 'num_data']).reset_index(),
        by_hand.sort_values(['far_out', 'num_data']).reset_index()
    )


def test_smooth_residual_model_extra_smooth(smooth_rm, residual_data):
    smooth_rm.num_smooth_iterations = 2
    smooth_rm.fit_residuals(residual_data)
    by_hand = local_deviations(
        df=residual_data[residual_data.num_data >= 2],
        col_val='residual',
        col_axis=['far_out', 'num_data'],
        radius=[1, 1]
    )
    by_hand = local_smoother(
        df=by_hand,
        col_val='residual_std',
        col_axis=['far_out', 'num_data'],
        radius=[1, 1]
    )
    by_hand.rename(columns={
        'residual_std_mean': 'residual_std'
    }, inplace=True)
    pd.testing.assert_frame_equal(
        smooth_rm.smoothed_residual_data.sort_values(['far_out', 'num_data']).reset_index(),
        by_hand.sort_values(['far_out', 'num_data']).reset_index()
    )


def test_smooth_residual_predictions(smooth_rm, residual_data):
    smooth_rm.fit_residuals(residual_data)
