import pytest
import numpy as np

from curvefit.core.residual_model import _ResidualModel, SmoothResidualModel


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
        covariates={'far_out': 'far_out >= 10', 'num_data': None},
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

