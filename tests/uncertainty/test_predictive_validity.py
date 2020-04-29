from typing import Dict

from curvefit.core.functions import ln_gaussian_cdf, gaussian_cdf
from curvefit.uncertainty.predictive_validity import PredictiveValidity


def test_predictive_validity_theta():
    pv = PredictiveValidity(evaluation_space=ln_gaussian_cdf)
    assert pv.theta == 0.

    pv = PredictiveValidity(evaluation_space=gaussian_cdf)
    assert pv.theta == 1.


def test_predictive_validity():
    pv = PredictiveValidity(evaluation_space=ln_gaussian_cdf)
    assert not pv.debug_mode

    assert isinstance(pv.group_residuals, Dict)
    assert isinstance(pv.group_records, Dict)


