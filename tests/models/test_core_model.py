import pytest
import numpy as np
import pandas as pd 

from curvefit.core.data import DataSpecs
from curvefit.core.functions import normal_loss, st_loss, ln_gaussian_cdf, ln_gaussian_pdf, gaussian_cdf, gaussian_pdf
from curvefit.models.core_model import Model

from data_simulator import generate_data, generate_parameter_set

class TestCoreModel:


    @pytest.mark.parametrize('curve_fun', [
        ln_gaussian_cdf, 
        ln_gaussian_pdf,
        gaussian_cdf,
        gaussian_pdf,
    ])
    @pytest.mark.parametrize('loss_fun', [normal_loss, st_loss])
    def test_core_model_sanity(self, curve_fun, loss_fun):
        data = generate_data()
        param_set = generate_parameter_set()
        model = Model(param_set, curve_fun, loss_fun)
        x0 = np.array([0.0] * param_set.num_fe * 2)
        model.objective(x0, data)




