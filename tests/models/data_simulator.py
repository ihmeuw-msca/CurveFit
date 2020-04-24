import pandas as pd
import numpy as np

from curvefit.core.data import DataSpecs
from curvefit.core.functions import ln_gaussian_cdf
from curvefit.core.parameter import Variable, Parameter, ParameterSet

def generate_data(seed=123):
    np.random.seed(seed)
    df = pd.DataFrame({
        't': np.arange(5),
        'obs': np.random.rand(5),
        'group': 'All',
        'intercept': np.ones(5),
        'se': np.random.randn(5) * 1e-3,

    })
    data_specs = DataSpecs('t', 'obs', ['intercept'], 'group', ln_gaussian_cdf, 'se')
    return df, data_specs

def generate_parameter_set():
    variable = Variable('intercept', lambda x:x, 0.0, 0.0)
    parameter = Parameter('p1', np.exp, [variable])
    parameter_set = ParameterSet([parameter] * 3)
    assert parameter_set.num_fe == 3
    return parameter_set


