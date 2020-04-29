import pytest
import pandas as pd
import numpy as np
from dataclasses import FrozenInstanceError

from curvefit.core.data import Data, DataSpecs
from curvefit.core.functions import ln_gaussian_pdf, gaussian_pdf


class TestDataSpecs:

    def test_frozen(self):
        specs = DataSpecs('t', 'obs', ['cov'], 'group', ln_gaussian_pdf)
        with pytest.raises(FrozenInstanceError):
            specs.col_t = 'new'


class TestData:
    
    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            'time': np.array([1, 4, 5, 6, 7, 1]),
            'ln_death_rate': np.random.randn(6),
            'covariate_1': 1.,
            'group': np.repeat(['group1', 'group2'], repeats=3)
        })

    def test_data(self, df):
        d = Data(
            df=df,
            col_t='time',
            col_obs='ln_death_rate',
            col_covs=['covariate_1'],
            col_group='group',
            obs_space=ln_gaussian_pdf,
            obs_se_func=lambda x: x ** (-2.)
        )
        assert len(d.df == 6)
        np.testing.assert_equal(d.df.time, np.array([1, 4, 5, 1, 6, 7]))
        np.testing.assert_equal(d.df.group, np.repeat(['group1', 'group2'], repeats=3))
        np.testing.assert_equal(d.df.obs_se, np.array([1, 4, 5, 1, 6, 7]) ** (-2.))

    def test_get_df(self, df):
        d = Data(
            df=df,
            col_t='time',
            col_obs='ln_death_rate',
            col_covs=['covariate_1'],
            col_group='group',
            obs_space=ln_gaussian_pdf,
        )
        pd.testing.assert_frame_equal(d.df, d._get_df())
        pd.testing.assert_frame_equal(d.df[0:3], d._get_df(group='group1'))

    def test_get_translated_observations(self, df):
        d = Data(
            df=df,
            col_t='time',
            col_obs='ln_death_rate',
            col_covs=['covariate_1'],
            col_group='group',
            obs_space=ln_gaussian_pdf,
        )
        np.testing.assert_equal(
            np.exp(d.df[d.col_obs])[0:3],
            d._get_translated_observations(group='group1', space=gaussian_pdf)
        )
