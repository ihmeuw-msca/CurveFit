# -*- coding: utf-8 -*-
"""
    test utils
    ~~~~~~~~~~

    Test utils module
"""

import numpy as np
import pandas as pd
import pytest
from curvefit.core.functions import gaussian_pdf, dgaussian_pdf
import curvefit.core.utils as utils
from curvefit.utils.data import data_translator


@pytest.mark.parametrize(('sizes', 'indices'),
                         [(np.array([1, 1, 1]), [np.array([0]),
                                                 np.array([1]),
                                                 np.array([2])]),
                          (np.array([1, 2, 3]), [np.array([0]),
                                                 np.array([1, 2]),
                                                 np.array([3, 4, 5])])])
def test_sizes_to_indices(sizes, indices):
    my_indices = utils.sizes_to_indices(sizes)
    print(my_indices)
    assert all([np.allclose(indices[i], my_indices[i])
                for i in range(sizes.size)])


@pytest.mark.parametrize('func', [lambda x: 1 / (1 + x),
                                  lambda x: x**2])
def test_get_obs_se(func):
    data = pd.DataFrame({
        't': np.arange(5),
        'group': 'All',
        'obs': np.ones(5),
        'cov': np.zeros(5)
    })
    result = utils.get_obs_se(data, 't', func=func)
    assert np.allclose(result['obs_se'], func(data['t']))


@pytest.mark.parametrize('data', [np.arange(1, 6)[None, :]])
@pytest.mark.parametrize(('input_space', 'output_space'),
                         [('gaussian_cdf', 'gaussian_pdf'),
                          ('ln_gaussian_cdf', 'ln_gaussian_pdf')])
def test_data_translator_diff(data, input_space, output_space):
    result = data_translator(data, input_space, output_space)
    result[0, 0] = data[0, 0]
    if input_space.startswith('ln'):
        assert np.allclose(np.exp(data), np.cumsum(np.exp(result), axis=1))
    else:
        assert np.allclose(data, np.cumsum(result, axis=1))


@pytest.mark.parametrize('data', [np.arange(1, 6)[None, :]])
@pytest.mark.parametrize(('input_space', 'output_space'),
                         [('gaussian_cdf', 'ln_gaussian_cdf'),
                          ('gaussian_pdf', 'ln_gaussian_pdf')])
def test_data_translator_exp(data, input_space, output_space):
    result = data_translator(data, input_space, output_space)
    assert np.allclose(data, np.exp(result))


@pytest.mark.parametrize('data', [np.arange(1, 6)[None, :]])
@pytest.mark.parametrize(('input_space', 'output_space'),
                         [('gaussian_cdf', 'gaussian_cdf'),
                          ('gaussian_pdf', 'gaussian_pdf'),
                          ('ln_gaussian_cdf', 'ln_gaussian_cdf'),
                          ('ln_gaussian_pdf', 'ln_gaussian_pdf')])
def test_data_translator_exp(data, input_space, output_space):
    result = data_translator(data, input_space, output_space)
    assert np.allclose(data, result)


@pytest.mark.parametrize('alpha', [1.0])
@pytest.mark.parametrize('beta', [5.0])
@pytest.mark.parametrize('slopes', [0.5])
@pytest.mark.parametrize('slope_at', [1, 2, 3])
def test_solve_p_from_dgaussian_pdf(alpha, beta, slopes, slope_at):
    result = utils.solve_p_from_dgaussian_pdf(
        alpha,
        beta,
        slopes,
        slope_at=slope_at
    )
    np.random.seed(100)

    def fun(t, a, b, p, s):
        return dgaussian_pdf(t, [a, b, p]) - s

    assert np.abs(fun(slope_at, alpha, beta, result, slopes)) < 1e-10


def test_split_by_group():
    df = pd.DataFrame({
        'group': ['a', 'a', 'b', 'b'],
        'val': [1.0, 1.0, 2.0, 2.0]
    })

    data = utils.split_by_group(df, 'group')
    assert np.allclose(data['a']['val'].values, 1.0)
    assert np.allclose(data['b']['val'].values, 2.0)


def test_filter_death_rate():
    df = pd.DataFrame({
        't': [0, 1, 2, 3, 4],
        'rate': [0.0, 0.1, 0.0, 0.1, 0.4]
    })

    df_result = utils.filter_death_rate(df, 't', 'rate')
    assert np.allclose(df_result['t'], [0, 1, 4])
    assert np.allclose(df_result['rate'], [0.0, 0.1, 0.4])


def test_filter_death_rate_by_group():
    df = pd.DataFrame({
        'group': ['a', 'a', 'a', 'b', 'b', 'b'],
        't': [0, 1, 2, 0, 1, 2],
        'rate': [0.0, 0.1, 0.2, 0.0, 0.0, 0.2]
    })

    df_result = utils.filter_death_rate_by_group(df, 'group', 't', 'rate')
    assert np.allclose(df_result['t'], [0, 1, 2, 0, 2])
    assert np.allclose(df_result['rate'], [0.0, 0.1, 0.2, 0.0, 0.2])


def test_process_input():
    df = pd.DataFrame({
        'group': ['a', 'a', 'a', 'b', 'b', 'b'],
        't': [0, 1, 2, 0, 1, 2],
        'rate': [0.1, 0.2, 0.3, 0.1, 0.0, 0.3]
    })

    df_result = utils.process_input(df, 'group', 't', 'rate')

    assert 'location' in df_result
    assert 'days' in df_result
    assert 'ascdr' in df_result
    assert 'asddr' in df_result
    assert 'ln ascdr' in df_result
    assert 'ln asddr' in df_result

    days = np.array([0, 1, 2, 0, 2])
    ascdr = np.array([0.1, 0.2, 0.3, 0.1, 0.3])
    asddr = np.array([0.1, 0.1, 0.1, 0.1, 0.2])
    ln_ascdr = np.log(ascdr)
    ln_asddr = np.log(asddr)

    assert np.allclose(df_result['days'], days)
    assert np.allclose(df_result['ascdr'], ascdr)
    assert np.allclose(df_result['asddr'], asddr)
    assert np.allclose(df_result['ln ascdr'], ln_ascdr)
    assert np.allclose(df_result['ln asddr'], ln_asddr)
