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


@pytest.fixture()
def testing_problem(locations=("USA", "Europe", "Asia"),
                    timelines=(10, 20, 30),
                    seed=42):
    """ Generates sample problem for testing utils.neighbor_mean_std function.
    The columns are:
        - 'group': group parameter,
        - 'far_out': first axis,
        - 'num_data': second axis,
        - 'residual': value to aggregate, generated from U[0, 1]

    Args:
        locations: Set{String}
            Locations, group parameter.
        timelines: Set{int}
            How many data points to generate per location
        seed: int
            Random seed

    Returns:
        new_df: pd.DataFrame
            Random dataset suitable for testing neighbor_mean_std function.
    """
    far_out = []
    num_data = []
    location = []
    residual = []
    np.random.seed(seed)
    for t, place in zip(timelines, locations):
        for horizon in np.arange(1, t):
            far_out += [horizon] * (t - horizon)
            num_data += np.arange(1, t - horizon + 1).tolist()
            location += [place] * (t - horizon)
            residual += np.random.rand(t - horizon).tolist()
    new_df = pd.DataFrame({
        'group': location,
        'far_out': far_out,
        'num_data': num_data,
        'residual': residual,
    })
    return new_df


def test_neighbor_mean_std(testing_problem):
    data = testing_problem
    my_result = utils.neighbor_mean_std(
        data,
        col_axis=['far_out', 'num_data'],
        col_val='residual',
        col_group='group',
        radius=[2, 2]
    )
    cols = ['far_out', 'num_data', 'group', 'residual_mean', 'residual_std']
    assert all([c in my_result for c in cols])


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


@pytest.mark.parametrize('t', [np.arange(5)])
@pytest.mark.parametrize(('start_day', 'end_day', 'pred_fun'),
                         [(1, 3, gaussian_pdf)])
@pytest.mark.parametrize(('mat1', 'mat2', 'result'),
                         [(np.ones(5), np.ones(5), np.ones(5)),
                          (np.arange(5), np.ones(5),
                           np.array([1.0, 1.0, 1.5, 3.0, 4.0]))])
def test_convex_combination(t, mat1, mat2, pred_fun, start_day, end_day,
                            result):
    my_result = utils.convex_combination(t, mat1, mat2, pred_fun,
                                         start_day=start_day,
                                         end_day=end_day)

    assert np.allclose(result, my_result)

@pytest.mark.parametrize(('w1', 'w2', 'pred_fun'),
                         [(0.3, 0.7, gaussian_pdf)])
@pytest.mark.parametrize(('mat1', 'mat2', 'result'),
                         [(np.ones(5), np.ones(5), np.ones(5)),
                          (np.ones(5), np.zeros(5), np.ones(5)*0.3),
                          (np.zeros(5), np.ones(5), np.ones(5)*0.7)])
def test_model_average(mat1, mat2, w1, w2, pred_fun, result):
    my_result = utils.model_average(mat1, mat2, w1, w2, pred_fun)
    assert np.allclose(result, my_result)


@pytest.mark.parametrize('mat', [np.arange(9).reshape(3, 3)])
@pytest.mark.parametrize(('radius', 'result'),
                         [((0, 0), np.arange(9).reshape(3, 3)),
                          ((1, 1), np.array([[ 8, 15, 12],
                                            [21, 36, 27],
                                            [20, 33, 24]]))])
def test_convolve_sum(mat, radius, result):
    my_result = utils.convolve_sum(mat, radius=radius)
    assert np.allclose(result, my_result)


def test_df_to_mat():
    df = pd.DataFrame({
        'val': np.ones(5),
        'axis0': np.arange(5, dtype=float),
        'axis1': np.arange(5, dtype=float)
    })

    my_result, indices, axis = utils.df_to_mat(df, 'val', ['axis0', 'axis1'],
                                               return_indices=True)
    assert np.allclose(my_result[indices[:, 0], indices[:, 1]], 1.0)


@pytest.mark.parametrize('radius', [[1, 1]])
def test_local_smoother(radius):
    data = pd.DataFrame({
        'val': np.arange(5),
        'axis0': np.arange(5),
        'axis1': np.arange(5)
    })
    result = utils.local_smoother(data, 'val', ['axis0', 'axis1'],
                                  radius=radius)
    assert np.allclose(result['val_' + 'mean'].values,
                       np.array([0.5, 1.0, 2.0, 3.0, 3.5]))
    assert np.allclose(result['val_' + 'std'].values,
                       np.array([np.std([0.0, 1.0]),
                                 np.std([0.0, 1.0, 2.0]),
                                 np.std([1.0, 2.0, 3.0]),
                                 np.std([2.0, 3.0, 4.0]),
                                 np.std([3.0, 4.0])]))


@pytest.mark.parametrize('data', [np.arange(1, 6)[None, :]])
@pytest.mark.parametrize(('input_space', 'output_space'),
                         [('gaussian_cdf', 'gaussian_pdf'),
                          ('ln_gaussian_cdf', 'ln_gaussian_pdf')])
def test_data_translator_diff(data, input_space, output_space):
    result = utils.data_translator(data, input_space, output_space)
    if 'log' in input_space:
        assert np.allclose(np.exp(data), np.cumsum(np.exp(result), axis=1))
    else:
        assert np.allclose(data, np.cumsum(result, axis=1))


@pytest.mark.parametrize('data', [np.arange(1, 6)[None, :]])
@pytest.mark.parametrize(('input_space', 'output_space'),
                         [('gaussian_cdf', 'ln_gaussian_cdf'),
                          ('gaussian_pdf', 'ln_gaussian_pdf')])
def test_data_translator_exp(data, input_space, output_space):
    result = utils.data_translator(data, input_space, output_space)
    assert np.allclose(data, np.exp(result))


@pytest.mark.parametrize('data', [np.arange(1, 6)[None, :]])
@pytest.mark.parametrize(('input_space', 'output_space'),
                         [('gaussian_cdf', 'gaussian_cdf'),
                          ('gaussian_pdf', 'gaussian_pdf'),
                          ('ln_gaussian_cdf', 'ln_gaussian_cdf'),
                          ('ln_gaussian_pdf', 'ln_gaussian_pdf')])
def test_data_translator_exp(data, input_space, output_space):
    result = utils.data_translator(data, input_space, output_space)
    assert np.allclose(data, result)


@pytest.mark.parametrize('alpha', [1.0])
@pytest.mark.parametrize('beta', [5.0])
@pytest.mark.parametrize('slopes', [0.5])
@pytest.mark.parametrize('slope_at', [1, 2, 3])
def test_solve_p_from_dgaussian_pdf(alpha, beta, slopes, slope_at):
    result = utils.solve_p_from_dgaussian_pdf(alpha,
                                      beta,
                                      slopes,
                                      slope_at=slope_at)
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
