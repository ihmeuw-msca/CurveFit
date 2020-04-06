# -*- coding: utf-8 -*-
"""
    test utils
    ~~~~~~~~~~

    Test utils module
"""

import numpy as np
import pandas as pd
from curvefit.legacy.utils import neighbor_mean_std as old_algorithm
from curvefit.utils import neighbor_mean_std as new_algorithm


def generate_testing_problem(locations=("USA", "Europe", "Asia"),
                             timelines=(10, 20, 30),
                             seed=42):
    """
    Generates sample problem for testing utils.neighbor_mean_std function. The columns are:
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


def test_neighbor_mean_std_consistent_with_old_algorithm():
    """
    Compares that new (Aleksei's) algorithm works consistently with old (Peng's) algorithm

    Returns:
        None
    """
    data = generate_testing_problem()
    old_alg_result = old_algorithm(data,
                                   col_axis=['far_out', 'num_data'],
                                   col_val='residual',
                                   col_group='group',
                                   radius=[2, 2]
                                   )
    new_alg_result = new_algorithm(data,
                                   col_axis=['far_out', 'num_data'],
                                   col_val='residual',
                                   col_group='group',
                                   radius=[2, 2]
                                   )

    assert np.allclose(old_alg_result["residual_mean"], new_alg_result["residual_mean"])
    assert np.allclose(old_alg_result["residual_std"], new_alg_result["residual_std"])
    return None
