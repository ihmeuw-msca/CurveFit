import numpy as np
import pandas as pd


def sizes_to_indices(sizes):
    """Converting sizes to corresponding indices.
    Args:
        sizes (numpy.dnarray):
            An array consist of non-negative number.
    Returns:
        list{range}:
            List the indices.
    """
    indices = []
    a = 0
    b = 0
    for i, size in enumerate(sizes):
        b += size
        indices.append(range(a, b))
        a += size

    return indices


def get_obs_se(df, col_t, func=lambda x: 1 / (1 + x)):
    """
    Get observation standard deviation based on some function
    Args:
        df:
        col_t:
        func: callable

    Returns:

    """
    data = df.copy()
    data['obs_se'] = data[col_t].apply(func)
    return data


def get_derivative_of_column_in_log_space(df, col_obs, col_t, col_grp):
    """
    Adds a new column for the derivative of col_obs.
    Col_obs needs to be in log space. # TODO: Change this later to allow for other spaces.

    Args:
        df: (pd.DataFrame) data frame
        col_obs: (str) observation column to get derivative of
        col_grp: (str) group column
        col_t: (str) time column

    Returns:
        pd.DataFrame
    """
    df.sort_values([col_grp, col_t], inplace=True)
    groups = df[col_grp].unique()
    new_col = f'd {col_obs}'

    # partition the data frame by group
    df_all = {}
    for g in groups:
        df_all.update({
            g: df[df[col_grp] == g].reset_index(drop=True)
        })
    # for each location compute the log daily death rate increments
    for g in groups:
        df_g = df_all[g]
        obs_now = np.exp(df_g[col_obs]).values
        obs_pre = np.insert(obs_now[:-1], 0, 0.0)
        t_now = df_g[col_t].values
        t_pre = np.insert(t_now[:-1], 0, -1.0)
        ln_slope = np.log(np.maximum(1e-10, (obs_now - obs_pre)/(t_now - t_pre)))
        df_g[new_col] = ln_slope
        df_all[g] = df_g
    # combine all the data frames
    df_result = pd.concat([df_all[g] for g in groups])
    return df_result
