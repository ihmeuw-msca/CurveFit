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


def neighbor_mean_std(df,
                      col_val,
                      col_group,
                      col_axis,
                      axis_offset=None,
                      radius=None):
    """Compute the neighbor mean and std of the residual matrix.

    Args:
        df (pd.DataFrame): Residual data frame.
        col_val ('str'): Name for column that store the residual.
        col_group ('str'): Name for column that store the group label.
        col_axis (list{str}): List of two axis column names.
        axis_offset (list{int} | None, optional):
            List of offset for each axis to make it suitable as numpy array.
        radius (list{int} | None, optional):
            List of the neighbor radius for each dimension.

    Returns:
        pd.DataFrame:
            Return the data frame with two extra columns contains neighbor
            mean and std.
    """
    axis_offset = [0, 0] if axis_offset is None else axis_offset
    radius = [1, 1] if radius is None else radius
    assert col_val in df
    assert col_group in df
    assert len(col_axis) == 2
    assert len(axis_offset) == 2
    assert len(radius) == 2
    assert all([col in df for col in col_axis])
    assert all([isinstance(offset, int) for offset in axis_offset])
    assert all([isinstance(r, int) for r in radius])

    df_list = [
        df[df[col_group] == group]
        for group in df[col_group].unique()
    ]
    for i, df_sub in enumerate(df_list):
        index = df_sub[col_axis].values.astype(int)
        index += np.array(axis_offset)
        shape = tuple(index.max(axis=0) + 1)

        mat = np.empty(shape)
        mat.fill(np.nan)
        mat[index[:, 0], index[:, 1]] = df_sub[col_val]

        window_shape = tuple(np.array(radius)*2 + 1)
        mat = np.pad(mat, ((radius[0],), (radius[1],)), 'constant',
                     constant_values=np.nan)
        view_shape = tuple(
            np.subtract(mat.shape, window_shape) + 1) + window_shape
        strides = mat.strides + mat.strides
        sub_mat = np.lib.stride_tricks.as_strided(mat, view_shape, strides)
        sub_mat = sub_mat.reshape(*shape, np.prod(window_shape))

        mat_mean = np.nanmean(sub_mat, axis=2)
        mat_std = np.nanstd(sub_mat, axis=2)

        df_sub['residual_mean'] = mat_mean[index[:, 0], index[:, 1]]
        df_sub['residual_std'] = mat_std[index[:, 0], index[:, 1]]

        df_list[i] = df_sub

    return pd.concat(df_list)
