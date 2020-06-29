import pandas as pd
import numpy as np


def local_deviations(df, col_val,
                     col_axis, axis_offset=None, radius=None,
                     robust=True):
    """
    {begin_markdown local_deviations}

    {spell_markdown utils bool}

    # `curvefit.utils.smoothing.local_deviations`
    ## Computes standard deviation within a neighborhood of covariate values

    Compute standard deviation of residuals within a neighborhood
    defined by `col_axis`. Optionally use the median absolute deviation
    as a robust estimator of the standard deviation.

    ## Arguments

    - `df (pd.DataFrame)`: Residual data frame.
    - `col_val (str)`: Name for column that store the residual.
    - `col_axis (List[str])`: List of two axis column names.
    - `axis_offset (List[int] | None, optional)`:
        List of offset for each axis to make it suitable as numpy array.
    - `radius (List[int] | None, optional)`:
        List of the neighbor radius for each dimension.
    - `robust (bool)`: use the median absolute deviation * 1.4826 as a robust estimator for the standard
        deviation

    ## Returns

    - `pd.DataFrame`: return the data frame an extra column for the median absolute deviation
            within a range

    {end_markdown local_deviations}
    """
    axis_offset = [0, 0] if axis_offset is None else axis_offset
    radius = [1, 1] if radius is None else radius

    assert col_val in df
    assert len(col_axis) == 2
    assert len(axis_offset) == 2
    assert len(radius) == 2

    assert all([col in df for col in col_axis])
    assert all([isinstance(offset, int) for offset in axis_offset])
    assert all([isinstance(r, int) for r in radius])

    index = np.unique(np.asarray(df[col_axis].values), axis=0).astype(int)
    new_df = pd.DataFrame({
        col_axis[0]: index[:, 0],
        col_axis[1]: index[:, 1],
        'residual_std': np.nan
    })
    for j in index:
        df_filter = df.copy()
        for k, ax in enumerate(col_axis):
            rad = radius[k]
            ax_filter = np.abs(df[col_axis[k]] - j[k]) <= rad
            df_filter = df_filter.loc[ax_filter]

        if robust:
            std = df_filter[col_val].mad() * 1.4826
        else:
            std = df_filter[col_val].std(ddof=0)

        subset = np.all(new_df[col_axis] == j, axis=1).values
        new_df.loc[subset, 'residual_std'] = std

    return new_df


def local_smoother(df,
                   col_val,
                   col_axis,
                   radius=None):
    """
    {begin_markdown local_smoother}

    {spell_markdown utils}

    # `curvefit.utils.smoothing.local_smoother`
    ## Runs a local smoother over a neighborhood of covariate values

    Runs a local smoother over a neighborhood
    defined by `col_axis`, where the neighborhood is defined by radius.

    ## Arguments

    - `df (pd.DataFrame)`: data frame with values to smooth over
    - `col_val (str)`: name for column that stores the value to smooth over
    - `col_axis (List[str])`: list of column names that store the axes corresponding
        to the variables that define the neighborhood
    - `radius (List[int] | None, optional)`: list of the neighbor radius for each axis dimension

    ## Returns

    - `pd.DataFrame`: return the data frame with an extra column for the smoothed value column
        called `col_val + "mean"`

    {end_markdown local_smoother}
    """
    radius = [0, 0] if radius is None else radius
    assert col_val in df
    assert len(col_axis) == 2
    assert len(radius) == 2
    assert all([col in df for col in col_axis])
    assert all([r >= 0 for r in radius])

    col_mean = '_'.join([col_val, 'mean'])

    # group by the axis
    df = df.groupby(col_axis, as_index=False).agg({
        col_val: [np.sum, 'count']
    })

    col_sum = '_'.join([col_val, 'sum'])
    col_count = '_'.join([col_val, 'count'])

    df.columns = df.columns.droplevel(1)
    df.columns = list(df.columns[:-2]) + [col_sum, col_count]

    sum_mat, indices, axis = df_to_mat(
        df, col_val=col_sum, col_axis=col_axis,
        return_indices=True
    )
    count_mat = df_to_mat(
        df, col_val=col_count, col_axis=col_axis
    )

    sum_vec = convolve_sum(sum_mat, radius)[indices[:, 0], indices[:, 1]]
    count_vec = convolve_sum(count_mat, radius)[indices[:, 0], indices[:, 1]]

    df[col_mean] = sum_vec/count_vec
    df.drop(columns=[col_sum, col_count], inplace=True)

    return df


def convolve_sum(mat, radius=None):
    """
    {begin_markdown convolve_sum}

    {spell_markdown utils convolve ndarray convolved}

    # `curvefit.utils.smoothing.convolve_sum`
    ## Convolve sum a 2D matrix by given radius.

    ## Arguments

    - `mat (numpy.ndarray)`: matrix of interest
    - `radius (array-like{int} | None, optional)`: given radius, if None assume radius = (0, 0)

    ## Returns

    - `numpy.ndarray`: the convolved sum, with the same shape with original matrix.

    {end_markdown convolve_sum}
    """
    mat = np.array(mat).astype(float)
    assert mat.ndim == 2
    if radius is None:
        return mat
    assert hasattr(radius, '__iter__')
    radius = np.array(radius).astype(int)
    assert radius.size == 2
    assert all([r >= 0 for r in radius])

    shape = np.array(mat.shape)
    window_shape = tuple(radius*2 + 1)

    mat = np.pad(mat, ((radius[0],),
                       (radius[1],)), 'constant', constant_values=np.nan)
    view_shape = tuple(np.subtract(mat.shape, window_shape) + 1) + window_shape
    strides = mat.strides*2
    sub_mat = np.lib.stride_tricks.as_strided(mat, view_shape, strides)
    sub_mat = sub_mat.reshape(*shape, np.prod(window_shape))

    return np.nansum(sub_mat, axis=2)


def df_to_mat(df, col_val, col_axis, return_indices=False):
    """
    {begin_markdown df_to_mat}

    {spell_markdown utils bool ndarray}

    # `curvefit.utils.smoothing.df_to_mat`
    ## Convert columns in data frame to matrix

    ## Arguments

    - `df (pandas.DataFrame)`: given data frame.
    - `col_val (str)`: value column.
    - `col_axis (List[str])`: axis column.
    - `return_indices (bool, optional)`: if True, return indices of the original values and the corresponding
            axis values in the data frame.

    ## Returns

    - `(numpy.ndarray)`: converted matrix

    {end_markdown df_to_mat}
    """
    assert col_val in df
    assert all([c in df for c in col_axis])

    values = df[col_val].values
    axis = df[col_axis].values.astype(int)
    indices = (axis - axis.min(axis=0)).astype(int)
    shape = tuple(indices.max(axis=0).astype(int) + 1)

    mat = np.empty(shape)
    mat.fill(np.nan)
    mat[indices[:, 0], indices[:, 1]] = values

    if return_indices:
        return mat, indices, axis
    else:
        return mat
