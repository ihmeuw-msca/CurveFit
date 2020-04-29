import numpy as np
import pandas as pd
from copy import deepcopy
from collections import OrderedDict
from curvefit.core.functions import *
from curvefit.utils.data import data_translator


def sizes_to_indices(sizes):
    """
    {begin_markdown sizes_to_indices}
    {spell_markdown subvector subvectors iterable}
    # `curvefit.core.utils.sizes_to_indices`
    ## Converting sizes to corresponding indices.

    ## Syntax
    `indices = curvefit.sizes_to_indices(sizes)`

    ## Arguments

    - `sizes (iterable)`: The argument *sizes* is an iterable object with integer values.
        The i-th value in `sizes[i]` is the number of elements in the i-th
        subvector of a larger total vector that contains the subvectors in order.

    ## Returns

    - `indices`: The return value *indices* is a `list` of one dimensional numpy arrays.
        The value `indices[i]` has length equal to the i-th size.
        It starts (ends) with the index in the total vector
        of the first (last) element of the i-th subvector.  The elements of
        `indices[i]` are monotone and increase by one between elements.

    ## Example
    [sizes_to_indices_xam](sizes_to_indices_xam.md)

    {end_markdown sizes_to_indices}
    """
    indices = []
    a = 0
    b = 0
    for i, size in enumerate(sizes):
        b += size
        indices.append(np.arange(a, b))
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


def get_derivative_of_column_in_ln_space(df, col_obs, col_t, col_grp):
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
        ln_slope = np.log(np.maximum(1e-10, (obs_now - obs_pre) / (t_now - t_pre)))
        df_g[new_col] = ln_slope
        df_all[g] = df_g
    # combine all the data frames
    df_result = pd.concat([df_all[g] for g in groups])
    return df_result


def cumulative_derivative(array):
    arr = array.copy()
    return arr - np.insert(arr[:, :-1], 0, 0.0, axis=1)


def solve_p_from_dgaussian_pdf(alpha, beta, slopes, slope_at=14):
    """Compute p from alpha, beta and slopes of gaussian_pdf at given point.

    Args:
        alpha (np.ndarray | float):
            Array of alpha values.
        beta (np.ndarray | float):
            Array of beta values.
        slopes (np.ndarray | float):
            Array of slopes
        slope_at (float | int, optional):
            Point where slope is calculated.

    Returns:
        np.ndarray | float:
            The corresponding p value.
    """
    is_scalar = np.isscalar(alpha)

    alpha = np.array([alpha]) if np.isscalar(alpha) else alpha
    beta = np.array([beta]) if np.isscalar(beta) else beta

    assert alpha.size == beta.size
    assert (alpha > 0.0).all()

    if np.isscalar(slopes):
        slopes = np.repeat(slopes, alpha.size)

    assert alpha.size == slopes.size
    assert all(slopes > 0.0)
    assert all(beta >= slope_at)

    tmp = alpha*(slope_at - beta)
    p = np.sqrt(np.pi)*slopes/(2.0*alpha**2*np.abs(tmp)*np.exp(-tmp**2))

    if is_scalar:
        p = p[0]

    return p


def sample_from_samples(samples, sample_size):
    """Sample from given samples.

    Args:
        samples (np.ndarray):
            Given samples, assume to be 1D array.
        sample_size (int):
            Number of samples want to predict.

    Returns:
        new_samples (np.ndarray):
            Generated new samples.
    """
    mean = np.mean(samples)
    std = np.std(samples)

    new_samples = mean + np.random.randn(sample_size) * std

    return new_samples


def truncate_draws(t, draws, draw_space, last_day, last_obs, last_obs_space):
    """Truncating draws to the given last day and last obs.

    Args:
        t (np.ndarray):
            Time variables for the draws.
        draws (np.ndarray):
            Draws matrix.
        draw_space (str | callable):
            Which space is the draw in.
        last_day (int | float):
            From which day, should the draws start.
        last_obs (int | float):
            From which observation value, should the draws start.
        last_obs_space (str | callable):
            Which space is the last observation in.

    Returns:
        np.ndarray:
            Truncated draws.
    """
    draw_ndim = draws.ndim
    if draw_ndim == 1:
        draws = draws[None, :]

    assert draws.shape[1] == t.size

    if callable(draw_space):
        draw_space = draw_space.__name__
    if callable(last_obs_space):
        last_obs_space = last_obs_space.__name__

    assert draw_space in ['gaussian_cdf', 'gaussian_pdf', 'ln_gaussian_cdf', 'ln_gaussian_pdf']
    assert last_obs_space in ['gaussian_cdf', 'gaussian_pdf', 'ln_gaussian_cdf', 'ln_gaussian_pdf']

    if last_obs_space == 'gaussian_cdf':
        assert last_obs >= 0.0
    else:
        last_obs = np.exp(last_obs)

    last_day = int(np.round(last_day))
    assert t.min() <= last_day < t.max()

    gaussian_pdf_draws = data_translator(draws, draw_space, 'gaussian_pdf')
    gaussian_pdf_draws = gaussian_pdf_draws[:, last_day + 1:]

    if draw_space == 'gaussian_pdf':
        final_draws = gaussian_pdf_draws
    elif draw_space == 'ln_gaussian_pdf':
        final_draws = data_translator(gaussian_pdf_draws, 'gaussian_pdf', 'ln_gaussian_pdf')
    elif draw_space == 'gaussian_cdf':
        assert last_obs_space in ['gaussian_cdf', 'ln_gaussian_cdf']
        last_obs = last_obs if last_obs_space == 'gaussian_cdf' else np.exp(last_obs)
        final_draws = data_translator(gaussian_pdf_draws, 'gaussian_pdf', 'gaussian_cdf') + last_obs
    else:
        assert last_obs_space in ['gaussian_cdf', 'ln_gaussian_cdf']
        last_obs = last_obs if last_obs_space == 'gaussian_cdf' else np.exp(last_obs)
        final_draws = data_translator(gaussian_pdf_draws, 'gaussian_pdf', 'gaussian_cdf') + last_obs
        final_draws = np.log(final_draws)

    if draw_ndim == 1:
        final_draws = final_draws.ravel()

    return final_draws


def smooth_draws(mat, radius=0, sort=False):
    """Smooth the draw matrix in the column direction.

    Args:
        mat (np.ndarray):
            Input matrix, either 1d or 2d array.
        radius (int, optional):
            Smoothing radius.
        sort (bool, optional):
            If `sort`, we sorting the matrix along the first dimension before
            smoothing.

    Returns:
        np.ndarray:
            Smoothed matrix.
    """
    mat = np.array(mat).copy()
    if radius == 0:
        return mat

    radius = radius if mat.ndim == 1 else (0, radius)

    if sort and mat.ndim == 2:
        mat.sort(axis=0)

    return smooth_mat(mat, radius=radius)


def smooth_mat(mat, radius=None):
    """Smooth the draw matrix in the column direction.

        Args:
            mat (np.ndarray):
                Input matrix, either 1d or 2d array.
            radius (int | tuple{int} | None, optional):
                Smoothing radius.

        Returns:
            np.ndarray:
                Smoothed matrix.
    """
    mat = np.array(mat).copy()

    is_vector = mat.ndim == 1
    if is_vector:
        if isinstance(radius, int):
            radius = (0, radius)
        elif isinstance(radius, tuple):
            assert len(radius) == 1
            radius = (0, radius[0])
        else:
            RuntimeError('Wrong input of radius.')
        mat = mat[None, :]

    assert len(radius) == mat.ndim

    shape = mat.shape

    window_shape = tuple(np.array(radius)*2 + 1)
    mat = np.pad(mat, ((radius[0],), (radius[1],)), 'constant',
                 constant_values=np.nan)
    view_shape = tuple(
        np.subtract(mat.shape, window_shape) + 1) + window_shape
    strides = mat.strides + mat.strides
    sub_mat = np.lib.stride_tricks.as_strided(mat, view_shape, strides)
    sub_mat = sub_mat.reshape(*shape, np.prod(window_shape))

    mean = np.nanmean(sub_mat, axis=2)

    if is_vector:
        mean = mean.ravel()

    return mean


def split_by_group(df, col_group):
    """{begin_markdown split_by_group}
    {spell_markdown dataframe}
    # Split the dataframe by the group definition.

    ## Syntax
    `data = split_by_group(df, col_group)`

    ## df
    Provided dataframe.

    ## col_group
    Column name in the dataframe contains group definition.

    ## data
    Dictionary with key as the group definition and value as the
    corresponding dataframe.

    ## Example

    {end_markdown split_by_group}"""
    assert col_group in df
    data = {
        group: df[df[col_group] == group].reset_index(drop=True)
        for group in df[col_group].unique()
    }

    return data


def filter_death_rate(df, col_t, col_death_rate):
    """Filter cumulative death rate. Remove non-monotonically increasing points.

    Args:
        df (pd.DataFrame): Provided data frame.
        col_t (str): Column name of the independent variable.
        col_death_rate (str): Name for column that contains the death rate.

    Returns:
        pd.DataFrame: Filtered data frame.
    """
    df = df.sort_values(col_t).reset_index(drop=True)
    t = df[col_t]
    death_rate = df[col_death_rate]
    drop_idx = [i for i in range(1, t.size)
                if np.any(death_rate[i] <= death_rate[:i])]
    df = df.drop(drop_idx).reset_index(drop=True)
    return df


def filter_death_rate_by_group(df, col_group, col_t, col_death_rate):
    """Filter cumulative death rate within each group.
    Args:
        df (pd.DataFrame): Provided data frame.
        col_group (str): Column name of group definition.
        col_t (str): Column name of the independent variable.
        col_death_rate (str): Name for column that contains the death rate.

    Returns:
        pd.DataFrame: Filtered data frame.
    """
    df_split = list(split_by_group(df, col_group).values())
    for i, df_sub in enumerate(df_split):
        df_split[i] = filter_death_rate(df_sub, col_t, col_death_rate)

    return pd.concat(df_split)


def create_potential_peaked_groups(df, col_group, col_t, col_death_rate,
                                   tol_num_obs=20,
                                   tol_after_peak=3,
                                   return_poly_fit=False):
    """Create potential peaked groups.
    Args:
        df (pd.DataFrame): Provided data frame.
        col_group (str): Column name of group definition.
        col_t (str): Column name of the independent variable.
        col_death_rate (str): Name for column that contains the death rate.
        tol_num_obs (int, optional):
            Only the ones with number of observation above or equal to this
            threshold will be considered as the potential peaked group.
        tol_after_peak (int | float, optional):
            Pick the ones already pass peaked day for this amount of time.
        return_poly_fit (bool, optional):
            If True, return the spline fits as well.
    Returns:
        list | tuple(list, dict):
            List of potential peaked groups or with the spline fit as well.
    """
    data = process_input(df, col_group, col_t, col_death_rate, return_df=False)

    poly_fit = {}
    for location in data.keys():
        df = data[location]
        t = df['days']
        y = df['ln asddr']

        c = np.polyfit(t, y, 2)
        poly_fit.update({
            location: deepcopy(c)
        })

    potential_groups = []
    for i, (location, df) in enumerate(data.items()):
        c = poly_fit[location]
        last_day = df['days'].max()
        num_obs = df.shape[0]
        b = np.inf if np.isclose(c[0], 0.0) else -0.5*c[1]/c[0]
        if c[0] < 0.0 <= b and num_obs >= tol_num_obs and last_day - b >= tol_after_peak:
            potential_groups.append(location)

    if return_poly_fit:
        return potential_groups, poly_fit
    else:
        return potential_groups


def process_input(df, col_group, col_t, col_death_rate, return_df=True):
    """
    Trim filter and adding extra information to the data frame.

    Args:
        df (pd.DataFrame): Provided data frame.
        col_group (str): Column name of group definition.
        col_t (str): Column name of the independent variable.
        col_death_rate (str): Name for column that contains the death rate.
        return_df (bool, optional):
            If True return the combined data frame, otherwise return the
            splitted dictionary.

    Returns:
        pd.DataFrame: processed data frame.
    """
    assert col_group in df
    assert col_t in df
    assert col_death_rate in df

    # trim down the data frame
    df = df[[col_group, col_t, col_death_rate]].reset_index(drop=True)
    df.sort_values([col_group, col_t], inplace=True)
    df.columns = ['location', 'days', 'ascdr']

    # check and filter and add more information
    data = split_by_group(df, col_group='location')
    for location, df_location in data.items():
        assert df_location.shape[0] == df_location['days'].unique().size
        df_location = filter_death_rate(df_location,
                                        col_t='days',
                                        col_death_rate='ascdr')
        df_location['ln ascdr'] = np.log(df_location['ascdr'])
        df_location['asddr'] = df_location['ascdr'].values - \
            np.insert(df_location['ascdr'].values[:-1], 0, 0.0)
        df_location['ln asddr'] = np.log(df_location['asddr'])

        data.update({
            location: df_location.copy()
        })

    if return_df:
        return pd.concat(list(data.values()))
    else:
        return data


def peak_score(t, y, c, num_obs,
               tol_num_obs=5,
               weight_num_obs=1.0,
               min_score=0.1,
               max_score=1.0,
               lslope=0.1,
               rslope=0.1):
    """Compute the peak score of give prediction.

    Args:
        t (numpy.ndarray): Time array.
        y (numpy.ndarray): Prediction in the daily death space.
        c (numpy.ndarray): The coefficient of the polyfit.
        num_obs (int): Number of the observations.
        tol_num_obs (int, optional):
            If num_obs lower than this value, then assign equal weights.
        weight_num_obs (float, optional):
            Weight for importance of the number of observations.
        min_score (float, optional): Minimum score, required to be positive.
        max_score (float, optional):
            Maximum score, required greater than min_score.
        lslope (float, optional): Slope for underestimate the peak time.
        rslope (float, optional): Slope for overestimate the peak time.

    Returns:
        float: The score.
    """
    assert isinstance(t, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert isinstance(c, np.ndarray)
    assert t.size == y.size
    assert c.size == 3
    assert num_obs >= 1.0
    assert tol_num_obs >= 0.0
    assert 0.0 <= weight_num_obs <= 1.0
    assert min_score >= 0.0
    assert max_score >= min_score
    assert lslope >= 0.0
    assert rslope >= 0.0

    b = -0.5*c[1]/c[0]
    beta = t[np.argmax(y)]
    if np.isclose(c[0], 0.0) or c[0] > 0.0 or num_obs <= tol_num_obs or b <= 0.0:
        return 0.5*(min_score + max_score)

    if min_score == max_score:
        return min_score

    height = max_score - min_score

    score = min_score + height*(1.0 - weight_num_obs/num_obs)*np.exp(-(
        lslope*min(beta - b, 0.0)**2 +
        rslope*max(beta - b, 0.0)**2
    ))

    return score
