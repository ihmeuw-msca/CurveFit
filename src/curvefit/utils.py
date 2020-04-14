import numpy as np
import pandas as pd
from copy import deepcopy
from collections import OrderedDict
from scipy.optimize import bisect
from .functions import *
try :
	from scipy.stats import median_absolute_deviation
except ImportError :
	# median_absolute_deviation is not in scipy before version 1.3.0
	def median_absolute_deviation(vec, nan_policy='omit', scale=1.4826 ) :
		assert nan_polisy == 'omit'
		assert scale == 1.4826
		assert len( vec.shape ) == 1
		med = numpy.median( vec )
		mad = numpy.median( abs(vec - med) )
		return scale * mad


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
        ln_slope = np.log(np.maximum(1e-10, (obs_now - obs_pre) / (t_now - t_pre)))
        df_g[new_col] = ln_slope
        df_all[g] = df_g
    # combine all the data frames
    df_result = pd.concat([df_all[g] for g in groups])
    return df_result


def across_group_mean_std(df,
                          col_val,
                          col_group,
                          col_axis):
    """Compute the mean and std of the residual matrix across locations.

    Args:
        df (pd.DataFrame): Residual data frame.
        col_val ('str'): Name for column that store the residual.
        col_group ('str'): Name for column that store the group label.
        col_axis (list{str}): List of two axis column names.

    Returns:
        pd.DataFrame:
            Averaged residual mean and std data frame.
    """
    df_list = [
        df[df[col_group] == group].copy()
        for group in df[col_group].unique()
    ]
    for i, df_sub in enumerate(df_list):
        df_sub_result = df_sub.groupby(col_axis, as_index=False).agg(
            {col_val: [np.nanmean, np.nanstd]}
        )
        df_sub_result.columns = [*col_axis, 'mean', 'std']
        df_list[i] = df_sub_result

    return pd.concat(df_list)


def neighbor_mean_std(df,
                      col_val,
                      col_group,
                      col_axis,
                      axis_offset=None,
                      radius=None,
                      compute_mad=False):
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
        compute_mad (bool, optional):
            If compute_mad, also compute median absolute deviation.

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
        df[df[col_group] == group].reset_index()
        for group in df[col_group].unique()
    ]  # separate dataset by groups

    for i, df_sub in enumerate(df_list):
        index = np.unique(np.asarray(df_sub[col_axis].values), axis=0).astype(int)
        new_df = pd.DataFrame({
            'group': df_sub[col_group].iloc[0],
            col_axis[0]: index[:, 0],
            col_axis[1]: index[:, 1],
            'residual_mean': np.nan,
            'residual_std': np.nan
        })
        # Group rows by indices. It's like a 2d matrix, but each cell can have multiple entries (one per Location)
        df_grouped_by_idx = df_sub.reset_index().fillna(0).groupby(by=[col_axis[0], col_axis[1]])
        max_idx = np.max(index)
        groups_locations = np.zeros((max_idx, max_idx)) - 1
        groups = []
        # Since the number of entities for any col_axis is undetermined, we have to keep it in a long list.
        for idx, (name, group) in enumerate(df_grouped_by_idx):
            s = int(name[0]) - 1
            j = int(name[1]) - 1
            groups.append(np.array(group[col_val].to_list()))
            groups_locations[s, j] = idx

        # Iterate over all combination of indices, calculate mean and variance around it.
        for idx, row in new_df.iterrows():
            s = int(row[col_axis[0]]) - 1
            j = int(row[col_axis[1]]) - 1
            total_sum = 0
            total_count = 0
            total_deviations_squared = 0

            for k in range(max(s - radius[0], 0), min(s + radius[0] + 1, groups_locations.shape[0])):
                for t in range(max(j - radius[1], 0), min(j + radius[1] + 1, groups_locations.shape[1])):
                    location = int(groups_locations[k, t])
                    if location == -1:
                        continue
                    residuals = groups[location]
                    total_sum += residuals.sum()
                    total_count += len(residuals)
            if total_count == 0:
                continue
            mean = total_sum / total_count
            new_df.at[idx, 'residual_mean'] = mean
            for k in range(max(s - radius[0], 0), min(s + radius[0] + 1, groups_locations.shape[0])):
                for t in range(max(j - radius[1], 0), min(j + radius[1] + 1, groups_locations.shape[1])):
                    location = int(groups_locations[k, t])
                    if location == -1:
                        continue
                    residuals = groups[location]
                    total_deviations_squared += ((residuals - mean) ** 2).sum()
            std = np.sqrt(total_deviations_squared / (total_count - 1))
            new_df.at[idx, 'residual_std'] = std

        df_list[i] = new_df

    return pd.concat(df_list)


def cumulative_derivative(array):
    arr = array.copy()
    return arr - np.insert(arr[:, :-1], 0, 0.0, axis=1)


def convex_combination(t, pred1, pred2, pred_fun,
                       start_day=2, end_day=20):
    """Combine the prediction.

    Args:
        t (np.ndarray): Time axis for the prediction.
        pred1 (np.ndarray): First set of the prediction.
        pred2 (np.ndarray): Second set of the prediction.
        pred_fun (function): Function that used to generate the prediction.
        start_day (int, optional):
            Which day start to blend, before follow `pred2`.
        end_day (int, optional):
            Which day end to blend, after follow `pred1`.
    """
    pred_ndim = pred1.ndim
    if pred1.ndim == 1:
        pred1 = pred1[None, :]
    if pred2.ndim == 1:
        pred2 = pred2[None, :]

    num_time_points = t.size
    assert pred1.shape == pred2.shape
    assert pred1.shape[1] == num_time_points
    assert callable(pred_fun)
    assert start_day < end_day

    a = 1.0 / (end_day - start_day)
    b = -start_day * a
    lam = np.maximum(0.0, np.minimum(1.0, a * t + b))

    if pred_fun.__name__ == 'log_erf':
        pred1 = np.exp(pred1)
        pred2 = np.exp(pred2)
        pred1_tmp = cumulative_derivative(pred1)
        pred2_tmp = cumulative_derivative(pred2)
        pred_tmp = lam * pred1_tmp + (1.0 - lam) * pred2_tmp
        pred = np.log(np.cumsum(pred_tmp, axis=1))
    elif pred_fun.__name__ == 'erf':
        pred1_tmp = cumulative_derivative(pred1)
        pred2_tmp = cumulative_derivative(pred2)
        pred_tmp = lam * pred1_tmp + (1.0 - lam) * pred2_tmp
        pred = np.cumsum(pred_tmp, axis=1)
    elif pred_fun.__name__ == 'log_derf':
        pred1_tmp = np.exp(pred1)
        pred2_tmp = np.exp(pred2)
        pred_tmp = lam * pred1_tmp + (1.0 - lam) * pred2_tmp
        pred = np.log(pred_tmp)
    elif pred_fun.__name__ == 'derf':
        pred = lam * pred1 + (1.0 - lam) * pred2
    else:
        pred = None
        RuntimeError('Unknown prediction functional form')

    if pred_ndim == 1:
        pred = pred.ravel()

    return pred


def model_average(pred1, pred2, w1, w2, pred_fun):
    """
    Average two models together in linear space.

    Args:
        pred1: (np.array) first set of predictions
        pred2: (np.array) second set of predictions
        w1: (float) weight for first predictions
        w2: (float) weight for second predictions
        pred_fun (function): Function that used to generate the prediction.
    """
    assert callable(pred_fun)
    assert w1 + w2 == 1

    if pred_fun.__name__ == 'log_erf':
        pred1 = np.exp(pred1)
        pred2 = np.exp(pred2)
        pred1_tmp = cumulative_derivative(pred1)
        pred2_tmp = cumulative_derivative(pred2)
        pred_tmp = w1 * pred1_tmp + w2 * pred2_tmp
        pred = np.log(np.cumsum(pred_tmp, axis=1))
    elif pred_fun.__name__ == 'erf':
        pred1_tmp = cumulative_derivative(pred1)
        pred2_tmp = cumulative_derivative(pred2)
        pred_tmp = w1 * pred1_tmp + w2 * pred2_tmp
        pred = np.cumsum(pred_tmp, axis=1)
    elif pred_fun.__name__ == 'log_derf':
        pred1_tmp = np.exp(pred1)
        pred2_tmp = np.exp(pred2)
        pred_tmp = w1 * pred1_tmp + w2 * pred2_tmp
        pred = np.log(pred_tmp)
    elif pred_fun.__name__ == 'derf':
        pred = w1 * pred1 + w2 * pred2
    else:
        pred = None
        RuntimeError('Unknown prediction functional form')

    return pred


def condense_residual_matrix(matrix, sequential_diffs, data_density):
    """
    Condense the residuals from a residual matrix to three columns
    that represent how far out the prediction was, the number of data points,
    and the observed residual.

    Args:
        matrix: (np.ndarray)
        sequential_diffs:
        data_density:

    Returns:
        numpy.ndarray:
            Combined matrix.
    """
    row_idx, col_idx = np.triu_indices(matrix.shape[0], 1)
    map1 = np.cumsum(np.insert(sequential_diffs, 0, 0))
    map2 = data_density

    far_out = map1[col_idx] - map1[row_idx]
    num_data = map2[row_idx]
    robs = matrix[row_idx, col_idx]

    # return the results for the residual matrix as a (len(available_times), 3) shaped matrix
    r_matrix = np.vstack([far_out, num_data, robs]).T
    return r_matrix


def data_translator(data, input_space, output_space,
                    threshold=1e-16):
    """Data translator, move data from one space to the other.

    Args:
        data (np.ndarray): data matrix or vector
        input_space (str | callable): input data space.
        output_space (str | callable): output data space.
        threshold (float, optional):
            Thresholding for the number below 0 in the linear space.

    Returns:
        np.ndarray:
            translated data.
    """
    if callable(input_space):
        input_space = input_space.__name__
    if callable(output_space):
        output_space = output_space.__name__

    total_space = ['erf', 'derf', 'log_erf', 'log_derf']

    assert input_space in total_space
    assert output_space in total_space
    assert isinstance(data, np.ndarray)
    assert threshold > 0.0

    data_ndim = data.ndim
    if data_ndim == 1:
        data = data[None, :]

    # thresholding the data in the linear space
    if input_space in ['erf', 'derf']:
        data = np.maximum(threshold, data)

    if input_space == output_space:
        output_data = data.copy()
    elif output_space == 'log_' + input_space:
        output_data = np.log(data)
    elif input_space == 'log_' + output_space:
        output_data = np.exp(data)
    elif 'derf' in input_space:
        if 'log' in input_space:
            data = np.exp(data)
        output_data = np.cumsum(data, axis=1)
        if 'log' in output_space:
            output_data = np.log(output_data)
    else:
        if 'log' in input_space:
            data = np.exp(data)
        output_data = data - np.insert(data[:, :-1], 0, 0.0, axis=1)
        if 'log' in output_space:
            output_data = np.log(output_data)

    # reverting the shape back if necessary
    if data_ndim == 1:
        output_data = output_data.ravel()

    return output_data


def get_initial_params(model, groups, fit_arg_dict):
    """
    Runs a separate model for each group fixing the random effects to 0
    and calculates what the initial values should be for the optimization
    of the whole model.

    Args:
        model: (curvefit.CurveModel)
        groups: (list) list of groups to get smart starting params for
        fit_arg_dict: keyword arguments in dict that are passed to the
            fit_params function

    Returns:
        (np.array) fe_init: fixed effects initial value
        (np.array) re_init: random effects initial value
    """
    fixed_effects = OrderedDict()
    fit_kwargs = deepcopy(fit_arg_dict)

    # Fit a model for each group with fit_kwargs carried over
    # from the settings for the overall model with a couple of adjustments.
    for g in groups:
        fixed_effects[g] = model.run_one_group_model(group=g, **fit_kwargs)
    return fixed_effects


def compute_starting_params(fe_dict):
    """
    Compute the starting parameters for a dictionary of fixed effects
    by averaging them to get fixed effects for overall model and finding
    deviation from average as the random effect.
    Args:
        fe_dict: OrderedDict of fixed effects to put together that are ordered
            in the way that you want them to go into the model

    Returns:
        (np.array) fe_init: fixed effects initial value
        (np.array) re_init: random effects initial value
    """
    fe_values = []
    for k, v in fe_dict.items():
        fe_values.append(v)
    all_fixed_effects = np.vstack(fe_values)

    # The new fixed effects initial value is the mean of the fixed effects
    # across all single-group models.
    fe_init = all_fixed_effects.mean(axis=0)

    # The new random effects initial value is the single-group models' deviations
    # from the mean, which is now the new fixed effects initial value.
    re_init = (all_fixed_effects - fe_init).ravel()
    return fe_init, re_init


def solve_p_from_dderf(alpha, beta, slopes, slope_at=14):
    """Compute p from alpha, beta and slopes of derf at given point.

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

    # p = np.zeros(alpha.size)
    #
    # for i in range(alpha.size):
    #     x = bisect(lambda x: dderf(slope_at, [alpha[i], beta[i], np.exp(x)]) -
    #                slopes[i], -15.0, 0.0)
    #     p[i] = np.exp(x)

    tmp = alpha*(slope_at - beta)
    p = np.sqrt(np.pi)*slopes/(2.0*alpha**2*np.abs(tmp)*np.exp(-tmp**2))

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

    assert draw_space in ['erf', 'derf', 'log_erf', 'log_derf']
    assert last_obs_space in ['erf', 'derf', 'log_erf', 'log_derf']

    if last_obs_space == 'erf':
        assert last_obs >= 0.0
    else:
        last_obs = np.exp(last_obs)

    last_day = int(np.round(last_day))
    assert last_day >= t.min() and last_day < t.max()

    derf_draws = data_translator(draws, draw_space, 'derf')
    derf_draws = derf_draws[:, last_day + 1:]

    if draw_space == 'derf':
        final_draws = derf_draws
    elif draw_space == 'log_derf':
        final_draws = data_translator(derf_draws, 'derf', 'log_derf')
    elif draw_space == 'erf':
        assert last_obs_space in ['erf', 'log_erf']
        last_obs = last_obs if last_obs_space == 'erf' else np.exp(last_obs)
        final_draws = data_translator(derf_draws, 'derf', 'erf') + last_obs
    else:
        assert last_obs_space in ['erf', 'log_erf']
        last_obs = last_obs if last_obs_space == 'erf' else np.exp(last_obs)
        final_draws = data_translator(derf_draws, 'derf', 'erf') + last_obs
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
    """Split the data frame by the group definition.
    Args:
        df (pd.DataFrame): Provided data frame.
        col_group (str): Column name of group definition.
    Returns:
        dict{str, pd.DataFrame}:
            Dictionary with key as the group definition and value as the
            corresponding data frame.
    """
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
        spline_konts (np.array, optional):
            Knots for the spline fits.
        spline_degree (int, optional):
            Degree for the spline fits.
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
        if c[0] < 0.0 and b >= 0.0 and num_obs >= tol_num_obs and \
            last_day - b >= tol_after_peak:
            potential_groups.append(location)

    if return_poly_fit:
        return potential_groups, poly_fit
    else:
        return potential_groups


def process_input(df, col_group, col_t, col_death_rate, return_df=True):
    """Trim filter and adding extra information to the data frame.
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
            Weight for importancy of the number of observations.
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
    if np.isclose(c[0], 0.0) or \
            c[0] > 0.0 or \
            num_obs <= tol_num_obs or \
            b <= 0.0:
        return 0.5*(min_score + max_score)

    if min_score == max_score:
        return min_score

    height = max_score - min_score

    score = min_score + height*(1.0 - weight_num_obs/num_obs)*np.exp(-(
        lslope*min(beta - b, 0.0)**2 +
        rslope*max(beta - b, 0.0)**2
    ))

    return score
