import numpy as np
import pandas as pd


def neighbor_mean_std_v1(df,
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
    ]   # separate dataset by groups

    for i, df_sub in enumerate(df_list):

        index = np.unique(np.asarray(df_sub[col_axis].values), axis=0).astype(int)
        new_df = pd.DataFrame({
            'group': df_sub[col_group].iloc[0],
            col_axis[0]: index[:, 0],
            col_axis[1]: index[:, 1],
            'residual_mean': np.nan,
            'residual_std': np.nan
        })
        for j in index:
            print(j, end='\r')
            df_filter = df_sub.copy()
            for k, ax in enumerate(col_axis):
                rad = radius[k]
                ax_filter = np.abs(df_sub[col_axis[k]] - j[k]) <= rad
                df_filter = df_filter.loc[ax_filter]
            mean = df_filter[col_val].mean()
            std = df_filter[col_val].std()
            subset = np.all(new_df[col_axis] == j, axis=1).values
            new_df.loc[subset, 'residual_mean'] = mean
            new_df.loc[subset, 'residual_std'] = std

        df_list[i] = new_df

    return pd.concat(df_list)


def neighbor_mean_std_v2(df,
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
