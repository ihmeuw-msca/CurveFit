import numpy as np
import pandas as pd


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
