import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
from curvefit.core.utils import data_translator


def plot_fits(generator, prediction_times, sharex, sharey, draw_space, plot_obs=None, plot_uncertainty=False):
    """
    Plot the result and draws from a model generator at some prediction times.

    Args:
        generator: (curvefit.model_generator.ModelPipeline) that has some draws
        prediction_times: (np.array) of prediction times
        sharex: (bool) fix the x axes
        sharey: (bool) fix the y axes
        draw_space: (callable) which curvefit.functions space to plot the draws in
        plot_obs: (optional str) column of observations to plot
        plot_uncertainty: (optional bool) plot the uncertainty intervals
    """
    fig, ax = plt.subplots(len(generator.groups), 1, figsize=(8, 4 * len(generator.groups)),
                           sharex=sharex, sharey=sharey)
    if len(generator.groups) == 1:
        ax = [ax]
    for i, group in enumerate(generator.groups):
        draws = generator.draws[group].copy()
        draws = data_translator(
            data=draws,
            input_space=generator.predict_space,
            output_space=draw_space
        )
        mean_fit = generator.mean_predictions[group].copy()
        mean_fit = data_translator(
            data=mean_fit,
            input_space=generator.predict_space,
            output_space=draw_space
        )
        mean = draws.mean(axis=0)
        ax[i].plot(prediction_times, mean, c='red', linestyle=':')
        ax[i].plot(prediction_times, mean_fit, c='black')

        if plot_uncertainty:
            lower = np.quantile(draws, axis=0, q=0.025)
            upper = np.quantile(draws, axis=0, q=0.975)
            ax[i].plot(prediction_times, lower, c='red', linestyle=':')
            ax[i].plot(prediction_times, upper, c='red', linestyle=':')

        if plot_obs is not None:
            df_data = generator.all_data.loc[generator.all_data[generator.col_group] == group].copy()
            ax[i].scatter(df_data[generator.col_t], df_data[plot_obs])

        ax[i].set_title(f"{group} predictions")


def plot_es(pv_group, exp_smoothing, prediction_times, max_last):
    """
    Plot results with different levels of exponential smoothing.

    Args:
        pv_group: (curvefit.pv.pv.PVGroup)
        exp_smoothing: (np.array) exponential smoothing parameter
        prediction_times: (np.array) prediction times
        max_last: (int) number of last times to consider

    Returns:

    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    times = pv_group.times
    observations = pv_group.compare_observations

    for i, exp in enumerate(exp_smoothing):
        p = ax.plot(
            prediction_times,
            pv_group.exp_smooth_preds(
                exp_smoothing=exp,
                prediction_times=prediction_times,
                max_last=max_last
            ),
            label=str(exp),
        )
        leg = ax.legend([p], [str(exp)])
        ax.add_artist(leg)

    ax.scatter(times, observations, zorder=len(exp_smoothing), color='black')
    ax.set_title(f"{pv_group.predict_group}")
    ax.legend()


def plot_residuals_1d(residual_df, group_col, x_axis,
                      y_axis, group=None, color=None):
    """
    Plot a scatter plot of some residual column
    potentially sub setting by group, across some other column.

    Args:
        residual_df: (pd.DataFrame) with columns x_axis and y_axis
        group_col: (str) column name of group col
        x_axis: (str) x axis
        y_axis: (str) y axis
        color: (str) color variable
        group: (str) optional group to subset by
    Returns:

    """

    if group is not None:
        df = residual_df.loc[residual_df[group_col] == group].copy()
    else:
        df = residual_df.copy()
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    if color is not None:
        scatter = ax.scatter(df[x_axis], df[y_axis], c=df[color])
        plt.colorbar(scatter)
    else:
        ax.scatter(df[x_axis], df[y_axis])

    if y_axis in ['residual', 'residual_mean']:
        ax.axhline(y=0, color='black')

    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    title = f"Residuals"
    if group is not None:
        title += f" -- GROUP {group}"
    if color is not None:
        title += f" -- COLOR {color}"
    ax.set_title(title)


def plot_residuals(residual_array, group_name, x_label, y_label, absolute=False, fig=None, axis=None):
    """
    Plot the residuals from a group (or all groups).

    Args:
        residual_array: (np.ndarray) 2 dimensional array with
            residual_array[:,0] column for how far predicting out
            residual_array[:,1] column for number of data points
            residual_array[:,2] residual observation
        group_name: (str) name for labeling the plot
        x_label: (str) the label for x axis
        y_label: (str) the label for y axis
        absolute: (bool) plot absolute value of the residuals
        fig: existing figure from matplotlib.pyplot.subplots to add the plots to
        axis: existing axis from matplotlib.pyplot.subplots to add the plots to
    """
    mat = np.copy(residual_array)

    if axis is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    else:
        ax = axis
        fig = fig

    # Normalize the color scale
    if absolute:
        scatter = ax.scatter(x=mat[:, 0], y=mat[:, 1], c=np.abs(mat[:, 2]), cmap='Reds')
    else:
        max_abs_res = max(abs(residual_array[:, 2]))
        norm = colors.Normalize(vmin=-max_abs_res, vmax=max_abs_res)
        scatter = ax.scatter(x=mat[:, 0], y=mat[:, 1], c=mat[:, 2], cmap='PRGn', norm=norm)
    fig.colorbar(scatter, format='%.0e', ax=ax)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{group_name} residuals")


def plot_predictions(prediction_array, group_name, times, observations):
    fig, ax = plt.subplots(prediction_array.shape[0], 1, figsize=(8, 4 * prediction_array.shape[0]))
    for i in range(prediction_array.shape[0]):
        used_to_fit = times <= times[i]
        not_used = times > times[i]
        ax[i].scatter(times[used_to_fit], observations[used_to_fit], color='green', alpha=0.5)
        ax[i].scatter(times[not_used], observations[not_used], color='red', alpha=0.5)
        ax[i].plot(times, prediction_array[i, :], color='green')
        ax[i].set_title(f"{group_name} predictions based on time {i}")
