import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


def plot_residuals(residual_array, group_name, absolute=False):
    """
    Plot the residuals from a group (or all groups).

    Args:
        residual_array: (np.ndarray) 2 dimensional array with
            residual_array[:,0] column for how far predicting out
            residual_array[:,1] column for number of data points
            residual_array[:,2] residual observation
        group_name: (str) name for labeling the plot
        absolute: (bool) plot absolute value of the residuals
    """
    mat = np.copy(residual_array)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Normalize the color scale
    if absolute:
        scatter = ax.scatter(x=mat[:, 0], y=mat[:, 1], c=np.abs(mat[:, 2]), cmap='Reds')
    else:
        max_abs_res = max(abs(residual_array[:, 2]))
        norm = colors.Normalize(vmin=-max_abs_res, vmax=max_abs_res)
        scatter = ax.scatter(x=mat[:, 0], y=mat[:, 1], c=mat[:, 2], cmap='PRGn', norm=norm)
    plt.colorbar(scatter, format='%.0e')

    ax.set_xlabel('Number Predicting Out')
    ax.set_ylabel('Number of Data Points')
    ax.set_title(f"{group_name} residuals")
