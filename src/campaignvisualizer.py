import subprocess
import os

import baybe.surrogates
from matplotlib import pyplot as plt

import pandas as pd
import torch
import numpy as np
import baybe
from tqdm import tqdm

import numbers

import logging

def plot_1D_campaign_history(
    campaign: baybe.Campaign,
    x_name: str,
    output_folder: str,
    fixed_x_values: dict[str, float] = {},
    y_name: str | None = None,
    range_granularity: int = 60,
    x_label: str | None = None,
    y_label: str | None = None,
    show_legend: bool = False,
    ) -> list[plt.Axes]:
    fixed_params = _fix_other_params(campaign, [x_name], fixed_x_values=fixed_x_values)
    x_range = torch.linspace(campaign.measurements[x_name].min(), campaign.measurements[x_name].max(), range_granularity)

    # Create a DataFrame for the surrogate model predictions.
    x_for_prediction = pd.DataFrame({x_name: x_range.numpy()})
    for key, value in fixed_params.items():
        x_for_prediction[key] = value

    model = campaign.get_surrogate()

    # Create the output folder if it doesn't exist.
    os.makedirs(output_folder, exist_ok=True)

    size_norm = _get_euclidean_scatter_size(campaign, x_name, x_for_prediction)

    # Iteratively update the surrogate model and plot its predictions.
    axes = []
    for i in tqdm(range(1, campaign.measurements.shape[0]), colour='green'):
        this_ax = plot_1D_campaign_cross_section(
            campaign,
            x_name,
            i,
            output_folder=output_folder,
            show_legend=show_legend,
            x_label=x_label,
            y_label=y_label,
            y_name=y_name,
            scatter_size=size_norm,
            model=model,
            x_for_prediction=x_for_prediction,
            )
        axes.append(this_ax)

    return axes

def plot_1D_campaign_cross_section(
    campaign: baybe.Campaign,
    x_name: str,
    measurement_step: int,
    fixed_x_values: dict[str, float] = {},
    range_granularity: int = 60,
    output_folder: str | None = None,
    y_name: str | None = None,
    y_axis_min: float | None = None,
    y_axis_max: float | None = None,
    show_legend: bool = False,
    x_label: str | None = None,
    y_label: str | None = None,
    scatter_size: np.ndarray | float | None = None,
    change_scatter_transparency: bool = False,
    model: baybe.surrogates.base.SurrogateProtocol | None = None,
    x_for_prediction: pd.DataFrame | None = None,
    show_plot: bool = True,
    ) -> plt.Axes:
    #TODO: Color and size for two dimensions within a range.
    """
    Plot a 1D cross-section of the parameter hyperspace with the surrogate model and measurements.

    Args:
        campaign: The `BayBE` campaign object
        x_name: The name of the parameter to plot on the x-axis
        measurement_step: The amount of included measurements in the plot and model.
        fixed_x_values: A dictionary with fixed values for the other parameters.
        range_granularity: The amount of points to use for the x-axis range.
        output_folder: The folder to save the plot images to.
            If `None`, the plot is shown instead of saved.
        y_name: The name of the target to plot on the y-axis.
        y_axis_min: The minimum value for the y-axis.
        y_axis_max: The maximum value for the y-axis.
        show_legend: Whether to show the legend on the plot.
        x_label: The label for the x-axis.
        y_label: The label for the y-axis.
        scatter_size: The size of the scatter points. If a float, the size is constant.
        change_scatter_transparency: Whether to change the transparency of the scatter points.
        model: The surrogate model to use for the plot. If `None`, the campaign's surrogate is used.
        x_for_prediction: The DataFrame with the x-axis values to predict. If `None`, it is generated.

    Returns:
        The `axes` object of the plot
    """

    x_label = x_label or x_name
    y_name = y_name or campaign.targets[0].name
    y_label = y_label or y_name
    model = model or campaign.get_surrogate()
    y_name = y_name or campaign.targets[0].name

    # Generate a range for the x-axis.
    x_min, x_max = campaign.measurements[x_name].min(), campaign.measurements[x_name].max()
    x_range = torch.linspace(x_min, x_max, range_granularity)

    fixed_params = _fix_other_params(campaign, [x_name], fixed_x_values=fixed_x_values)
    if x_for_prediction is None:
        x_for_prediction = pd.DataFrame({x_name: x_range.numpy()})
        for key, value in fixed_params.items():
            x_for_prediction[key] = value

    if scatter_size is None:
        scatter_size = _get_euclidean_scatter_size(campaign, x_name, x_for_prediction)
    elif isinstance(scatter_size, numbers.Number):
        scatter_size = np.full(measurement_step, scatter_size)

    # Determine limits for the y-axis from the measurements.
    y_axis_min = y_axis_min or campaign.measurements[y_name].min()
    y_axis_max = y_axis_max or campaign.measurements[y_name].max()

    model = campaign.get_surrogate()
    # Fit the surrogate model to the measurements up to the current iteration.
    model.fit(campaign.searchspace, campaign.objective, campaign.measurements.iloc[:measurement_step])
    posterior = model.posterior(x_for_prediction)
    
    # Extract mean and standard deviation from the posterior.
    mean = posterior.mean.detach().numpy()
    std_dev = posterior.variance.sqrt().detach().numpy()

    fig, ax = plt.subplots()

    # Plot the surrogate model mean as a line.
    ax.plot(x_range.numpy(), mean, color='darkblue', label='Surrogate Model Mean')

    # Plot the uncertainty as a shaded area.
    ax.fill_between(
        x_range.numpy(),
        (mean - std_dev).flatten(),
        (mean + std_dev).flatten(),
        color='lightblue',
        edgecolor='blue',
        alpha=0.3,
        label='Surrogate Model Std. Dev.',
    )

    # Scatter the measurements taken so far.
    ax.scatter(
        campaign.measurements.iloc[:measurement_step][x_name],
        campaign.measurements.iloc[:measurement_step][y_name],
        c='black',
        s=scatter_size[:measurement_step]*10,
        alpha=scatter_size[:measurement_step] if change_scatter_transparency else 0.7,
        label='Measurement'
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_axis_min, y_axis_max)

    if show_legend:
        ax.legend(facecolor='lightgrey', edgecolor='black', loc='upper left')

    # Save the figure.
    if output_folder:
        plt.savefig(os.path.join(output_folder, f'plot_{measurement_step:03}.png'))
        plt.close()
    else:
        if show_plot:
            plt.show()
    return ax

def plot_2D_campaign_cross_section(
        campaign: baybe.Campaign,
        x1_name: str,
        x2_name: str,
        y_name: str | None = None,
        measurement_step: int | None = None,
        grid_granularity: int = 60,
        x1_label: str | None = None,
        x2_label: str | None = None,
        y_label: str | None = None,
        show_legend: bool = False,
        fixed_x_values: dict[str, float] = {},
        scatter: bool = True,
        scatter_size: np.ndarray | float | None = None,
        cmap: str = 'plasma',
        ax: plt.Axes | None = None,
    )  -> plt.Axes:
    x1_label = x1_label or x1_name
    x2_label = x2_label or x2_name
    y_label = y_label or y_name
    y_name = y_name or campaign.targets[0].name
    measurement_step = measurement_step or campaign.measurements.shape[0]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    fixed_params = _fix_other_params(campaign, [x1_name, x2_name], fixed_x_values=fixed_x_values)

    x1_grid = torch.linspace(campaign.measurements[x1_name].min(), campaign.measurements[x1_name].max(), grid_granularity)
    x2_grid = torch.linspace(campaign.measurements[x2_name].min(), campaign.measurements[x2_name].max(), grid_granularity)
    X1_GRID, X2_GRID = torch.meshgrid(x1_grid, x2_grid, indexing='ij')

    df = pd.DataFrame({
        x1_name: X1_GRID.flatten(),
        x2_name: X2_GRID.flatten()
        })

    for key, value in fixed_params.items():
        df[key] = value

    model = campaign.get_surrogate()
    
    if scatter_size is None:
        euclidean_distance = np.linalg.norm(
        (campaign.measurements[[
            p.name for p in campaign.parameters if p.name not in [x1_name, x2_name]
            ]] - df.loc[0]).dropna(axis=1).values,
        axis=1,
        )
        scatter_size = 10 - (euclidean_distance - euclidean_distance.min()) / (euclidean_distance.max() - euclidean_distance.min()) * 10
    elif isinstance(scatter_size, numbers.Number):
        scatter_size = [scatter_size] * campaign.measurements.shape[0] 

    model.fit(campaign.searchspace, campaign.objective, campaign.measurements.iloc[:measurement_step])

    posterior = model.posterior(df)
    
    mean = posterior.mean.reshape(grid_granularity, grid_granularity)
    std_dev = posterior.variance.sqrt().reshape(grid_granularity, grid_granularity)

    # Plot the surrogate model mean as a filled contour plot.
    contour = ax.contourf(
        X1_GRID.numpy(),
        X2_GRID.numpy(),
        mean.detach().numpy(),
        levels=1000,
        cmap=cmap,
    )


    cbar = fig.colorbar(
        contour,
        label=y_label,
        )
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(_sci_text_formatter)) #TODO: Implement scientific notation for colorbar so that it matches the tick positions better.

    # Plot the standard deviation as contour lines.
    uncert = ax.contour(
        X1_GRID.numpy(),
        X2_GRID.numpy(),
        std_dev.detach().numpy(),
        levels=6,
        colors="#33d1f5",
        linewidths=1.5,
        linestyles='dashed',
    )

    # Use scientific notation for the contour labels based on lambda function with LaTeX notation.
    ax.clabel(uncert, inline=True, fontsize=10, 
            fmt = _sci_text_formatter
        )

    if scatter:
        ax.scatter(
            campaign.measurements.iloc[:measurement_step][x1_name],
            campaign.measurements.iloc[:measurement_step][x2_name],
            c='black',
            alpha=0.7,
            s=scatter_size[:measurement_step],
        )

    ax.set_xlabel(x1_label)
    ax.set_ylabel(x2_label)

    if show_legend:
        legend_elements = [
            plt.Line2D([0], [0], color='white', lw=2, label='Surrogate Model Std. Dev.'),
            plt.Line2D([0], [0], color='black', lw=4, label='Surrogate Model Mean'),
            plt.Line2D([0], [0], marker='o', color='r', linestyle='None', markersize=6, label='Measurement')
        ]
        ax.legend(handles=legend_elements, facecolor='lightgrey', edgecolor='black', loc='upper left')

    return ax


def plot_2D_campaign_history(
        campaign: baybe.Campaign,
        x1_name: str,
        x2_name: str,
        output_folder: str,
        filename_prefix: str = 'plot',
        y_name: str | None = None,
        grid_granularity: int = 60,
        x1_label: str | None = None,
        x2_label: str | None = None,
        y_label: str | None = None,
        show_legend: bool = False,
        fixed_x_values: dict[str, float] = {},
        scatter_size: np.ndarray | float | None = None,
        cmap: str = 'plasma',
    ):

    common_kwargs = dict(
        campaign=campaign,
        x1_name=x1_name,
        x2_name=x2_name,
        y_name=y_name,
        grid_granularity=grid_granularity,
        x1_label=x1_label,
        x2_label=x2_label,
        y_label=y_label,
        show_legend=show_legend,
        fixed_x_values=fixed_x_values,
        scatter_size=scatter_size,
        cmap=cmap,
    )

    if common_kwargs['scatter_size'] is None:
        euclidean_distance = np.linalg.norm(
            (campaign.measurements[[
                p.name for p in campaign.parameters
                if p.name not in [x1_name, x2_name]
            ]] - pd.DataFrame({_k: _v for _k, _v in fixed_x_values.items()}, index=[0])).dropna(axis=1).values,
            axis=1,
        )
        common_kwargs['scatter_size'] = 10 - (
            (euclidean_distance - euclidean_distance.min()) /
            (euclidean_distance.max() - euclidean_distance.min())
        ) * 10
    elif isinstance(common_kwargs['scatter_size'], numbers.Number):
        common_kwargs['scatter_size'] = [common_kwargs['scatter_size']] * campaign.measurements.shape[0]

    for i in tqdm(range(1, campaign.measurements.shape[0]), colour='green'):
        ax = plot_2D_campaign_cross_section(
            measurement_step=i,
            **common_kwargs
        )
        fig = ax.get_figure()
        fig.savefig(f'{output_folder}/{filename_prefix}_{i:03}.png')
        plt.close(fig)

def _fix_other_params(
        campaign: baybe.Campaign,
        x_names: list[str],
        fixed_x_values: dict[str, float] | None = None,
    ):
    # For all parameters except the x parameter, use the value from the dict fixed_x_values.
    fixed_params = {}
    for param in campaign.parameters:
        if param.name not in x_names:
            fixed_params[param.name] = fixed_x_values.get(param.name, campaign.measurements[param.name].mean())

    logging.info(f'Fixed parameters: {fixed_params}')

    return fixed_params

def _get_euclidean_scatter_size(
        campaign: baybe.Campaign,
        x_name: str,
        df: pd.DataFrame,
    ):
    """
    Helper function to get the size of the scatter points based on the Euclidean distance of the other parameters.

    Args:
        campaign: The campaign object.
        x_name: The name of the parameter to plot.
        df: The DataFrame with the other parameters.
    
    Returns:
        The size of the scatter points based on the Euclidean distance. The size is normalized to be between 0 and 1.
    """
    euclidean_distance = np.linalg.norm(
        (campaign.measurements[
            [p.name for p in campaign.parameters if p.name != x_name]
            ] - df.loc[0]).dropna(axis=1).values,
        axis=1,
        )
    
    size_norm = 1 - (euclidean_distance - euclidean_distance.min()) / (euclidean_distance.max() - euclidean_distance.min())

    return size_norm

def _sci_text_formatter(
        x: float,
        pos = None,
    ) -> str:
    """
    Format a number in scientific notation using LaTeX notation with cdots and 10^ exponents.

    Args:
        x: The number to format.
        pos: Unused argument for compatibility with the matplotlib formatter.

    Returns:
        The formatted number in LaTeX notation.
    """
    # Format the number in scientific notation using Python's formatting.
    if x == 0:
        return '0'
    else:
        s = "{:.1e}".format(x)
        # Split the result into the base and exponent parts.
        base, exp = s.split('e')
        # Convert the exponent to an integer (this removes any unnecessary zeros)
        exp = int(exp)
        # Return the formatted string using LaTeX notation.
        return r'${} \cdot 10^{{{}}}$'.format(base, exp)

def video_from_folder(
        folder: str,
        output_filename: str,
        frame_rate: float = 5,
        file_format_scheme: str = 'plot_with_legend_%03d.png',
    ):
    ffmpeg_cmd = [
        'ffmpeg',
        '-framerate', str(frame_rate),
        '-i', os.path.join(os.path.abspath(folder), file_format_scheme),
        '-c:v', 'libx264',
        '-profile:v', 'high',
        '-level', '3.0',
        '-r', '30',
        '-pix_fmt', 'yuv420p',
        '-an',
        '-y', os.path.join(os.path.abspath(folder), output_filename),
        '-preset', 'veryslow'
    ]

    subprocess.run(ffmpeg_cmd)