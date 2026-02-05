import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Literal
import matplotlib as mpl

import shap

from matplotlib.colors import LinearSegmentedColormap

# --- Font consistency across plots ---
LABEL_FONT_SIZE = 10 #11
TICK_FONT_SIZE = 10
TITLE_FONT_SIZE = 12

HEATMAP_CBAR_KWARGS = {
    'orientation': 'vertical',
    'location': 'right',
    'pad': 0.02,
}

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import colorsys

def make_monochrome_cmap(hex_color, n=256):
    """
    Creates a gradient from Light -> Color -> Dark Version of Color.
    Does NOT go to pure black.
    """
    # 1. Get the RGB of your base color
    if isinstance(hex_color, str):
        base_rgb = mcolors.hex2color(hex_color)
    else:
        base_rgb = hex_color[:3]
        
    # 2. Convert to HLS (Hue, Lightness, Saturation) to modify lightness safely
    h, l, s = colorsys.rgb_to_hls(*base_rgb)
    
    # 3. Create the Anchors
    # Anchor 1: Very Light Version (Almost white, but tinted)
    # We increase lightness to 0.8, keep hue the same
    light_rgb = colorsys.hls_to_rgb(h, max(0.8, l), s)
    
    # Anchor 3: Dark Version (Your "Dark Brown")
    # We reduce lightness to 50% of the original (0.5 * l), but keep the saturation
    dark_rgb = colorsys.hls_to_rgb(h, l * 0.5, s)

    # 4. Define the gradient points: Light -> Base -> Dark
    colors = [light_rgb, base_rgb, dark_rgb]
    
    return mcolors.LinearSegmentedColormap.from_list("custom_mono", colors, N=n)


def plot_clean_correlation_matrix(
        df: pd.DataFrame,
        correlation_type: Literal['pearson', 'spearman'] = 'spearman',
        ax: plt.Axes = None,
        aliases: dict = {},
        cbar_kwargs: dict = HEATMAP_CBAR_KWARGS,
        show_cbar: bool = True,
    ) -> plt.Axes:

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    fig = ax.get_figure()

    
    correlations = df.corr(method=correlation_type, numeric_only=True)
    mask = np.triu(np.ones_like(correlations, dtype=bool))

    if aliases is not None:
        mapping = {k: v for k, v in aliases.items() if k in correlations.index}
        correlations = correlations.rename(index=mapping, columns=mapping)

    # --- Heatmap setup ---
    annot_df = correlations.round(2).applymap(
        lambda v: "" if pd.isna(v) else f"{v:.2f}".replace("0.", ".")
    )

    sns.heatmap(
        correlations,
        mask=mask,
        cmap='RdBu',
        vmin=-1,
        vmax=1,
        annot=annot_df,
        fmt='',
        alpha=0.8,
        xticklabels=correlations.columns,
        yticklabels=correlations.index,
        cbar=False,  # Disable default colorbar
        ax=ax,
    )

    if show_cbar:
        # Add heatmap colorbar via use_gridspec=True to avoid overlap issues
        im = ax.collections[0]  # Grab the heatmap artist
        heatmap_cbar = fig.colorbar(im, ax=ax, use_gridspec=True, **cbar_kwargs)
        heatmap_cbar.outline.set_visible(False)

    # Crop last row/col visually
    ax.set_xlim(None, ax.get_xlim()[1] - 1)
    ax.set_ylim(None, ax.get_ylim()[1] + 1)

def apply_standard_font_sizes(
        axs: dict[str, plt.Axes],
        label_fontsize: int = LABEL_FONT_SIZE,
        tick_fontsize: int = TICK_FONT_SIZE,
        title_fontsize: int = TITLE_FONT_SIZE,
    ) -> None:

    for key, ax in axs.items():
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=tick_fontsize)
        plt.setp(ax.get_yticklabels(), fontsize=tick_fontsize)
        ax.set_xlabel(ax.get_xlabel(), fontsize=label_fontsize)
        ax.set_ylabel(ax.get_ylabel(), fontsize=label_fontsize)
        ax.set_title(ax.get_title(), fontsize=title_fontsize)

def truncate_colormap(
        cmap: Literal[str] | LinearSegmentedColormap,
        minval: float = 0.0,
        maxval: float = 1.0,
        n: int = 1024,
    ) -> LinearSegmentedColormap:
    """
    Return a new colormap which is cmap[minval:maxval]

    Args:
        cmap: Original colormap or its name.
        minval: Minimum value (between 0 and 1).
        maxval: Maximum value (between 0 and 1).
        n: Number of colors in the new colormap.

    Returns:
        Truncated colormap.
    """
    orig = plt.cm.get_cmap(cmap) if isinstance(cmap, str) else cmap
    colors = orig(np.linspace(minval, maxval, n))
    return LinearSegmentedColormap.from_list(
        f"{orig.name}_trunc_{minval:.2f}_{maxval:.2f}", colors)

def plot_shap_beeswarm_with_colorbar(
    shap_explanation: shap.Explanation,
    aliases: dict[str, str],
    heatmap_cbar_kwargs: dict,
    title: str,
    xlabel: str,
    cmap: str = "plasma",
    alpha: float = 0.5,
    ax: plt.Axes | None = None,
):
    """
    Plot a SHAP beeswarm with:
      - aliased feature names
      - removal of SHAP's default colorbar
      - custom Seaborn-like colorbar labeled 'Low' → 'High'
    on the given axis.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    fig = ax.get_figure()

    # --- Prepare explanation display with aliased feature names ---
    explanation_display = shap_explanation

    orig_names = getattr(explanation_display, "feature_names", None)
    if orig_names is not None:
        display_names = [aliases.get(n, n) for n in orig_names]
        explanation_display.feature_names = display_names

    # --- SHAP beeswarm (track axes to kill SHAP’s auto colorbar) ---
    axes_before = list(fig.axes)

    shap.plots.beeswarm(
        explanation_display,
        ax=ax,
        plot_size=None,
        show=False,
        color=cmap,
        alpha=alpha,
    )

    # --- Remove SHAP’s default colorbar axis (new axis added by beeswarm) ---
    axes_after = list(fig.axes)
    new_axes = [a for a in axes_after if a not in axes_before]
    for extra_ax in new_axes:
        fig.delaxes(extra_ax)

    # --- Custom colorbar using full feature value range ---
    vmin = explanation_display.values.min()
    vmax = explanation_display.values.max()

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, ax=ax, use_gridspec=True, **heatmap_cbar_kwargs)
    cbar.set_label("Feature value", fontsize=11, labelpad=2)
    cbar.ax.tick_params(labelsize=10, width=0.8, length=3)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(color="black", labelcolor="black")
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels(["Low", "High"])

    # --- Axis cosmetics ---
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.xaxis.set_label_coords(None, -0.25)
    ax.xaxis.set_major_formatter(sci_text_format)

    return ax