import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker
from pathlib import Path
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr

from src.shap_utils import names


def create_lime_explainer(
        X_train: pd.DataFrame,
        feature_names: list[str],
        mode: str = 'regression',
        kernel_width: float | None = None,
    ) -> LimeTabularExplainer:
    """
    Create a LIME tabular explainer for regression.

    Args:
        X_train: Training data used as background for perturbations.
        feature_names: List of feature column names.
        mode: 'regression' or 'classification'.
        kernel_width: Width of the exponential kernel. None uses LIME default.

    Returns:
        A configured LimeTabularExplainer.
    """
    kwargs = {}
    if kernel_width is not None:
        kwargs['kernel_width'] = kernel_width

    explainer = LimeTabularExplainer(
        training_data=X_train.values if isinstance(X_train, pd.DataFrame) else X_train,
        feature_names=feature_names,
        mode=mode,
        discretize_continuous=False,
        **kwargs,
    )
    return explainer


def explain_all_samples(
        explainer: LimeTabularExplainer,
        predict_fn,
        X: pd.DataFrame,
        num_features: int = 6,
    ) -> tuple[pd.DataFrame, list]:
    """
    Generate LIME explanations for all samples in X.

    Args:
        explainer: A LimeTabularExplainer instance.
        predict_fn: Model prediction function (e.g., model.predict).
        X: DataFrame of samples to explain.
        num_features: Number of top features to include in each explanation.

    Returns:
        Tuple of (df_LIME, explanations):
            - df_LIME: DataFrame of LIME coefficients (n_samples x n_features).
            - explanations: List of raw LIME Explanation objects.
    """
    feature_names = explainer.feature_names
    explanations = []
    coefficients = []

    for i in range(len(X)):
        instance = X.iloc[i].values if isinstance(X, pd.DataFrame) else X[i]
        exp = explainer.explain_instance(
            instance,
            predict_fn,
            num_features=num_features,
        )
        explanations.append(exp)

        # Extract coefficients into a dict keyed by feature name
        coeff_dict = {name: 0.0 for name in feature_names}
        for feat_expr, coeff in exp.local_exp[1] if 1 in exp.local_exp else exp.local_exp[list(exp.local_exp.keys())[0]]:
            coeff_dict[feature_names[feat_expr]] = coeff
        coefficients.append(coeff_dict)

    df_LIME = pd.DataFrame(coefficients, columns=feature_names)
    return df_LIME, explanations


def explain_single_sample(
        explainer: LimeTabularExplainer,
        predict_fn,
        instance: np.ndarray,
        num_features: int = 6,
    ):
    """
    Generate a LIME explanation for a single instance.

    Args:
        explainer: A LimeTabularExplainer instance.
        predict_fn: Model prediction function.
        instance: 1D array of feature values.
        num_features: Number of top features to include.

    Returns:
        A LIME Explanation object.
    """
    return explainer.explain_instance(
        instance,
        predict_fn,
        num_features=num_features,
    )


# --- Serialization ---

def save_lime_explanations(
        df_lime: pd.DataFrame,
        explanations: list,
        filepath: Path,
        extra: dict | None = None,
    ) -> None:
    """
    Pickle LIME explanations to disk.

    Args:
        df_lime: DataFrame of LIME coefficients.
        explanations: List of raw LIME Explanation objects.
        filepath: Path to save the pickle file.
        extra: Optional dict of additional data to save (e.g., model, splits, metrics).
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    payload = {'df_lime': df_lime, 'explanations': explanations}
    if extra is not None:
        payload.update(extra)
    with open(filepath, 'wb') as f:
        pickle.dump(payload, f)


def load_lime_explanations(
        filepath: Path,
    ) -> dict:
    """
    Load pickled LIME explanations.

    Args:
        filepath: Path to the pickle file.

    Returns:
        Dict with at least 'df_lime' and 'explanations' keys,
        plus any extra data that was saved (model, X_train, metrics, etc.).
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


# --- Visualization ---

def lime_dependence_plot(
        df_raw: pd.DataFrame,
        df_LIME: pd.DataFrame,
        feature: str,
        feature_comp: str,
        ax: plt.Axes,
        ylim: tuple[float, float] | None = None,
        show_y_labels: bool = True,
        fontsize: int = 8,
        color: str = 'magma',
        show_colorbar: bool = False,
        title: str | None = None,
        s: int = 40,
    ) -> plt.Axes:
    """
    LIME dependence plot analogous to SHAP dependence_plot_single.

    X-axis: raw feature value, Y-axis: LIME coefficient, color: companion feature value.

    Args:
        df_raw: DataFrame of raw feature values.
        df_LIME: DataFrame of LIME coefficients.
        feature: Feature to plot on x-axis / y-axis coefficient.
        feature_comp: Feature to use for color encoding.
        ax: Matplotlib Axes to plot on.
        ylim: Optional y-axis limits.
        show_y_labels: Whether to show y-axis label.
        fontsize: Font size for labels and ticks.
        color: Colormap name.
        show_colorbar: Whether to show a colorbar.
        title: Optional plot title.
        s: Marker size.

    Returns:
        Matplotlib Axes with the plot.
    """
    sc = ax.scatter(
        df_raw[feature], df_LIME[feature],
        c=df_raw[feature_comp], cmap=color, s=s,
        edgecolor='white', linewidth=0.5,
    )

    ax.set_xlabel(names[feature], fontsize=fontsize)
    if ylim is None:
        ylim = [ax.get_ylim()[0], ax.get_ylim()[1]]
    ax.set_ylim(ylim)

    if not show_y_labels:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel(r'LIME Coefficient ($\mathrm{\AA\,s^{-1}}$)', fontsize=fontsize)

    ax.minorticks_on()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(3))
    ax.tick_params(direction='in', top=True, right=True)
    ax.tick_params(which='minor', direction='in', top=True, right=True)
    ax.tick_params(which='both', labelsize=fontsize - 2)

    if show_colorbar:
        cbar = plt.colorbar(sc)
        cbar.set_label(names[feature_comp], rotation=270, labelpad=15, fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize - 2)

    if title is not None:
        ax.set_title(title, fontsize=fontsize + 2)

    # Dotted zero line
    x = np.linspace(-10, 1.2 * np.max(df_raw[feature]), 1000)
    mean = [0] * 1000
    sns.lineplot(x=x, y=mean, ax=ax, linestyle=':', color='black', linewidth=1.5, zorder=0)
    ax.set_xlim([df_raw[feature].min() - 5, df_raw[feature].max() + 5])

    # Histogram overlay for data distribution
    num_bins = 5
    bins = np.linspace(df_raw[feature].min(), df_raw[feature].max(), num_bins + 1)
    counts, _ = np.histogram(df_raw[feature], bins=bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    counts = counts * (ylim[1] * 0.5 / counts.max())
    bar_width = (bins[1] - bins[0])
    ymin = ylim[0]
    ax.bar(bin_centers, counts, width=bar_width, color='grey', alpha=0.2, bottom=ymin)

    return ax


def lime_importance_barplot(
        df_LIME: pd.DataFrame,
        ax: plt.Axes,
        feature_names: list[str] | None = None,
        aliases: dict[str, str] | None = None,
        fontsize: int = 10,
        color: str = '#1f77b4',
    ) -> plt.Axes:
    """
    Horizontal bar chart of mean absolute LIME coefficients per feature.

    Args:
        df_LIME: DataFrame of LIME coefficients.
        ax: Matplotlib Axes.
        feature_names: Optional list of feature names to include.
        aliases: Optional dict mapping raw feature names to display names.
        fontsize: Font size.
        color: Bar color.

    Returns:
        Matplotlib Axes with the bar plot.
    """
    if aliases is None:
        aliases = names

    if feature_names is None:
        feature_names = list(df_LIME.columns)

    mean_abs = df_LIME[feature_names].abs().mean().sort_values(ascending=True)
    display_labels = [aliases.get(f, f) for f in mean_abs.index]

    ax.barh(display_labels, mean_abs.values, color=color, edgecolor='white')
    ax.set_xlabel(r'Mean |LIME Coefficient| ($\mathrm{\AA\,s^{-1}}$)', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize - 1)
    ax.minorticks_on()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(3))
    ax.tick_params(direction='in', top=True, right=True)
    ax.tick_params(which='minor', direction='in', top=True, right=True)

    return ax


def lime_waterfall_plot(
        explanation,
        ax: plt.Axes,
        feature_names: list[str],
        aliases: dict[str, str] | None = None,
        fontsize: int = 10,
    ) -> plt.Axes:
    """
    Bar chart showing LIME coefficients for a single instance.

    Args:
        explanation: A single LIME Explanation object.
        ax: Matplotlib Axes.
        feature_names: List of feature names.
        aliases: Optional dict mapping raw feature names to display names.
        fontsize: Font size.

    Returns:
        Matplotlib Axes with the waterfall plot.
    """
    if aliases is None:
        aliases = names

    # Extract coefficients from the explanation
    exp_key = list(explanation.local_exp.keys())[0]
    coeff_dict = {}
    for feat_idx, coeff in explanation.local_exp[exp_key]:
        coeff_dict[feature_names[feat_idx]] = coeff

    # Sort by absolute value
    sorted_features = sorted(coeff_dict.keys(), key=lambda f: abs(coeff_dict[f]))
    display_labels = [aliases.get(f, f) for f in sorted_features]
    values = [coeff_dict[f] for f in sorted_features]

    colors = ['#d62728' if v < 0 else '#1f77b4' for v in values]
    ax.barh(display_labels, values, color=colors, edgecolor='white')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel(r'LIME Coefficient ($\mathrm{\AA\,s^{-1}}$)', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize - 1)

    # Annotate intercept
    intercept = explanation.intercept[1] if isinstance(explanation.intercept, dict) else explanation.intercept
    ax.set_title(f'Intercept: {intercept:.3f}', fontsize=fontsize)

    return ax


def compare_shap_lime_importance(
        df_SHAP: pd.DataFrame,
        df_LIME: pd.DataFrame,
        feature_names: list[str],
        ax: plt.Axes,
        aliases: dict[str, str] | None = None,
        fontsize: int = 10,
    ) -> plt.Axes:
    """
    Side-by-side bar chart comparing mean |SHAP| vs mean |LIME| feature importance.
    Annotates Spearman rank correlation.

    Args:
        df_SHAP: DataFrame of SHAP values.
        df_LIME: DataFrame of LIME coefficients.
        feature_names: List of feature names to compare.
        ax: Matplotlib Axes.
        aliases: Optional dict mapping raw feature names to display names.
        fontsize: Font size.

    Returns:
        Matplotlib Axes with the comparison plot.
    """
    if aliases is None:
        aliases = names

    shap_importance = df_SHAP[feature_names].abs().mean()
    lime_importance = df_LIME[feature_names].abs().mean()

    # Normalize to [0, 1] for comparison
    shap_norm = shap_importance / shap_importance.max()
    lime_norm = lime_importance / lime_importance.max()

    # Sort by SHAP importance
    order = shap_norm.sort_values(ascending=True).index
    display_labels = [aliases.get(f, f) for f in order]

    y = np.arange(len(order))
    bar_height = 0.35

    ax.barh(y - bar_height / 2, shap_norm[order].values, bar_height,
            label='SHAP', color='#1f77b4', edgecolor='white')
    ax.barh(y + bar_height / 2, lime_norm[order].values, bar_height,
            label='LIME', color='#ff7f0e', edgecolor='white')

    ax.set_yticks(y)
    ax.set_yticklabels(display_labels)
    ax.set_xlabel('Normalized Feature Importance', fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize - 1)
    ax.legend(fontsize=fontsize - 1, frameon=False)

    # Spearman rank correlation
    rho, p_val = spearmanr(
        shap_importance[feature_names].values,
        lime_importance[feature_names].values,
    )
    ax.annotate(
        f'Spearman $\\rho$ = {rho:.2f} (p = {p_val:.3f})',
        xy=(0.98, 0.02), xycoords='axes fraction',
        ha='right', va='bottom', fontsize=fontsize - 1,
    )

    return ax
