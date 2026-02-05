"""
A collection of utility functions for training and evaluating XGBoost models based on BayBE campaigns.
"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from baybe.campaign import Campaign
from xgboost import XGBRegressor
from numpy.typing import NDArray
from typing import Any

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

def campaign_split_test_train(
        campaign: Campaign,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Splits the campaign measurements into training and testing sets.
    
    Args:
        campaign: The campaign containing the measurements.
        test_size: Proportion (0, 1) of the dataset to include in the test split.
        random_state: Random seed for reproducibility.
        
    Returns:
        A tuple containing the training features, testing features, training targets, and testing targets.
    """
    measurements = campaign.measurements

    X = measurements[[p.name for p in campaign.parameters]]
    y = measurements[campaign.targets[0].name]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    return X_train, X_test, y_train, y_test

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

def train_final_model(
        X_train: NDArray | pd.DataFrame,
        y_train_scaled: NDArray | pd.Series,
        best_params: dict[str, Any],
        val_fraction: float = 0.2,
        verbose: bool = False,
    ) -> XGBRegressor:
    """
    Trains the final XGBoost model using the best hyperparameters found during optimization.
    This includes splitting the training data into training and validation sets.

    Args:
        X_train: Training features.
        y_train_scaled: Scaled training targets.
        best_params: Best hyperparameters found during optimization.
        val_fraction: Fraction of the training data to use for validation.
        verbose: Whether to print training progress.

    Returns:
        An XGBRegressor model trained on the training data.
    """
    # Split data into training and validation sets
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train_scaled, test_size=val_fraction, random_state=42
    )

    model = XGBRegressor(**best_params)

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=verbose
    )

    return model


def evaluate_model(
        model: XGBRegressor,
        X_test: NDArray | pd.DataFrame,
        y_test_scaled: NDArray | pd.Series,
    ) -> tuple[float, float, NDArray]:
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test_scaled, y_pred))
    r2 = r2_score(y_test_scaled, y_pred)
    return rmse, r2, y_pred

def plot_parity(
    y_train: NDArray | pd.Series, 
    y_train_pred: NDArray | pd.Series,
    y_test: NDArray | pd.Series,
    y_test_pred: NDArray | pd.Series,
    scaled: bool = True
    ) -> plt.Axes:
    """
    Plot true vs predicted values with metrics overlay.
    
    Args:
        y_train: True values for the training set.
        y_train_pred: Predicted values for the training set.
        y_test: True values for the test set.
        y_test_pred: Predicted values for the test set.
        scaled: Whether the values are scaled (for labeling).

    Returns:
        A matplotlib Axes object with the parity plot.
    """
    
    # Compute metrics
    final_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    final_r2 = r2_score(y_test, y_test_pred)

    _, ax = plt.subplots()

    ax.scatter(y_train, y_train_pred, alpha=0.4, color='orange', label='Training data')
    ax.scatter(y_test, y_test_pred, alpha=0.6, color='blue', label='Test data')

    min_val = min(y_test.min(), y_test_pred.min())
    max_val = max(y_test.max(), y_test_pred.max())
    
    ax.plot([min_val, max_val], [min_val, max_val], '--', color='gray', lw=2)

    ax.annotate(
        f'$RMSE$: {final_rmse:.2f}\n$R^2$: {final_r2:.2f}',
        xy=(0.02, 0.98), xycoords='axes fraction',
        ha='left', va='top'
    )

    ax.set(
        xlabel=f"True Values{' (scaled)' if scaled else ''}",
        ylabel=f"Predictions{' (scaled)' if scaled else ''}",
        aspect='equal'
    )
    ax.legend()
    plt.tight_layout()
    return ax

def plot_residuals(
    y_train: NDArray | pd.Series,
    y_train_pred: NDArray | pd.Series,
    y_test: NDArray | pd.Series,
    y_test_pred: NDArray | pd.Series,
    scaled: bool = True
    ) -> plt.Axes:
    """
    Plot residuals (prediction - true) vs true values.

    Args:
        y_train: True values for the training set.
        y_train_pred: Predicted values for the training set.
        y_test: True values for the test set.
        y_test_pred: Predicted values for the test set.
        scaled: Whether the values are scaled (for labeling).

    Returns:
        A matplotlib Axes object with the residuals plot.
    """
    
    residuals_train = y_train_pred - y_train
    residuals_test = y_test_pred - y_test

    _, ax = plt.subplots()

    ax.scatter(y_train, residuals_train, alpha=0.4, color='orange', label='Training data')
    ax.scatter(y_test, residuals_test, alpha=0.6, color='blue', label='Test data')

    ax.axhline(0, linestyle='--', color='gray', lw=2)

    ax.set(
        xlabel=f"True Values{' (scaled)' if scaled else ''}",
        ylabel=f'Residuals (Prediction $-$ True)',
    )
    ax.legend()
    plt.tight_layout()
    return ax