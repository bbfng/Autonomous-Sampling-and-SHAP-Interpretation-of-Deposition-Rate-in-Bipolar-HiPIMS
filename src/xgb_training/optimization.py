from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

def create_objective(
        X_train: np.ndarray | pd.DataFrame,
        y_train_scaled: np.ndarray | pd.Series,
        c_val_splits: int = 5,
        early_stopping_rounds: int = 50,
    ):
    """
    Creates an objective function for hyperparameter optimization as used by Optuna.

    Args:
        X_train: Training features.
        y_train_scaled: Scaled training targets.
        c_val_splits: Number of splits for cross-validation.
        early_stopping_rounds: Number of rounds for early stopping in XGBoost.

    Returns:
        A function that takes a trial object and returns the mean squared error for the model.
    """

    def objective(trial):
        """
        Objective function for hyperparameter optimization.

        Args:
            trial: An Optuna trial object.

        Returns:
            The mean squared error of the model on the validation set.
        """

        params = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 1000),
            'max_depth': trial.suggest_int('max_depth', 1, 20),
            'subsample': trial.suggest_float("subsample", 0.5, 1.0),
            'learning_rate': trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            #'reg_alpha': trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True), # Regularization did not improve results and is hence commented out
            #'reg_lambda': trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            #'device': 'gpu', # Use GPU acceleration if available
        }

        model = XGBRegressor(**params, early_stopping_rounds=early_stopping_rounds)

        kf = KFold(n_splits=c_val_splits, shuffle=True, random_state=42)
        
        scores = []
        for train_idx, val_idx in kf.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train_scaled.iloc[train_idx], y_train_scaled.iloc[val_idx]

            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )

            y_pred = model.predict(X_val)
            score = mean_squared_error(y_val, y_pred)
            scores.append(score)

        return np.mean(scores)
    
    return objective

import optuna

class EarlyStoppingCallback:
    """
    Custom callback for early stopping in Optuna trials.
    """
    def __init__(self, patience: int):
        self.patience = patience
        self.best_value = float("inf")
        self.counter = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        if study.best_value < self.best_value:
            self.best_value = study.best_value
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            print(f"Early stopping triggered. No improvement in {self.patience} trials.")
            study.stop()
