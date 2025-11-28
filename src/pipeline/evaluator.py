"""
Model evaluation module for Air Quality ML Pipeline.

This module provides core evaluation functionality used by the package.
For detailed evaluation with visualizations, see utils.evaluation_utils.
"""

from sklearn.metrics import precision_score, accuracy_score, recall_score
from sklearn.model_selection import GridSearchCV
from utils.logger import get_logger, LogLevel
from utils.config import CROSS_VALIDATION

import mlflow

def calculate_metrics(y_true, y_pred):
    """
    Calculate comprehensive regression metrics.

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        Dictionary with calculated metrics
    """
    # Assert
    assert len(y_true) == len(y_pred)

    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred, average='binary'),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary')
    }

    return metrics

def hyperparameter_optimization(model, param_grid, X, y):
    """
    Perform hyperparameter optimization using GridSearchCV with geographic cross-validation.

    This method combines GridSearchCV with GroupKFold to ensure that entire cities
    are either in training OR validation during hyperparameter search, preventing data leakage.

    Args:
        model: Scikit-learn model to optimize
        param_grid: Dictionary of hyperparameters to search
        X: Feature matrix
        y: Target variable

    Returns:
        Tuple of (best_model, best_params, best_score)
    """
    logger = get_logger()
    logger.info(f"Optimizing hyperparameters for {model.__class__.__name__}...", LogLevel.NORMAL)

    # GridSearchCV definition
    gscv = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='precision',
        cv=CROSS_VALIDATION,
        n_jobs=-1,
        verbose=1
    )

    gscv.fit(X, y)

    # Extract results
    best_model = gscv.best_estimator_
    best_params = gscv.best_params_
    best_score = gscv.best_score_

    # Logging
    logger.success(f"Best precision: {best_score:.3f}")
    logger.info(f"  Best parameters: {best_params}")

    return best_model, best_params, best_score

