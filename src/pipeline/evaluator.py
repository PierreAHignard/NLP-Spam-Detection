"""
Model evaluation module for Air Quality ML Pipeline.

This module provides core evaluation functionality used by the package.
For detailed evaluation with visualizations, see utils.evaluation_utils.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

# Import GridSearchCV for hyperparameter optimization (Workshop 3)

from sklearn.model_selection import GridSearchCV

# Import MLflow (Workshop 4)

import mlflow

from utils.config import N_SPLITS, RANDOM_STATE
from utils.logger import get_logger, LogLevel


class Evaluator:
    """
    Core evaluator for air quality prediction models.
    
    This class handles basic model evaluation including cross-validation
    and metrics calculation used by the package components.
    """
    
    def __init__(self):
        """Initialize the evaluator."""
        pass
    
    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate comprehensive regression metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            
        Returns:
            Dictionary with calculated metrics
        """
        # Calculate comprehensive regression metrics
        rmse = root_mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Create metrics dictionary
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        print(" Y_TRUE : ", y_true)
        print(" Y_PRED : ", y_pred)
        print(metrics)
        return metrics
    
    def cross_validate_model(self, model, X, y, groups=None):
        """
        Perform cross-validation using GroupKFold.
        
        This method uses GroupKFold to ensure entire cities are either in training 
        OR validation, never both, preventing data leakage.
        
        Args:
            model: Scikit-learn model to evaluate
            X: Feature matrix
            y: Target variable
            groups: Grouping variable for GroupKFold (e.g., cities)
            
        Returns:
            Dictionary with cross-validation results
        """
        logger = get_logger()
        logger.info(f"Cross-validating {model.__class__.__name__}...", LogLevel.NORMAL)
        
        # TODO Set up GroupKFold cross-validation
        # If groups provided, use GroupKFold with N_SPLITS and RANDOM_STATE
        # Else if no groups provided, use KFold with N_SPLITS, shuffle=True and RANDOM_STATE

        if groups == None:
            gkf = GroupKFold(n_splits=N_SPLITS, shuffle=True, RANDOM_STATE=1) 
            groups = X["city"]
        else:
            gfk = GroupKFold(n_splits=N_SPLITS, RANDOM_STATE=1)

        for fold, (train_index, test_index) in enumerate(gkf.split(X, y, groups=groups)):
            X.loc[test_index, "fold"] = fold

        fold_results = []
        # TODO Perform cross-validation enumerating folds
        for i in range(N_SPLITS):
            # Split data
            X_i = X[X["fold"] != i]
            val_i = X[X["fold"] == i]
            # Train model
            model_i = model
            # Predict
            predict_i = model.predict(X)
            # Calculate metrics and append to fold_results
            rmse_best = root_mean_squared_error(y, predict_i)
            r2_best = r2_score(y, predict_i)
            # Logging
            print(" RMSE Score for i = ", i, " : ", rmse_best)
            print("r2_best for i = ", i, " : ", r2_best)
        
        # Aggregate results
        cv_results = {}
        # Enumerate metrics and calculate mean/std across folds
        for metric in fold_results[0].keys():
            values = [fold[metric] for fold in fold_results]
            cv_results[f'{metric}_mean'] = np.mean(values)
            cv_results[f'{metric}_std'] = np.std(values)
        
        # TODO Add MLflow cross-validation metrics logging (Workshop 4)
        if mlflow.active_run():
            # Log cross-validation results (metrics only - must be numeric)

            # Add additional CV metadata (metrics only - must be numeric)

            # Log strategy as parameter (strings allowed in parameters)
            mlflow.log_params({
                'key1': value1,
                'key2': value2,
            })

        # Logging
        if logger.level >= LogLevel.NORMAL:
            print(f"  Average: RMSE={cv_results['rmse_mean']:.3f}±{cv_results['rmse_std']:.3f}")
        
        return cv_results
    
    
    def hyperparameter_optimization_cv(self, model, param_grid, X, y, groups=None):
        """
        Perform hyperparameter optimization using GridSearchCV with geographic cross-validation.
        
        This method combines GridSearchCV with GroupKFold to ensure that entire cities
        are either in training OR validation during hyperparameter search, preventing data leakage.
        
        Args:
            model: Scikit-learn model to optimize
            param_grid: Dictionary of hyperparameters to search
            X: Feature matrix
            y: Target variable
            groups: Grouping variable for GroupKFold (e.g., cities)
            
        Returns:
            Tuple of (best_model, best_params, best_score)
        """
        logger = get_logger()
        logger.info(f"Optimizing hyperparameters for {model.__class__.__name__}...", LogLevel.NORMAL)
        
        # Add hyperparameter optimization with geographic cross-validation (Workshop 3)
        # Configure GridSearchCV with geographic cross-validation

        gscv = GridSearchCV(model,
                            param_grid,
                            n_jobs=-1,
                            scoring='neg_root_mean_squared_error',
                            verbose=logger.level # TODO check if it works (idk)
        )

        # Fit GridSearchCV
        if groups is not None:
            n_unique_groups = len(np.unique(groups))
            cv = GroupKFold(n_splits=min(N_SPLITS, n_unique_groups))
            gscv.fit(X, y, groups=groups, cv=cv)
        else:
            gscv.fit(X, y)

        # Extract results (GridSearchCV returns negative RMSE, convert to positive)

        best_model = gscv.best_estimator_
        best_params = gscv.best_params_
        best_score = -gscv.best_score_ #Since output is negative

        # Logging
        logger.success(f"Best RMSE: {best_score:.3f}")
        if logger.level >= LogLevel.NORMAL:
            print(f"  Best parameters: {best_params}")
        
        return best_model, best_params, best_score

