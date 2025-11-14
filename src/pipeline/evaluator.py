"""
Model evaluation module for Air Quality ML Pipeline.

This module provides core evaluation functionality used by the package.
For detailed evaluation with visualizations, see utils.evaluation_utils.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

# TODO Import GridSearchCV for hyperparameter optimization (Workshop 3)

# TODO Import MLflow (Workshop 4)

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
        # TODO Calculate comprehensive regression metrics
        
        # Create metrics dictionary
        metrics = {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
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
        
        fold_results = []
        # TODO Perform cross-validation enumerating folds
            # Split data
            
            # Train model
            
            # Predict
            
            # Calculate metrics and append to fold_results
            
            # Logging
        
        # Aggregate results
        cv_results = {}
        # Enumerate metrics and calculate mean/std across folds
        for metric in fold_results[0].keys():
            values = [fold[metric] for fold in fold_results]
            cv_results[f'{metric}_mean'] = np.mean(values)
            cv_results[f'{metric}_std'] = np.std(values)
        
        # TODO Add MLflow cross-validation metrics logging (Workshop 4)       
            # Log cross-validation results (metrics only - must be numeric)
            
            # Add additional CV metadata (metrics only - must be numeric)
            
            
            # Log strategy as parameter (strings allowed in parameters)

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
        
        # TODO Add hyperparameter optimization with geographic cross-validation (Workshop 3)
        # Set up cross-validation strategy
        
        # Configure GridSearchCV with geographic cross-validation
        
        # Fit GridSearchCV
        
        # Extract results (GridSearchCV returns negative RMSE, convert to positive)
        
        # Logging
        logger.success(f"Best RMSE: {best_score:.3f}")
        if logger.level >= LogLevel.NORMAL:
            print(f"  Best parameters: {best_params}")
        
        return best_model, best_params, best_score

