"""
Model training module for Air Quality ML Pipeline.

This module handles model training, comparison and evaluation.
Students need to complete the TODO sections.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# TODO Import additional models (Workshop 3)

# TODO Import MLflow (Workshop 4)

from utils.config import MODEL_TYPES, RANDOM_STATE, TARGET_COL, CITY_COL
from .evaluator import Evaluator
from utils.logger import get_logger



class ModelTrainer:
    """
    Model trainer for air quality prediction.
    
    Handles training of different regression models, model comparison,
    and model persistence.
    """
    
    def __init__(self):
        """Initialize the model trainer."""
        self.trained_models = {}
        self.evaluator = Evaluator()
        self.best_model = None
        self.best_model_name = None
    
    def create_model(self, model_type, **params):
        """
        Create a model instance of the specified type.
        
        Args:
            model_type: Type of model ('linear', ...)
            **params: Model-specific parameters
            
        Returns:
            Initialized model instance
        """
        if model_type not in MODEL_TYPES:
            raise ValueError(f"Unknown model type: {model_type}. Available: {MODEL_TYPES}")
        
        # Create model instance based on model_type
        if model_type == 'linear':
            model = LinearRegression(**params)
        # TODO Add XGBoost and LightGBM model creation (Workshop 3)
        else:
            raise ValueError(f"Unknown model type: {model_type}. Available: {MODEL_TYPES}")
        
        return model
    
    def train_single_model(self, X, y, model_type='linear', **model_params):
        """
        Train a single model.
        
        Args:
            X: Feature matrix
            y: Target variable
            model_type: Type of model to train
            **model_params: Model-specific parameters
            
        Returns:
            Trained model instance
        """
        logger = get_logger()
        logger.substep(f"Training {model_type.title()} Model")

        # TODO Add MLflow parameter logging (Workshop 4)
        
        # TODO Train the model
        
        # TODO Store trained model
        
        # TODO Calculate training score

        # Logging
        with logger.indent():
            logger.model_info(f"Training R² score: {train_score:.4f}")
        
        logger.success(f"{model_type.title()} model training completed")

        return model
    
    
    def predict(self, model, X):
        """
        Make predictions using a trained model.
        
        Args:
            model: Trained model
            X: Feature matrix
            
        Returns:
            Predictions array
        """
        logger = get_logger()
        
        # TODO Make predictions using model
        
        # Logging
        logger.success(f"Generated {len(predictions)} predictions")
        
        return predictions
