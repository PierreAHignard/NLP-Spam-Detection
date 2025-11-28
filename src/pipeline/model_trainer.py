"""
Model training module for Air Quality ML Pipeline.

This module handles model training, comparison and evaluation.
Students need to complete the sections.
"""

from sklearn.linear_model import LinearRegression, Ridge, Lasso

import mlflow

from utils.config import MODEL_TYPES, RANDOM_STATE
from .evaluator import Evaluator
from utils.logger import get_logger
from sklearn.linear_model import LogisticRegression



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
        
        # Create model instance based on model_type
        if model_type == 'logistic':
            model = LogisticRegression(**params)

        else:
            raise ValueError(f"Unknown model type: {model_type}. Available: {MODEL_TYPES}")
        
        return model
    
    def train_single_model(self, X, y, model_type='logistic', **model_params):
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

        # Add MLflow parameter logging (Workshop 4)

        if mlflow.active_run():
            mlflow.log_params({
                'model_type': model_type,
                **model_params  # Unpacks additional parameters
            })
        
        # Train the model
        model = self.create_model(model_type, **model_params)
        model = model.fit(X, y)

        # Store trained model
        self.trained_models[model_type] = model
        y_pred = model.predict(X)

        # Calculate training score
        metrics = self.evaluator.calculate_metrics(y, y_pred)
        train_score = metrics["accuracy"]

        # Logging
        with logger.indent():
            logger.model_info(f"Training accuracy: {train_score:.4f}")
        
        logger.success(f"{model_type.title()} model training completed")

        return model
    
    
    def predict(self, model, X):
        """
        Make predictions using a trained model.
        
        Args:
            model: Trained model
            X: Tokenised texts
            
        Returns:
            Predictions array
        """
        logger = get_logger()
        logger.substep(f"Starting model inference on {len(X)} samples")
        
        # Make predictions using model
        predictions = model.predict(X)

        # Logging
        logger.success(f"Generated {len(predictions)} predictions")
        
        return predictions
