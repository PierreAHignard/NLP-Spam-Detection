"""
Technical validation tests for ModelTrainer class.

These tests validate the model training functionality including
model creation, training, feature importance, and model persistence.
"""

import pandas as pd
import numpy as np
import pytest
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression

from pipeline.model_trainer import ModelTrainer
from utils.config import MODEL_TYPES, TARGET_COL


class TestModelTrainer:
    """Test suite for ModelTrainer technical validation."""
    
    def test_create_model_linear(self):
        """Test creation of linear regression model."""
        trainer = ModelTrainer()
        
        # Create linear model
        model = trainer.create_model('linear')
        
        # Assertions
        assert isinstance(model, LinearRegression), "Should create LinearRegression instance"
        assert hasattr(model, 'fit'), "Model should have fit method"
        assert hasattr(model, 'predict'), "Model should have predict method"
    
    
    def test_create_model_invalid_type(self):
        """Test error handling for invalid model type."""
        trainer = ModelTrainer()
        
        # Test invalid model type
        with pytest.raises(ValueError, match="Unknown model type"):
            trainer.create_model('invalid_model')
    
    def test_train_single_model(self, sample_X_y):
        """Test training a single model."""
        trainer = ModelTrainer()
        X, y = sample_X_y
        
        # Train linear model
        model = trainer.train_single_model(X, y, model_type='linear')
        
        # Assertions
        assert isinstance(model, LinearRegression), "Should return trained LinearRegression"
        assert hasattr(model, 'coef_'), "Trained model should have coefficients"
        assert hasattr(model, 'intercept_'), "Trained model should have intercept"
        assert 'linear' in trainer.trained_models, "Model should be stored in trained_models"
        assert trainer.trained_models['linear'] is model, "Stored model should be the same instance"
        
        # Check that model can make predictions
        predictions = model.predict(X)
        assert len(predictions) == len(y), "Predictions should have same length as target"
        assert isinstance(predictions, np.ndarray), "Predictions should be numpy array"
    
    def test_train_multiple_models(self, sample_X_y):
        """Test training multiple model types."""
        trainer = ModelTrainer()
        X, y = sample_X_y
        
        # Train all model types from config
        trained_models = {}
        
        for model_type in MODEL_TYPES:
            model = trainer.train_single_model(X, y, model_type=model_type)
            trained_models[model_type] = model
        
        # Assertions
        assert len(trainer.trained_models) == len(MODEL_TYPES), f"Should have trained {len(MODEL_TYPES)} models"
        for model_type in MODEL_TYPES:
            assert model_type in trainer.trained_models, f"Should have {model_type} model"
            
            # Check appropriate attributes based on model type
            if model_type == 'linear':
                # Linear models have coefficients
                assert hasattr(trainer.trained_models[model_type], 'coef_'), f"{model_type} should have coefficients"
            elif model_type in ['xgboost', 'lightgbm']:
                # Tree-based models have feature importances
                assert hasattr(trainer.trained_models[model_type], 'feature_importances_'), f"{model_type} should have feature importances"
            
            # All models should have predict method
            assert hasattr(trainer.trained_models[model_type], 'predict'), f"{model_type} should have predict method"
    
    def test_predict(self, sample_X_y):
        """Test model prediction functionality."""
        trainer = ModelTrainer()
        X, y = sample_X_y
        
        # Train a model
        model = trainer.train_single_model(X, y, model_type='linear')
        
        # Make predictions
        predictions = trainer.predict(model, X)
        
        # Assertions
        assert isinstance(predictions, np.ndarray), "Predictions should be numpy array"
        assert len(predictions) == len(X), "Predictions should have same length as input"
        assert predictions.dtype in [np.float64, np.float32], "Predictions should be float type"
        
        # Check that predictions are reasonable (not all zeros or identical)
        assert not np.all(predictions == 0), "Predictions should not all be zero"
        assert len(np.unique(predictions)) > 1, "Predictions should have some variation"
    
    def test_model_training_with_parameters(self, sample_X_y):
        """Test model training with custom parameters."""
        trainer = ModelTrainer()
        X, y = sample_X_y
        
        # Train linear model with custom parameters
        model = trainer.train_single_model(X, y, model_type='linear', fit_intercept=False)
        
        # Assertions
        assert isinstance(model, LinearRegression), "Should create LinearRegression model"
        assert model.fit_intercept == False, "Should use custom fit_intercept parameter"
    
    def test_trained_models_storage(self, sample_X_y):
        """Test that trained models are properly stored."""
        trainer = ModelTrainer()
        X, y = sample_X_y
        
        # Initially no models
        assert len(trainer.trained_models) == 0, "Should start with no trained models"
        
        # Train all available model types
        for i, model_type in enumerate(MODEL_TYPES):
            trainer.train_single_model(X, y, model_type=model_type)
            assert len(trainer.trained_models) == i + 1, f"Should have {i + 1} model(s) after training"
            assert model_type in trainer.trained_models, f"Should store {model_type} model"
        
        # Check that all models are stored
        assert len(trainer.trained_models) == len(MODEL_TYPES), f"Should have {len(MODEL_TYPES)} total models"
        
        # If there are multiple model types, check that models are different instances
        if len(MODEL_TYPES) > 1:
            model_instances = list(trainer.trained_models.values())
            for i in range(len(model_instances)):
                for j in range(i + 1, len(model_instances)):
                    assert model_instances[i] is not model_instances[j], "Models should be different instances"


def run_model_trainer_tests():
    """
    Function to run all ModelTrainer tests programmatically.
    """
    import pytest
    
    result = pytest.main([__file__, "-v", "--tb=short"])
    
    if result == 0:
        print("✅ All ModelTrainer tests passed!")
        return True
    else:
        print("❌ Some ModelTrainer tests failed!")
        return False


if __name__ == "__main__":
    run_model_trainer_tests()
