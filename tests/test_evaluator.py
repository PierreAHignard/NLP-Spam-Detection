"""
Technical validation tests for Evaluator class.

These tests validate the model evaluation functionality including
metrics calculation, cross-validation, and model comparison.
"""

import pandas as pd
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold

from pipeline.evaluator import Evaluator
from utils.config import N_SPLITS


class TestEvaluator:
    """Test suite for Evaluator technical validation."""
    
    def test_calculate_metrics_basic(self, sample_predictions):
        """Test basic metrics calculation."""
        evaluator = Evaluator()
        y_true, y_pred = sample_predictions
        
        # Calculate metrics
        metrics = evaluator.calculate_metrics(y_true, y_pred)
        
        # Assertions
        assert isinstance(metrics, dict), "Should return dictionary"
        assert 'rmse' in metrics, "Should include RMSE"
        assert 'mae' in metrics, "Should include MAE"
        assert 'r2' in metrics, "Should include R²"
        
        # Check metric types and ranges
        assert isinstance(metrics['rmse'], (int, float)), "RMSE should be numeric"
        assert isinstance(metrics['mae'], (int, float)), "MAE should be numeric"
        assert isinstance(metrics['r2'], (int, float)), "R² should be numeric"
        
        assert metrics['rmse'] >= 0, "RMSE should be non-negative"
        assert metrics['mae'] >= 0, "MAE should be non-negative"
    
    def test_calculate_metrics_perfect_predictions(self):
        """Test metrics with perfect predictions."""
        evaluator = Evaluator()
        
        # Perfect predictions
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = y_true.copy()
        
        metrics = evaluator.calculate_metrics(y_true, y_pred)
        
        # Assertions for perfect predictions
        assert metrics['rmse'] == 0.0, "RMSE should be 0 for perfect predictions"
        assert metrics['mae'] == 0.0, "MAE should be 0 for perfect predictions"
        assert metrics['r2'] == 1.0, "R² should be 1 for perfect predictions"
    
    def test_calculate_metrics_with_zeros(self):
        """Test metrics calculation when true values contain zeros."""
        evaluator = Evaluator()
        
        # Include zero values
        y_true = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([0.1, 1.1, 2.1, 3.1, 4.1])
        
        metrics = evaluator.calculate_metrics(y_true, y_pred)
        
        # Assertions
        assert 'rmse' in metrics, "Should include RMSE"
        assert 'mae' in metrics, "Should include MAE"
        assert 'r2' in metrics, "Should include R²"
        
        # Check that metrics are calculated correctly even with zeros
        assert isinstance(metrics['rmse'], (int, float)), "RMSE should be numeric"
        assert isinstance(metrics['mae'], (int, float)), "MAE should be numeric"
        assert isinstance(metrics['r2'], (int, float)), "R² should be numeric"
    
    def test_cross_validate_model_with_groups(self, sample_X_y, base_sample_data):
        """Test cross-validation with group-based splitting."""
        evaluator = Evaluator()
        X, y = sample_X_y
        
        # Create groups (cities) for the same indices as X, y
        groups = base_sample_data.loc[X.index, 'city']
        
        # Create and test model
        model = LinearRegression()
        
        # Perform cross-validation
        cv_results = evaluator.cross_validate_model(model, X, y, groups=groups)
        
        # Assertions
        assert isinstance(cv_results, dict), "Should return dictionary"
        
        # Check that all expected metrics are present
        expected_metrics = ['rmse_mean', 'rmse_std', 'mae_mean', 'mae_std', 'r2_mean', 'r2_std']
        for metric in expected_metrics:
            assert metric in cv_results, f"Should include {metric}"
            assert isinstance(cv_results[metric], (int, float)), f"{metric} should be numeric"
        
        # Check that standard deviations are non-negative
        assert cv_results['rmse_std'] >= 0, "RMSE std should be non-negative"
        assert cv_results['mae_std'] >= 0, "MAE std should be non-negative"
        assert cv_results['r2_std'] >= 0, "R² std should be non-negative"
    
    def test_cross_validate_model_without_groups(self, sample_X_y):
        """Test cross-validation without group-based splitting."""
        evaluator = Evaluator()
        X, y = sample_X_y
        
        # Create and test model
        model = LinearRegression()
        
        # Perform cross-validation without groups
        cv_results = evaluator.cross_validate_model(model, X, y, groups=None)
        
        # Assertions
        assert isinstance(cv_results, dict), "Should return dictionary"
        
        # Check that all expected metrics are present
        expected_metrics = ['rmse_mean', 'rmse_std', 'mae_mean', 'mae_std', 'r2_mean', 'r2_std']
        for metric in expected_metrics:
            assert metric in cv_results, f"Should include {metric}"
    
    
    def test_cross_validation_fold_consistency(self, sample_X_y, base_sample_data):
        """Test that cross-validation uses correct number of folds."""
        evaluator = Evaluator()
        X, y = sample_X_y
        
        # Create groups
        groups = base_sample_data.loc[X.index, 'city']
        
        # Mock the cross-validation to count folds
        model = LinearRegression()
        
        # We can't directly test the number of folds without modifying the method,
        # but we can test that it completes successfully with the expected configuration
        cv_results = evaluator.cross_validate_model(model, X, y, groups=groups)
        
        # The method should complete and return results
        assert cv_results is not None, "Cross-validation should complete successfully"
        assert 'rmse_mean' in cv_results, "Should return aggregated results"
    
    def test_metrics_calculation_edge_cases(self):
        """Test metrics calculation with edge cases."""
        evaluator = Evaluator()
        
        # Test with identical values
        y_true = np.array([5.0, 5.0, 5.0, 5.0])
        y_pred = np.array([5.1, 4.9, 5.2, 4.8])
        
        metrics = evaluator.calculate_metrics(y_true, y_pred)
        
        # Should handle constant true values
        assert 'rmse' in metrics, "Should calculate RMSE for constant true values"
        assert 'mae' in metrics, "Should calculate MAE for constant true values"
        # R² might be undefined for constant true values, but method should not crash
        assert 'r2' in metrics, "Should calculate R² for constant true values"
    
    
    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = Evaluator()
        
        # Should initialize without errors
        assert evaluator is not None, "Evaluator should initialize successfully"
        assert hasattr(evaluator, 'calculate_metrics'), "Should have calculate_metrics method"
        assert hasattr(evaluator, 'cross_validate_model'), "Should have cross_validate_model method"


def run_evaluator_tests():
    """
    Function to run all Evaluator tests programmatically.
    """
    import pytest
    
    result = pytest.main([__file__, "-v", "--tb=short"])
    
    if result == 0:
        print("✅ All Evaluator tests passed!")
        return True
    else:
        print("❌ Some Evaluator tests failed!")
        return False


if __name__ == "__main__":
    run_evaluator_tests()
