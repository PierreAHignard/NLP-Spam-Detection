"""
Air Quality ML Pipeline - Test Suite

This package contains comprehensive tests for validating the technical implementation
of the Air Quality ML Pipeline components.

Test Structure:
- test_data_processor.py: Technical validation for DataProcessor class
- test_feature_engineer.py: Technical validation for FeatureEngineer class (to be added)
- test_model_trainer.py: Technical validation for ModelTrainer class (to be added)
- test_integration.py: End-to-end integration tests (to be added)

Usage:
    # Run all tests
    python scripts/run_tests.py
    
    # Run specific test module
    python scripts/run_tests.py --module data_processor
    
    # Run with verbose output
    python scripts/run_tests.py --verbose
    
    # Run with coverage report
    python scripts/run_tests.py --coverage
"""

__version__ = "1.0.0"
__author__ = "Air Quality ML Workshop - Test Suite"

# Import test modules for easy access
try:
    from .test_data_processor import TestDataProcessor, run_dataprocessor_tests
    DATAPROCESSOR_AVAILABLE = True
except ImportError:
    DATAPROCESSOR_AVAILABLE = False

# Placeholder for future test imports
# from .test_feature_engineer import TestFeatureEngineer, run_feature_engineer_tests
# from .test_model_trainer import TestModelTrainer, run_model_trainer_tests
# from .test_integration import TestIntegration, run_integration_tests

__all__ = [
    'TestDataProcessor',
    'run_dataprocessor_tests',
    # Future test classes will be added here
]


def get_available_test_modules():
    """
    Get list of available test modules.
    
    Returns:
        dict: Dictionary mapping module names to their availability status
    """
    modules = {
        'data_processor': DATAPROCESSOR_AVAILABLE,
        'feature_engineer': False,  # To be implemented
        'model_trainer': False,     # To be implemented
        'integration': False,       # To be implemented
    }
    return modules


def run_all_available_tests():
    """
    Run all available test modules.
    
    Returns:
        dict: Results of each test module execution
    """
    results = {}
    
    # Run DataProcessor tests if available
    if DATAPROCESSOR_AVAILABLE:
        try:
            results['data_processor'] = run_dataprocessor_tests()
        except Exception as e:
            results['data_processor'] = f"Error: {str(e)}"
    
    # Future test modules will be added here
    # if FEATURE_ENGINEER_AVAILABLE:
    #     results['feature_engineer'] = run_feature_engineer_tests()
    
    return results
