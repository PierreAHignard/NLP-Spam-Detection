"""
Air Quality ML Package - Restructured

This package has been restructured into two main components:

1. Pipeline: Core machine learning pipeline components
   - DataProcessor: Data loading and preprocessing
   - FeatureEngineer: Feature extraction and selection
   - ModelTrainer: Model training and comparison
   - Evaluator: Core model evaluation

2. Utils: Utilities and configuration
   - config: Configuration constants and settings
   - utils: General utility functions
   - evaluation_utils: Detailed evaluation functions with visualizations

Usage:
    from pipeline import DataProcessor, FeatureEngineer, ModelTrainer, Evaluator
    from utils.config import *
    from utils.evaluation_utils import evaluate_model_detailed
"""

__version__ = "2.0.0"
__author__ = "Air Quality ML Workshop - Restructured"

# Import main pipeline classes for backward compatibility
from pipeline import DataProcessor, FeatureEngineer, ModelTrainer, Evaluator

__all__ = [
    'DataProcessor',
    'FeatureEngineer', 
    'ModelTrainer',
    'Evaluator'
]
