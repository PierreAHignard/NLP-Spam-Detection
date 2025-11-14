"""
Air Quality ML Pipeline Package

This package contains the core machine learning pipeline components:
- DataProcessor: Data loading and preprocessing
- FeatureEngineer: Feature extraction and selection
- ModelTrainer: Model training and comparison
- Evaluator: Core model evaluation

Usage:
    from pipeline import DataProcessor, FeatureEngineer, ModelTrainer, Evaluator
"""

__version__ = "1.0.0"

# Import main classes for easy access
from .data_processor import DataProcessor
from .feature_engineer import FeatureEngineer
from .model_trainer import ModelTrainer
from .evaluator import Evaluator

__all__ = [
    'DataProcessor',
    'FeatureEngineer', 
    'ModelTrainer',
    'Evaluator'
]
