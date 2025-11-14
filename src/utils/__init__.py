"""
Air Quality ML Utilities Package

This package contains utility functions and configurations:
- config: Configuration constants and settings
- utils: General utility functions
- evaluation_utils: Detailed evaluation functions with visualizations

Usage:
    from utils.config import *
    from utils.utils import print_step_header
    from utils.evaluation_utils import evaluate_model_detailed
"""

__version__ = "1.0.0"

# Import commonly used items
from .config import *
from .utils import print_step_header, print_results_summary

__all__ = [
    # Config constants
    'DATA_PATH', 'TARGET_COL', 'CITY_COL', 'DATE_COL',
    'MODEL_TYPES', 'N_SPLITS', 'RANDOM_STATE',
    # Utility functions
    'print_step_header', 'print_results_summary'
]
