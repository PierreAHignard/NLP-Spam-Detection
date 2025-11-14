"""
Configuration for Air Quality ML Pipeline.

This module contains all configuration constants used throughout
the pipeline. Students don't need to modify this file.
"""

from pathlib import Path

# =============================================================================
# PROJECT PATHS
# =============================================================================

# Base project directory (automatically detected)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data"

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# File names
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"

# Column names
ID_COL = "id"
TARGET_COL = "pm2_5"
DATE_COL = "date"
CITY_COL = "city"
COUNTRY_COL = "country"
LATITUDE_COL = "site_latitude"
LONGITUDE_COL = "site_longitude"

# =============================================================================
# PREPROCESSING CONFIGURATION
# =============================================================================

# Missing data threshold (drop columns with more than X% missing)
MISSING_THRESHOLD = 0.7

# Cross-validation splits
N_SPLITS = 4

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================

# Temporal features to extract
TEMPORAL_FEATURES = ["year", "month", "day", "quarter", "week", "dayofweek"]

# Number of features to select
N_FEATURES_SELECTKBEST = 15
N_FEATURES_RFE = 15

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Available model types
MODEL_TYPES = ["linear"]

# TODO Add XGBoost and LightGBM support (Workshop 3)
# Available model types
MODEL_TYPES = ["linear", "xgboost", "lightgbm"]

# Default hyperparameter grids for optimization
DEFAULT_PARAM_GRIDS = {
    "xgboost": {
        'n_estimators': [100],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.7, 1.0],
        'colsample_bytree': [0.7, 1.0]
    },
    "lightgbm": {
        'n_estimators': [100],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.7, 1.0],
        'colsample_bytree': [0.7, 1.0]
    }
}
# TODO_END

# Random state for reproducibility
RANDOM_STATE = 42


# =============================================================================
# MLFLOW CONFIGURATION
# =============================================================================

# TODO Add MLflow tracking configuration (Workshop 4)
MLFLOW_EXPERIMENT_NAME = "air_quality_ml"
MLFLOW_TRACKING_URI = "./mlruns"
# TODO_END

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================


# Metrics to calculate
METRICS = ["rmse", "mae", "r2", "mape"]

def get_data_file_path(filename):
    """Get full path to a data file."""
    return DATA_PATH / filename
