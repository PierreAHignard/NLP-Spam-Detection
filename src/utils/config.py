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
SMS_FILE = "sms_spam.csv"
EMAIL_FILE = "email_spam.csv"

# Column names
MESSAGE_COL = "message"
LABEL_COL = "label"

# =============================================================================
# PREPROCESSING CONFIGURATION
# =============================================================================

# Train-Test Split
TRAIN_TEST_SPLIT_SIZE = 0.2

# =============================================================================
# FEATURE ENGINEERING CONFIGURATION
# =============================================================================

TOKEN_REGEX = r"(\S+)"


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Available model types
MODEL_TYPES = ["Logistic_Regression", "Multinomial_NB", "Linear_SVC", "SGD_Classifier", "Random_Forest"]

# Default hyperparameter grids for optimization
DEFAULT_PARAM_GRIDS = {
    "Logistic_Regression": {
        # C: Inverse of regularization strength (smaller = stronger regularization)
        "C": [0.1, 1.0, 10.0],
        # L1 is good for sparse text (feature selection), L2 is standard
        "penalty": ["l1", "l2"]
    },

    "Multinomial_NB": {
        # Alpha: Additive smoothing (Laplace/Lidstone).
        # 0.0 means no smoothing, 1.0 is standard Laplace.
        "alpha": [0.01, 0.1, 0.5, 1.0],
        # Whether to learn class prior probabilities or assume uniform
        "fit_prior": [True, False]
    },

    "Linear_SVC": {
        "C": [0.1, 1.0, 10.0],
        # 'hinge' is the standard SVM loss, 'squared_hinge' is differentiable
        "loss": ["hinge", "squared_hinge"],
    },

    "SGD_Classifier": {
        # 'hinge' = SVM, 'log_loss' = Logistic Regression, 'perceptron'
        "loss": ["hinge", "log_loss"],
        # Regularization type
        "penalty": ["l2", "l1", "elasticnet"],
        # Constant that multiplies the regularization term (lambda)
        "alpha": [1e-4, 1e-3, 1e-2]
    },

    "Random_Forest": {
        "n_estimators": [50, 100, 200],
        # Max depth is crucial for NLP to prevent overfitting on noise
        "max_depth": [None, 10, 30],
        # Minimum samples required to split an internal node
        "min_samples_split": [2, 5]
    }
}

CROSS_VALIDATION = 5

# Random state for reproducibility
RANDOM_STATE = 42


# =============================================================================
# MLFLOW CONFIGURATION
# =============================================================================

# Add MLflow tracking configuration (Workshop 4)
MLFLOW_EXPERIMENT_NAME = "Spam Detection"
MLFLOW_TRACKING_URI = "./mlruns"
# TODO_END

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

# Metrics to calculate
METRICS = ["accuracy", "recall", "precision"]

def get_data_file_path(filename):
    """Get full path to a data file."""
    return DATA_PATH / filename
