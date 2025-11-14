"""
Utility functions for Air Quality ML Pipeline.

This module contains helper functions used across the pipeline.
Students don't need to modify this file.
"""

import pandas as pd
import numpy as np
from .logger import get_logger


def print_dataframe_info(df, name="DataFrame"):
    """Print useful information about a DataFrame - uses new logger."""
    logger = get_logger()
    logger.dataframe_info(df, name)


def print_step_header(step_number, step_name):
    """Print a formatted step header - uses new logger."""
    logger = get_logger()
    logger.step(step_name, step_number)


def print_func_header(step_name):
    """Print a formatted function header - uses new logger."""
    logger = get_logger()
    logger.substep(step_name)


def print_results_summary(results_dict):
    """Print a formatted summary of results - uses new logger."""
    logger = get_logger()
    logger.results_summary(results_dict)


def validate_data_files():
    """Check if required data files exist."""
    from .config import DATA_PATH, TRAIN_FILE, TEST_FILE
    
    logger = get_logger()
    train_path = DATA_PATH / TRAIN_FILE
    test_path = DATA_PATH / TEST_FILE
    
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    
    logger.success("Data files found")
    return True


def format_time_elapsed(start_time, end_time):
    """Format elapsed time in a readable way."""
    elapsed = end_time - start_time
    if elapsed < 60:
        return f"{elapsed:.1f} seconds"
    else:
        minutes = int(elapsed // 60)
        seconds = elapsed % 60
        return f"{minutes}m {seconds:.1f}s"
