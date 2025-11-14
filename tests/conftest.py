"""
Shared fixtures for all test modules.

This module contains common fixtures used across multiple test files
to ensure consistency and reduce code duplication.
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from utils.config import TARGET_COL, CITY_COL, DATE_COL


@pytest.fixture
def base_sample_data():
    """Create base sample data for testing all modules."""
    # Create realistic sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    cities = ['CityA', 'CityB', 'CityC', 'CityD']
    
    data = []
    for i, date in enumerate(dates):
        for city in cities:
            data.append({
                'id': f'id_{len(data):06d}_{city.lower()}',
                'date': date,
                'city': city,
                'pm2_5': np.random.normal(25, 10),
                'feature1': np.random.normal(0, 1),
                'feature2': np.random.normal(10, 5),
                'feature3': np.random.normal(5, 2),
                'site_latitude': np.random.uniform(-10, 10),
                'site_longitude': np.random.uniform(-10, 10),
                'country': f'Country_{city[-1]}'
            })
    
    df = pd.DataFrame(data)
    
    # Introduce some missing values for testing
    df.loc[0:10, 'feature1'] = np.nan
    df.loc[50:60, 'feature2'] = np.nan
    df.loc[5:8, 'pm2_5'] = np.nan
    
    return df


@pytest.fixture
def sample_train_test_data(base_sample_data):
    """Create train/test split from base sample data."""
    train_data = base_sample_data.iloc[:300].copy()
    test_data = base_sample_data.iloc[300:].copy()
    
    # Remove target from test data (realistic scenario)
    test_data = test_data.drop(columns=[TARGET_COL])
    
    return train_data, test_data


@pytest.fixture
def sample_features_data(base_sample_data):
    """Create sample data with additional engineered features."""
    df = base_sample_data.copy()
    
    # Add some engineered features that would be created by FeatureEngineer
    df['year'] = df[DATE_COL].dt.year
    df['month'] = df[DATE_COL].dt.month
    df['day'] = df[DATE_COL].dt.day
    df['dayofweek'] = df[DATE_COL].dt.dayofweek
    df['location'] = df['site_latitude'].astype('str') + '_' + df['site_longitude'].astype('str')
    
    return df


@pytest.fixture
def sample_X_y(sample_features_data):
    """Create feature matrix X and target y for model testing."""
    # Get feature columns (exclude metadata, target, and non-numeric columns)
    exclude_cols = ['id', 'date', 'city', 'country', 'site_latitude', 'site_longitude', 'location', TARGET_COL]
    feature_cols = [col for col in sample_features_data.columns if col not in exclude_cols]
    
    # Only keep numeric columns
    numeric_data = sample_features_data[feature_cols].select_dtypes(include=[np.number])
    
    X = numeric_data.dropna()
    y = sample_features_data.loc[X.index, TARGET_COL]
    
    return X, y


@pytest.fixture
def sample_predictions():
    """Create sample predictions for evaluation testing."""
    np.random.seed(42)
    n_samples = 100
    
    # Create realistic predictions with some noise
    y_true = np.random.normal(25, 10, n_samples)
    y_pred = y_true + np.random.normal(0, 2, n_samples)  # Add prediction error
    
    return y_true, y_pred


@pytest.fixture
def temp_model_path(tmp_path):
    """Create temporary path for model saving/loading tests."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir
