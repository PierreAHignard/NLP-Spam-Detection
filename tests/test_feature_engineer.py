"""
Technical validation tests for FeatureEngineer class.

These tests validate the feature engineering functionality including
temporal features, geographic features, categorical encoding, and feature selection.
"""

import pandas as pd
import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from pipeline.feature_engineer import FeatureEngineer
from utils.config import TARGET_COL, CITY_COL, DATE_COL, TEMPORAL_FEATURES


class TestFeatureEngineer:
    """Test suite for FeatureEngineer technical validation."""
    
    def test_extract_temporal_features(self, base_sample_data):
        """Test temporal feature extraction."""
        engineer = FeatureEngineer()
        
        # Extract temporal features
        df_with_temporal = engineer.extract_temporal_features(base_sample_data)
        
        # Assertions
        assert isinstance(df_with_temporal, pd.DataFrame), "Should return DataFrame"
        assert df_with_temporal.shape[0] == base_sample_data.shape[0], "Row count should remain same"
        
        # Check that temporal features are added
        expected_temporal_cols = ['year', 'month', 'day', 'quarter', 'week', 'dayofweek']
        for col in expected_temporal_cols:
            assert col in df_with_temporal.columns, f"Temporal feature '{col}' should be added"
        
        # Check data types and ranges
        assert df_with_temporal['year'].dtype in [int, 'int64', 'int32'], "Year should be integer"
        assert df_with_temporal['month'].min() >= 1, "Month should be >= 1"
        assert df_with_temporal['month'].max() <= 12, "Month should be <= 12"
        assert df_with_temporal['dayofweek'].min() >= 0, "Dayofweek should be >= 0"
        assert df_with_temporal['dayofweek'].max() <= 6, "Dayofweek should be <= 6"
    
    def test_extract_geographic_features(self, base_sample_data):
        """Test geographic feature extraction."""
        engineer = FeatureEngineer()
        
        # Extract geographic features
        df_with_geo = engineer.extract_geographic_features(base_sample_data)
        
        # Assertions
        assert isinstance(df_with_geo, pd.DataFrame), "Should return DataFrame"
        assert df_with_geo.shape[0] == base_sample_data.shape[0], "Row count should remain same"
        assert 'location' in df_with_geo.columns, "Location feature should be added"
        
        # Check that location combines lat/lon
        sample_location = df_with_geo['location'].iloc[0]
        assert isinstance(sample_location, str), "Location should be string"
        assert '_' in sample_location, "Location should combine lat_lon with underscore"
        
        # Check uniqueness
        n_unique_coords = base_sample_data[['site_latitude', 'site_longitude']].drop_duplicates().shape[0]
        n_unique_locations = df_with_geo['location'].nunique()
        assert n_unique_locations == n_unique_coords, "Number of unique locations should match unique coordinates"
    
    def test_encode_categorical_features(self, sample_train_test_data):
        """Test categorical feature encoding."""
        engineer = FeatureEngineer()
        train_data, test_data = sample_train_test_data
        
        # Add temporal and geographic features first
        train_with_features = engineer.extract_temporal_features(train_data)
        train_with_features = engineer.extract_geographic_features(train_with_features)
        
        test_with_features = engineer.extract_temporal_features(test_data)
        test_with_features = engineer.extract_geographic_features(test_with_features)
        
        # Encode categorical features
        train_encoded, test_encoded = engineer.encode_categorical_features(
            train_with_features, test_with_features
        )
        
        # Assertions
        assert isinstance(train_encoded, pd.DataFrame), "Should return DataFrame for training"
        assert isinstance(test_encoded, pd.DataFrame), "Should return DataFrame for test"
        assert train_encoded.shape[0] == train_with_features.shape[0], "Training row count should remain same"
        assert test_encoded.shape[0] == test_with_features.shape[0], "Test row count should remain same"
        
        # Check that categorical columns are encoded (should be numeric)
        categorical_cols = ['location', 'date']
        for col in categorical_cols:
            if col in train_encoded.columns:
                assert pd.api.types.is_numeric_dtype(train_encoded[col]), f"Column '{col}' should be numeric after encoding"
                assert pd.api.types.is_numeric_dtype(test_encoded[col]), f"Column '{col}' should be numeric after encoding"
    
    def test_get_feature_columns(self, sample_features_data):
        """Test feature column identification."""
        engineer = FeatureEngineer()
        
        # Test with default parameters
        feature_cols = engineer.get_feature_columns(sample_features_data)
        
        # Assertions
        assert isinstance(feature_cols, list), "Should return list of column names"
        assert len(feature_cols) > 0, "Should return at least some feature columns"
        
        # Check that target and metadata columns are excluded by default
        assert TARGET_COL not in feature_cols, "Target column should be excluded by default"
        assert 'id' not in feature_cols, "ID column should be excluded by default"
        assert 'city' not in feature_cols, "City column should be excluded by default"
        
        # Test with exclude_target=False
        feature_cols_with_target = engineer.get_feature_columns(
            sample_features_data, exclude_target=False
        )
        assert TARGET_COL in feature_cols_with_target, "Target should be included when exclude_target=False"
        
        # Test with exclude_metadata=False
        feature_cols_with_metadata = engineer.get_feature_columns(
            sample_features_data, exclude_metadata=False
        )
        assert 'city' in feature_cols_with_metadata, "City should be included when exclude_metadata=False"
    
    def test_select_features_selectkbest(self, sample_X_y):
        """Test feature selection using SelectKBest."""
        engineer = FeatureEngineer()
        X, y = sample_X_y
        
        # Test feature selection
        k = 5
        selected_features = engineer.select_features_selectkbest(X, y, k=k)
        
        # Assertions
        assert isinstance(selected_features, list), "Should return list of feature names"
        assert len(selected_features) == k, f"Should select exactly {k} features"
        assert all(feat in X.columns for feat in selected_features), "All selected features should be in original columns"
        
        # Check that selector and features are stored
        assert engineer.feature_selector is not None, "Feature selector should be stored"
        assert engineer.selected_features == selected_features, "Selected features should be stored"
    
    def test_select_features_rfe(self, sample_X_y):
        """Test feature selection using RFE."""
        engineer = FeatureEngineer()
        X, y = sample_X_y
        
        # Test feature selection
        n_features = 4
        selected_features = engineer.select_features_rfe(X, y, rfe_features=n_features)
        
        # Assertions
        assert isinstance(selected_features, list), "Should return list of feature names"
        assert len(selected_features) == n_features, f"Should select exactly {n_features} features"
        assert all(feat in X.columns for feat in selected_features), "All selected features should be in original columns"
        
        # Check that selector and features are stored
        assert engineer.feature_selector is not None, "Feature selector should be stored"
        assert engineer.selected_features == selected_features, "Selected features should be stored"
    
    def test_extract_all_features_pipeline(self, sample_train_test_data):
        """Test complete feature extraction pipeline."""
        engineer = FeatureEngineer()
        train_data, test_data = sample_train_test_data
        
        # Extract all features
        train_features, test_features = engineer.extract_all_features(train_data, test_data)
        
        # Assertions
        assert isinstance(train_features, pd.DataFrame), "Should return DataFrame for training"
        assert isinstance(test_features, pd.DataFrame), "Should return DataFrame for test"
        assert train_features.shape[0] == train_data.shape[0], "Training row count should remain same"
        assert test_features.shape[0] == test_data.shape[0], "Test row count should remain same"
        
        # Check that features are added
        original_train_cols = set(train_data.columns)
        new_train_cols = set(train_features.columns)
        added_cols = new_train_cols - original_train_cols
        
        assert len(added_cols) > 0, "Should add new feature columns"
        
        # Check for expected feature types
        expected_features = ['year', 'month', 'day', 'location']
        for feat in expected_features:
            assert feat in train_features.columns, f"Feature '{feat}' should be present"
    
    def test_select_best_features_selectkbest_method(self, sample_X_y):
        """Test select_best_features with selectkbest method."""
        engineer = FeatureEngineer()
        X, y = sample_X_y
        
        # Create a DataFrame with target for the method
        test_data = X.copy()
        test_data[TARGET_COL] = y
        
        # Test selectkbest method
        n_features = min(6, len(X.columns))  # Ensure we don't ask for more features than available
        selected_features = engineer.select_best_features(
            test_data, method='selectkbest', n_features=n_features
        )
        
        # Assertions
        assert isinstance(selected_features, list), "Should return list of feature names"
        assert len(selected_features) == n_features, f"Should select exactly {n_features} features"
        assert engineer.selected_features == selected_features, "Selected features should be stored"
    
    def test_select_best_features_rfe_method(self, sample_X_y):
        """Test select_best_features with RFE method."""
        engineer = FeatureEngineer()
        X, y = sample_X_y
        
        # Create a DataFrame with target for the method
        test_data = X.copy()
        test_data[TARGET_COL] = y
        
        # Test RFE method
        n_features = min(5, len(X.columns))  # Ensure we don't ask for more features than available
        selected_features = engineer.select_best_features(
            test_data, method='rfe', n_features=n_features
        )
        
        # Assertions
        assert isinstance(selected_features, list), "Should return list of feature names"
        assert len(selected_features) == n_features, f"Should select exactly {n_features} features"
        assert engineer.selected_features == selected_features, "Selected features should be stored"
    
    def test_select_best_features_invalid_method(self, sample_features_data):
        """Test select_best_features with invalid method."""
        engineer = FeatureEngineer()
        
        # Test invalid method
        with pytest.raises(ValueError, match="Unknown selection method"):
            engineer.select_best_features(sample_features_data, method='invalid_method')
    
    def test_feature_engineering_data_integrity(self, base_sample_data):
        """Test that feature engineering preserves data integrity."""
        engineer = FeatureEngineer()
        
        # Extract temporal features
        df_temporal = engineer.extract_temporal_features(base_sample_data)
        
        # Check that original data is preserved
        original_cols = base_sample_data.columns
        for col in original_cols:
            if col in df_temporal.columns:
                pd.testing.assert_series_equal(
                    base_sample_data[col], df_temporal[col], 
                    check_names=False
                )
        
        # Check that IDs are preserved
        assert set(base_sample_data['id']) == set(df_temporal['id']), "IDs should be preserved"


def run_feature_engineer_tests():
    """
    Function to run all FeatureEngineer tests programmatically.
    """
    import pytest
    
    result = pytest.main([__file__, "-v", "--tb=short"])
    
    if result == 0:
        print("✅ All FeatureEngineer tests passed!")
        return True
    else:
        print("❌ Some FeatureEngineer tests failed!")
        return False


if __name__ == "__main__":
    run_feature_engineer_tests()
