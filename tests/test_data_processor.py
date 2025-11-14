"""
Technical validation tests for DataProcessor class.

These tests validate the technical implementation without focusing on ML performance.
They check that methods execute correctly and produce expected data structures.
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path
import tempfile
import shutil

# Assuming the package structure from the provided code
from pipeline.data_processor import DataProcessor
from utils.config import TARGET_COL, CITY_COL, DATE_COL, N_SPLITS


class TestDataProcessor:
    """Test suite for DataProcessor technical validation."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Create realistic sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        cities = ['CityA', 'CityB', 'CityC', 'CityD']
        
        data = []
        for i, date in enumerate(dates):
            for city in cities:
                data.append({
                    'id': f'id_{len(data):06d}_{city.lower()}',  # Format string réaliste
                    'date': date,
                    'city': city,
                    'pm2_5': np.random.normal(25, 10),
                    'feature1': np.random.normal(0, 1),
                    'feature2': np.random.normal(10, 5),
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
    def processor_with_data(self, sample_data, tmp_path, monkeypatch):
        """Create DataProcessor with sample data files."""
        
        # Create temporary data files
        train_data = sample_data.iloc[:300].copy()
        test_data = sample_data.iloc[300:].copy()
        
        # Remove target from test data (realistic scenario)
        test_data = test_data.drop(columns=[TARGET_COL])
        
        # Save to temporary files
        train_path = tmp_path / "train.csv"
        test_path = tmp_path / "test.csv"
        
        train_data.to_csv(train_path, index=False)
        test_data.to_csv(test_path, index=False)
        
        # Patch the config module properly using monkeypatch
        monkeypatch.setattr('pipeline.data_processor.DATA_PATH', tmp_path)
        monkeypatch.setattr('pipeline.data_processor.TRAIN_FILE', "train.csv")
        monkeypatch.setattr('pipeline.data_processor.TEST_FILE', "test.csv")
        
        # Create processor after patching
        processor = DataProcessor()
        
        return processor, train_data, test_data

    def test_load_data_success(self, processor_with_data):
        """Test that data loading works correctly."""
        processor, expected_train, expected_test = processor_with_data
        
        # Test loading
        train_df, test_df = processor.load_data()
        
        # Assertions
        assert isinstance(train_df, pd.DataFrame), "Should return DataFrame for training data"
        assert isinstance(test_df, pd.DataFrame), "Should return DataFrame for test data"
        assert len(train_df) > 0, "Training data should not be empty"
        assert len(test_df) > 0, "Test data should not be empty"
        assert train_df.shape[0] == expected_train.shape[0], "Training data row count should match"
        assert TARGET_COL in train_df.columns, f"Target column '{TARGET_COL}' should be in training data"
        assert TARGET_COL not in test_df.columns, f"Target column '{TARGET_COL}' should not be in test data"

    def test_forward_back_fill_functionality(self, sample_data):
        """Test forward-backward fill method."""
        processor = DataProcessor()
        
        # Create data with missing values in specific pattern
        test_data = sample_data.copy()
        
        # Get columns with missing values
        cols_with_missing = test_data.columns[test_data.isnull().any()].tolist()
        
        # Apply forward-backward fill
        filled_data = processor.forward_back_fill(
            test_data, 
            cols_with_missing, 
            group_col=CITY_COL, 
            date_col=DATE_COL
        )
        
        # Assertions
        assert isinstance(filled_data, pd.DataFrame), "Should return DataFrame"
        assert filled_data.shape == test_data.shape, "Shape should remain unchanged"
        
        # Check that missing values are reduced (but not necessarily all filled)
        for col in cols_with_missing:
            original_missing = test_data[col].isnull().sum()
            filled_missing = filled_data[col].isnull().sum()
            assert filled_missing <= original_missing, f"Missing values in {col} should not increase"

    def test_handle_missing_values_execution(self, sample_data):
        """Test that handle_missing_values executes without errors."""
        processor = DataProcessor()
        
        train_df = sample_data.copy()
        test_df = sample_data.copy()
        
        # Execute the method
        train_processed, test_processed = processor.handle_missing_values(train_df, test_df)
        
        # Basic assertions
        assert isinstance(train_processed, pd.DataFrame), "Should return DataFrame for training"
        assert isinstance(test_processed, pd.DataFrame), "Should return DataFrame for test"
        assert train_processed.shape[1] == train_df.shape[1], "Number of columns should not change"
        assert test_processed.shape[1] == test_df.shape[1], "Number of columns should not change"
        
        # Check that method doesn't increase missing values
        for col in train_df.columns:
            if col in train_processed.columns:
                original_missing_train = train_df[col].isnull().sum()
                processed_missing_train = train_processed[col].isnull().sum()
                assert processed_missing_train <= original_missing_train, f"Training: {col} missing values should not increase"

    def test_drop_high_missing_columns_logic(self, sample_data):
        """Test column dropping logic."""
        processor = DataProcessor()
        
        # Create data with high missing percentage column
        train_df = sample_data.copy()
        test_df = sample_data.copy()
        
        # Add a column with high missing percentage
        high_missing_col = 'high_missing_feature'
        train_df[high_missing_col] = np.nan
        test_df[high_missing_col] = np.nan
        
        # Fill only small percentage to exceed threshold
        fill_count = int(len(train_df) * 0.2)  # Fill only 20%, leaving 80% missing
        train_df.loc[:fill_count, high_missing_col] = 1.0
        test_df.loc[:fill_count, high_missing_col] = 1.0
        
        original_cols_train = set(train_df.columns)
        original_cols_test = set(test_df.columns)
        
        # Execute method
        train_cleaned, test_cleaned = processor.drop_high_missing_columns(train_df, test_df)
        
        # Assertions
        assert isinstance(train_cleaned, pd.DataFrame), "Should return DataFrame for training"
        assert isinstance(test_cleaned, pd.DataFrame), "Should return DataFrame for test"
        assert high_missing_col not in train_cleaned.columns, "High missing column should be dropped from training"
        assert high_missing_col not in test_cleaned.columns, "High missing column should be dropped from test"
        assert len(train_cleaned.columns) <= len(original_cols_train), "Column count should not increase"

    def test_create_geographic_folds_structure(self, sample_data):
        """Test geographic folds creation."""
        processor = DataProcessor()
        
        # Execute method
        df_with_folds = processor.create_geographic_folds(sample_data)
        
        # Assertions
        assert isinstance(df_with_folds, pd.DataFrame), "Should return DataFrame"
        assert 'folds' in df_with_folds.columns, "Should add 'folds' column"
        assert df_with_folds['folds'].dtype == int, "Folds should be integer type"
        
        # Check fold values
        unique_folds = sorted(df_with_folds['folds'].unique())
        expected_folds = list(range(1, N_SPLITS + 1))
        assert unique_folds == expected_folds, f"Should have folds {expected_folds}"
        
        # Check that entire cities are in single folds (no city split across folds)
        city_fold_counts = df_with_folds.groupby(CITY_COL)['folds'].nunique()
        assert all(count == 1 for count in city_fold_counts), "Each city should be in exactly one fold"

    def test_preprocess_data_pipeline_execution(self, processor_with_data):
        """Test complete preprocessing pipeline execution."""
        processor, train_data, test_data = processor_with_data
        
        # Load data first
        processor.load_data()
        
        # Execute preprocessing pipeline
        train_processed, test_processed = processor.preprocess_data(
            handle_missing=True,
            drop_high_missing=True,
            create_folds=True
        )
        
        # Assertions
        assert isinstance(train_processed, pd.DataFrame), "Should return DataFrame for training"
        assert isinstance(test_processed, pd.DataFrame), "Should return DataFrame for test"
        assert 'folds' in train_processed.columns, "Training data should have folds column"
        assert 'folds' not in test_processed.columns, "Test data should not have folds column"
        assert len(train_processed) > 0, "Training data should not be empty"
        assert len(test_processed) > 0, "Test data should not be empty"

    def test_load_and_preprocess_convenience_method(self, processor_with_data):
        """Test the convenience method that combines loading and preprocessing."""
        processor, _, _ = processor_with_data
        
        # Execute convenience method
        train_processed, test_processed = processor.load_and_preprocess()
        
        # Assertions
        assert isinstance(train_processed, pd.DataFrame), "Should return DataFrame for training"
        assert isinstance(test_processed, pd.DataFrame), "Should return DataFrame for test"
        assert 'folds' in train_processed.columns, "Should include folds by default"

    def test_error_handling_no_data_loaded(self):
        """Test error handling when trying to preprocess without loading data."""
        processor = DataProcessor()
        
        # Should raise ValueError when no data is loaded
        with pytest.raises(ValueError, match="Data must be loaded first"):
            processor.preprocess_data()

    def test_data_integrity_after_processing(self, processor_with_data):
        """Test that data integrity is maintained during processing."""
        processor, original_train, original_test = processor_with_data
        
        # Load and process
        train_processed, test_processed = processor.load_and_preprocess()
        
        # Check ID integrity (assuming 'id' column exists)
        if 'id' in train_processed.columns and 'id' in original_train.columns:
            assert set(train_processed['id']) == set(original_train['id']), "Training IDs should be preserved"
        
        # Check that date and city columns are preserved
        assert DATE_COL in train_processed.columns, "Date column should be preserved"
        assert CITY_COL in train_processed.columns, "City column should be preserved"


def run_dataprocessor_tests():
    """
    Function to run all DataProcessor tests programmatically.
    Useful for quick validation during development.
    """
    import pytest
    
    # Run tests with verbose output
    result = pytest.main([__file__, "-v", "--tb=short"])
    
    if result == 0:
        print("✅ All DataProcessor tests passed!")
        return True
    else:
        print("❌ Some DataProcessor tests failed!")
        return False


if __name__ == "__main__":
    # Allow running tests directly
    run_dataprocessor_tests()
