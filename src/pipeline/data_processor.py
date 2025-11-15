"""
Data preprocessing module for Air Quality ML Pipeline.

This module handles data loading, cleaning, and preprocessing.
Handles time-series data with geographical groupings properly.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold

from utils.config import (
    DATA_PATH, TRAIN_FILE, TEST_FILE, MISSING_THRESHOLD, 
    N_SPLITS, TARGET_COL, CITY_COL, DATE_COL
)
from utils.logger import get_logger


class DataProcessor:
    """
    Data processor for air quality datasets.
    
    Handles loading, cleaning, and preprocessing of air quality data
    with special attention to temporal and geographic characteristics.
    """
    
    def __init__(self):
        """Initialize the data processor."""
        self.train_data = None
        self.test_data = None
    
    def load_data(self):
        """
        Load training and test datasets from CSV files.
        
        Returns:
            Tuple of (train_df, test_df)
        """
        logger = get_logger()
        logger.substep("Loading Data")
        
        # TODO Load training and test data using DATA_PATH and file names defined in config
        self.train_data = pd.read_csv(DATA_PATH / TRAIN_FILE)
        self.test_data = pd.read_csv(DATA_PATH / TEST_FILE)
        
        # Logging
        with logger.indent():
            logger.dataframe_info(self.train_data, "Training data")
            logger.dataframe_info(self.test_data, "Test data")
        
        logger.success("Data loading completed")
        return self.train_data.copy(), self.test_data.copy()
    
    def forward_back_fill(self, df, cols, group_col, date_col):
        """
        Fill missing values using forward and backward fill within each group.
        
        For time-series data, uses temporal relationships within geographic groups.

        Parameters:
        - df: DataFrame to process
        - cols: List of columns to fill
        - group_col: Column to group by (e.g., 'city')
        - date_col: Date column for sorting

        Returns:
        - DataFrame with missing values filled
        """
        logger = get_logger()
        logger.info("Forward-Backward Fill")
        
        # TODO Sort by group and date to ensure proper temporal order
        df = df.sort_values(by=[group_col, date_col]).reset_index(drop=True)

        # TODO Apply forward fill then backward fill for each col within each group

        for col in cols:
            df[col] = df.groupby(group_col)[col].ffill()
            df[col] = df.groupby(group_col)[col].bfill()

        return df
    
    def handle_missing_values(self, train_df, test_df):
        """
        Handle missing values for time-series data with geographic groupings.
        
        Process both datasets city by city using forward-backward fill
        within each geographic group.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Returns:
            Tuple of (train_processed, test_processed)
        """
        logger = get_logger()
        logger.substep("Handling Missing Values")
        
        # TODO Check initial missing values
        initial_missing_train = (train_df.isna().sum())
        initial_missing_test = (test_df.isna().sum())

        # Logging
        with logger.indent():
            logger.info(f"Initial missing values - Train: {initial_missing_train}, Test: {initial_missing_test}")
        
        # TODO Make copies to avoid modifying originals
        train = train_df.copy()
        test = test_df.copy()


        # TODO Process both datasets city by city
            # Process training data for this city
        train = self.forward_back_fill(train, train.columns, "city", "date")
            # Process test data for this city
        test = self.forward_back_fill(test, test.columns, "city", "date")
        
        # TODO Check remaining missing values
        final_missing_train = train.isnull().sum().sum()
        final_missing_test = test.isnull().sum().sum()
        
        # Logging
        with logger.indent():
            logger.info(f"Remaining missing values - Train: {final_missing_train}, Test: {final_missing_test}")
        
        if final_missing_train > 0 or final_missing_test > 0:
            logger.warning("Some missing values remain after imputation")
        else:
            logger.success("All missing values successfully handled")
        
        return train, test
    
    def drop_high_missing_columns(self, train_df, test_df):
        """
        Drop columns with more than MISSING_THRESHOLD proportion of missing data.
        
        Analyzes missing data patterns in training set and applies the same
        column drops to both training and test sets.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Returns:
            Tuple of (train_cleaned, test_cleaned)
        """
        logger = get_logger()
        logger.substep("Dropping High Missing Columns")
        threshold = 0.7
        
        # TODO Find columns to drop based on training data

        train_missing_data = train_df.isnull().sum()
        test_missing_data = test_df.isnull().sum()

        train_missing_percent = (train_missing_data/train_df.shape[0]) * 100
        test_missing_percent = (test_missing_data/test_df.shape[0]) * 100

        train_drop_cols = train_missing_percent[train_missing_percent > 70].index.tolist()
        test_drop_cols = test_missing_percent[test_missing_percent > 70].index.tolist()

        drop_cols = set(train_drop_cols + test_drop_cols)

        if len(drop_cols) > 0:
            # Logging
            with logger.indent():
                logger.info(f"Dropping {len(drop_cols)} columns with >{threshold*100}% missing data:")
                for col in drop_cols:
                    missing_pct = (train_df[col].isnull().sum() / len(train_df)) * 100
                    logger.info(f"  - {col}: {missing_pct:.1f}% missing")

            # TODO Drop from both datasets
            train = train_df.drop(drop_cols, axis=1)
            test = test_df.drop(drop_cols, axis=1)
            
        else:
            # TODO Copy original data if no columns to drop in order to maintain consistency with drop logic
            train = train_df.copy()
            test = test_df.copy()
            # Logging
            logger.success("No columns exceed missing data threshold")

        with logger.indent():
            logger.data_info(f"Remaining columns in train: {train.shape[1]}")
            logger.data_info(f"Remaining columns in test: {test.shape[1]}")
        
        return train, test
    
    def create_geographic_folds(self, df):
        """
        Create cross-validation folds based on geographic grouping.
        
        Uses GroupKFold to ensure entire cities are in training OR validation,
        never both, preventing data leakage.
        
        Args:
            df: DataFrame with city column
            
        Returns:
            DataFrame with 'folds' column added
        """
        logger = get_logger()
        logger.substep("Creating Geographic Folds")
        
        if CITY_COL not in df.columns:
            raise ValueError(f"City column '{CITY_COL}' not found in data")
        
        # TODO Copy the DataFrame to avoid modifying the original
        df_with_folds = df.copy()
        # TODO Create city-based folds using GroupKFold and N_SPLITS configured in utils/config.py
        gkf = GroupKFold(N_SPLITS)
        X = df_with_folds.drop(columns=["pm2_5", "city"])
        y = df_with_folds["pm2_5"]
        groups = df_with_folds["city"]


        # The 'groups' parameter tells GroupKFold which samples belong to the same group defined in CITY_COL
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
            df_with_folds.loc[val_idx, 'folds'] = fold

        # Convert to integer type for easier handling
        df_with_folds['folds'] = df_with_folds['folds'].astype(int)
        
        # Logging
        with logger.indent():
            logger.info("Fold distribution by city:")
            fold_dist = df_with_folds.groupby(['folds', CITY_COL]).size().reset_index(name='count')
            for fold in sorted(df_with_folds['folds'].unique()):
                cities = fold_dist[fold_dist['folds'] == fold][CITY_COL].tolist()
                total_samples = fold_dist[fold_dist['folds'] == fold]['count'].sum()
                logger.info(f"  Fold {fold}: {cities} ({total_samples} samples)")
        
        logger.success("Geographic folds created")
        return df_with_folds
    
    def preprocess_data(self, handle_missing=True, drop_high_missing=True, create_folds=True):
        """
        Complete preprocessing pipeline for air quality data.
        
        Args:
            handle_missing: Whether to handle missing values
            drop_high_missing: Whether to drop high missing columns
            create_folds: Whether to create CV folds (training data only)
            
        Returns:
            Tuple of (processed_train_df, processed_test_df)
        """
        if self.train_data is None or self.test_data is None:
            raise ValueError("Data must be loaded first. Call load_data()")
        
        logger = get_logger()
        logger.substep("Starting preprocessing pipeline...")
        
        # Work with copies
        train_processed = self.train_data.copy()
        test_processed = self.test_data.copy()

        # Step 1: Drop high missing columns
        if drop_high_missing:
            train_processed, test_processed = self.drop_high_missing_columns(train_processed, test_processed)
        
        # Step 2: Handle missing values
        if handle_missing:
            train_processed, test_processed = self.handle_missing_values(train_processed, test_processed)
        
        # Step 3: Create folds (training data only)
        if create_folds:
            train_processed = self.create_geographic_folds(train_processed)

        # Logging
        logger.success("Preprocessing pipeline completed")
        
        return train_processed, test_processed
    
    def load_and_preprocess(self, **preprocessing_kwargs):
        """
        Convenience method to load and preprocess data in one step.
        
        Args:
            **preprocessing_kwargs: Arguments for preprocess_data()
            
        Returns:
            Tuple of (processed_train_df, processed_test_df)
        """
        self.load_data()
        return self.preprocess_data(**preprocessing_kwargs)

