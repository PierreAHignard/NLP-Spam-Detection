"""
Feature engineering module for Air Quality ML Pipeline.

This module handles feature extraction and selection.
Students need to complete the sections.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.linear_model import LinearRegression

from utils.config import TARGET_COL
from utils.logger import get_logger, LogLevel

# Import MLflow (Workshop 4)

import mlflow

class FeatureEngineer:
    """
    Feature engineer for air quality prediction.
    
    Handles temporal feature extraction, geographic feature creation,
    categorical encoding, and feature selection.
    """
    
    def __init__(self):
        """Initialize the feature engineer."""
        self.feature_columns = None
        self.label_encoders = {}
        self.feature_selector = None
        self.selected_features = None
    
    def extract_temporal_features(self, df):
        """
        Extract temporal features from datetime column.
        
        Air quality varies with seasonal changes, weekly patterns,
        and daily cycles. These features help models understand when
        pollution levels are typically higher or lower.
        
        Args:
            df: DataFrame with date column
            
        Returns:
            DataFrame with additional temporal features
        """
        logger = get_logger()
        logger.substep("Extracting Temporal Features")
        
        date_col = "date"

        # Copy the DataFrame into df_features to avoid modifying the original
        df_features = df.copy()

        # Ensure the date column is datetime type for proper extraction
        df_features[date_col] = pd.to_datetime(df_features[date_col])

        # Extract basic temporal components that affect air quality
        df_features["year"] = df_features[date_col].dt.year
        df_features["month"] = df_features[date_col].dt.month
        df_features["day"] = df_features[date_col].dt.day
        df_features["quarter"] = df_features[date_col].dt.quarter
        df_features["week"] = df_features[date_col].dt.isocalendar().week
        df_features["dayofweek"] = df_features[date_col].dt.dayofweek
        
        # Logging
        logger.success("Temporal features extracted")

        return df_features
    
    def extract_geographic_features(self, df):
        """
        Extract geographic features from coordinate data.
        
        Different monitoring locations have unique pollution characteristics
        based on their surroundings. This creates location identifiers to
        help models learn location-specific patterns.
        
        Args:
            df: DataFrame with latitude and longitude columns
            
        Returns:
            DataFrame with additional geographic features
        """
        logger = get_logger()
        logger.substep("Extracting Geographic Features")
        
        # Copy the DataFrame into df_features to avoid modifying the original
        df_features = df.copy()

        # Create a unique location identifier by combining coordinates
        # This allows the model to learn location-specific patterns
        
        df_features['location'] = (
            df_features['site_latitude'].round(4).astype(str) + '_' + 
            df_features['site_longitude'].round(4).astype(str)
        )

        # Logging
        n_locations = df_features['location'].nunique()
        logger.success(f"Created location identifiers for {n_locations} unique locations")
        
        return df_features
    
    def encode_categorical_features(self, train_df, test_df, categorical_columns=None):
        """
        Encode categorical features for machine learning models.
        
        Machine learning algorithms require numerical inputs. This converts
        categorical variables to numerical format while ensuring consistency
        between training and test datasets.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame  
            categorical_columns: List of columns to encode (auto-detect if None)
            
        Returns:
            Tuple of (encoded_train_df, encoded_test_df)
        """
        logger = get_logger()
        logger.substep("Encoding Categorical Features")
        
        # Combine datasets to ensure consistent encoding across train/test
        # This prevents issues where test set has categories not seen in training
        train_df['_source'] = 'train'
        test_df['_source'] = 'test'
        df = pd.concat([train_df, test_df])
        # Initialize label encoder for consistent categorical-to-numerical conversion
        le = LabelEncoder()
        # Identify categorical columns that need encoding
        # These are high-cardinality categories (many unique values)
        to_encode = [ col for col in df.columns if df[col].nunique() > 2]
        print(" To Encode : ", to_encode)
        # Apply label encoding: convert each unique category to a unique integer
        for col in to_encode:
            df[col] = le.fit_transform(df[col])
        # Split back into separate train and test datasets
        train_encoded = df[df["_source"] == 'train']
        test_encoded = df[df["_source"] == 'test']
        # Logging
        logger.success("Categorical encoding completed")

        return train_encoded, test_encoded
    
    def select_features_selectkbest(self, X, y, k=15):
        """
        Select features using SelectKBest with f_regression.
        
        Args:
            X: Feature matrix
            y: Target variable
            k: Number of features to select
            
        Returns:
            List of selected feature names
        """
        logger = get_logger()
        logger.substep("Feature Selection - SelectKBest")
        
        # Initialize and fit the selector
        selector_kbest = SelectKBest(f_regression, k=k)
        selected_df = selector_kbest.fit_transform(X, y)
        # Get selected feature names and their scores
        selected_features_kbest = selector_kbest.get_feature_names_out()
        scores = selector_kbest.scores_
        # Create a summary DataFrame for selected features only
        summ = pd.DataFrame(selected_df, columns=selected_features_kbest)
        summ.describe()

        # Logging
        logger.info(f"Top {k} features selected by SelectKBest:", LogLevel.NORMAL)
        if logger.level >= LogLevel.NORMAL:
            print(selected_df)
        
        # Store selector and features
        self.feature_selector = selector_kbest
        self.selected_features = selected_features_kbest.tolist()
        
        return selected_features_kbest.tolist()
    
    def select_features_rfe(self, X, y, rfe_features=15):
        """
        Select features using Recursive Feature Elimination.
        
        Args:
            X: Feature matrix
            y: Target variable
            rfe_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        logger = get_logger()
        logger.substep("Feature Selection - RFE")
        
        # Initialize RFE with LinearRegression as the estimator
        selector_rfe = RFE(LinearRegression(), n_features_to_select=rfe_features)
        # Fit RFE and transform features
        selector_rfe.fit_transform(X, y)
        # Get selected features and their rankings
        selected_features_rfe = selector_rfe.get_feature_names_out()
        # Logging
        logger.info(f"Top {rfe_features} features selected by RFE:", LogLevel.NORMAL)
        if logger.level >= LogLevel.NORMAL:
            rfe_df = pd.DataFrame({
                'Feature': selected_features_rfe,
                'Coefficient': selector_rfe.estimator_.coef_
            }).sort_values('Coefficient', key=abs, ascending=False)
            print(rfe_df)
        
        # Store selector and features in instance variables
        self.feature_selector = selector_rfe
        self.selected_features = selected_features_rfe.tolist()
        
        return selected_features_rfe.tolist()
    
    def get_feature_columns(self, df, exclude_target=True, exclude_metadata=True):
        """
        Get list of columns that can be used as features.
        
        Args:
            df: DataFrame to analyze
            exclude_target: Whether to exclude target column
            exclude_metadata: Whether to exclude metadata columns
            
        Returns:
            List of feature column names
        """
        # Start with all columns
        feature_cols = df.columns.tolist()
        
        # Columns to exclude
        exclude_cols = []
        
        # Exclude target column if exclude_target is True and TARGET_COL is defined
        if exclude_target and TARGET_COL in feature_cols:
            exclude_cols.append(TARGET_COL)
        
        # Exclude metadata columns if exclude_metadata is True
        # Metadata columns are those that do not contribute to the model
        # but are useful for understanding the data context
        if exclude_metadata:
            # Common metadata columns to exclude (but keep date as it's encoded as feature)
            metadata_cols = ['id', 'folds', 'site_id', 'country', 'city', 'site_latitude', 'site_longitude','_source','location']
            exclude_cols.extend([col for col in metadata_cols if col in feature_cols])

        
        # Remove excluded columns
        feature_cols = [col for col in feature_cols if col not in exclude_cols]
        
        # Store features in instance variables
        self.feature_columns = feature_cols
        return feature_cols
    
    def extract_all_features(self, train_df, test_df):
        """
        Extract all features from train and test data.
        
        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            
        Returns:
            Tuple of (train_with_features, test_with_features)
        """
        logger = get_logger()
        logger.info("Extracting all features...")
        
        # Extract temporal features
        train_features = self.extract_temporal_features(train_df)
        test_features = self.extract_temporal_features(test_df)
        
        # Extract geographic features
        train_features = self.extract_geographic_features(train_features)
        test_features = self.extract_geographic_features(test_features)
        
        # Encode categorical features
        train_features, test_features = self.encode_categorical_features(
            train_features, test_features
        )
        
        # Logging
        logger.success("All features extracted")
        
        return train_features, test_features
    
    def select_best_features(self, train_df, method='selectkbest', n_features=None):
        """
        Select the best features using specified method.
        
        Args:
            train_df: Training DataFrame with features and target
            method: 'selectkbest' or 'rfe'
            n_features: Number of features to select
            
        Returns:
            List of selected feature names
        """
        # Get feature columns
        feature_cols = self.get_feature_columns(train_df)
        
        X = train_df[feature_cols]
        y = train_df[TARGET_COL]


        # Add MLflow feature selection logging (Workshop 4)
        if mlflow.active_run():
            # Log feature selection parameters
            mlflow.log_params({
                'method': method,
                'n_features': n_features,
                'feature_cols': feature_cols,
                'target_col': TARGET_COL,
            })
        
        if method == 'selectkbest':
            return self.select_features_selectkbest(X, y, n_features)
        elif method == 'rfe':
            return self.select_features_rfe(X, y, n_features)
        else:
            raise ValueError(f"Unknown selection method: {method}")
