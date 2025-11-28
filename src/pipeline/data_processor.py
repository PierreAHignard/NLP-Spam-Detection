"""
Data preprocessing module for Air Quality ML Pipeline.

This module handles data loading, cleaning, and preprocessing.
Handles time-series data with geographical groupings properly.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.config import DATA_PATH, SMS_FILE, EMAIL_FILE, RANDOM_STATE, LABEL_COL, MESSAGE_COL, TRAIN_TEST_SPLIT_SIZE

from utils.logger import get_logger
import mlflow
from mlflow.data import from_pandas


def balance_data(data):
    """
    Balance training data by oversampling the minority class.

    Addresses class imbalance by randomly sampling additional instances
    from the underrepresented class until both classes have equal frequency.
    This prevents model bias toward the majority class.

    Parameters
    ----------
    data: pandas.Dataframe containing
        Training text messages
        Corresponding class labels (0 for ham, 1 for spam)

    Returns
    -------
    data:
        Dataframe containing the input data with equal
        class representation

    Notes
    -----
    Uses random sampling with replacement to increase minority class size.
    Preserves original data distribution while achieving balance.
    """
    logger = get_logger()

    # Working with a copy of the input data
    data = data.copy()

    counts = data[LABEL_COL].value_counts()

    logger.info("Label counts before balancing:\n" + str(counts))

    if counts[1] > counts[0]:
        label_to_oversample = 0
        diff = counts[1] - counts[0]
    else:
        label_to_oversample = 1
        diff = counts[0] - counts[1]

    draw_from = data[data[LABEL_COL] == label_to_oversample]

    for i in range(diff):
        sample = draw_from.sample(random_state=RANDOM_STATE)
        data = pd.concat([data, sample], ignore_index=True)

    logger.info("Label counts after balancing:\n" + str(data[LABEL_COL].value_counts()))

    return data


class DataProcessor:
    """
    Data processor for air quality datasets.
    
    Handles loading, cleaning, and preprocessing of air quality data
    with special attention to temporal and geographic characteristics.
    """
    
    def __init__(self, train_selection, test_selection):
        """Initialize the data processor."""
        self.sms_data = None
        self.email_data = None
        self.train_selection = train_selection
        self.test_selection = test_selection
    
    def load_data(self):
        """
        Load training and test datasets from CSV files.
        
        Returns:
            Tuple of (train_df, test_df)
        """
        logger = get_logger()
        logger.substep("Loading Data")
        
        # Load training and test data using DATA_PATH and file names defined in config
        self.sms_data = pd.read_csv(DATA_PATH / SMS_FILE)
        self.email_data = pd.read_csv(DATA_PATH / EMAIL_FILE)
        
        # Logging
        with logger.indent():
            logger.dataframe_info(self.sms_data, "SMS data")
            logger.dataframe_info(self.email_data, "Email data")

        if mlflow.active_run():
            mlflow.log_input(from_pandas(self.sms_data, "SMS Data"), "Training")
            mlflow.log_input(from_pandas(self.email_data, "Email Data"), "Training")

        logger.success("Data loading completed")
        return self.sms_data.copy(), self.email_data.copy()

    def preprocess_data(self, drop_duplicates=True, balance=True):
        """
        Complete preprocessing pipeline.
        
        Args:
            train_selection: List containing "SMS" and/or "EMAIL" for train data
            test_selection: List containing "SMS" and/or "EMAIL" for test data
            balance: Whether to balance unequal classes
            drop_duplicates: Whether to drop duplicates before balancing
            
        Returns:
            Tuple of (train_msg, train_lab, test_msg, test_lab)
        """
        if self.sms_data is None or self.email_data is None:
            raise ValueError("Data must be loaded first. Call load_data()")
        
        logger = get_logger()
        logger.substep("Starting preprocessing pipeline...")
        
        # Work with copies
        sms_data = self.sms_data.copy()
        email_data = self.email_data.copy()

        # Step 1: Drop duplicates
        if drop_duplicates:
            sms_data.drop_duplicates(inplace=True, ignore_index=True)
            email_data.drop_duplicates(inplace=True, ignore_index=True)

        # Step 2: Create train/test datasets
        train_data = []
        test_data = []

            # SMS distribution
        if "SMS" in self.train_selection and "SMS" not in self.test_selection:
            train_data.append(sms_data)
        if "SMS" not in self.train_selection and "SMS" in self.test_selection:
            test_data.append(sms_data)
        if "SMS" in self.train_selection and "SMS" in self.test_selection:
            temp_test, temp_train = train_test_split(sms_data,
                test_size=TRAIN_TEST_SPLIT_SIZE,
                random_state=RANDOM_STATE,
                stratify=sms_data[LABEL_COL])

            test_data.append(temp_test)
            train_data.append(temp_train)

            # Email distribution
        if "EMAIL" in self.train_selection and "EMAIL" not in self.test_selection:
            train_data.append(email_data)
        if "EMAIL" not in self.train_selection and "EMAIL" in self.test_selection:
            test_data.append(email_data)
        if "EMAIL" in self.train_selection and "EMAIL" in self.test_selection:
            temp_test, temp_train = train_test_split(email_data,
                test_size=TRAIN_TEST_SPLIT_SIZE,
                random_state=RANDOM_STATE,
                stratify=email_data[LABEL_COL])

            test_data.append(temp_test)
            train_data.append(temp_train)

            # If only one df, it will simply be returned as is
        train_data = pd.concat(train_data, axis=0, ignore_index=True)
        test_data = pd.concat(test_data, axis=0, ignore_index=True)

        # Step 3: Balance the label proportions in train dataset
        if balance:
            train_data = balance_data(train_data)

        # Step 4: Separate between message and label
        train_msg = train_data[MESSAGE_COL]
        train_lab = train_data[LABEL_COL]
        test_msg = test_data[MESSAGE_COL]
        test_lab = test_data[LABEL_COL]

        # Logging
        logger.success("Preprocessing pipeline completed")
        
        return train_msg, train_lab, test_msg, test_lab
    
    def load_and_preprocess(self, **preprocessing_kwargs):
        """
        Convenience method to load and preprocess data in one step.
        
        Args:
            **preprocessing_kwargs: Arguments for preprocess_data()
            
        Returns:
            Tuple of (train_msg, train_lab, test_msg, test_lab)
        """
        self.load_data()
        return self.preprocess_data(**preprocessing_kwargs)

