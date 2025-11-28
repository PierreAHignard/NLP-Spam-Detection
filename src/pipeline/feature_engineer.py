"""
Feature engineering module for Air Quality ML Pipeline.

This module handles feature extraction and selection.
Students need to complete the sections.
"""

import re

from sklearn.feature_extraction.text import CountVectorizer

from utils.config import NUMBER_PLACEHOLDER, NB_FEATURES, TOKEN_REGEX
import mlflow

from utils.logger import get_logger


def preprocess(txt: str):
    """
    Replace all numbers in text with <NUM> token.

    Args:
        txt (str): Input text containing numbers

    Returns:
        str: Text with numbers replaced by <NUM>
    """

    return re.sub(r'\d+', NUMBER_PLACEHOLDER, txt)

class FeatureEngineer:
    """
    Feature engineer for air quality prediction.
    
    Handles temporal feature extraction, geographic feature creation,
    categorical encoding, and feature selection.
    """
    
    def __init__(self, stop_words=None):
        """Initialize the feature engineer."""
        self.count_vectorizer = None
        self.stop_words = stop_words

    def tokeniser(self, train_msg, test_msg):
        logger = get_logger()

        if self.count_vectorizer is None:
            self.count_vectorizer = CountVectorizer(
                input='content',
                max_features=NB_FEATURES,
                token_pattern=TOKEN_REGEX,
                preprocessor=preprocess,
                stop_words=self.stop_words
            )

            self.count_vectorizer.fit(train_msg)

            logger.info(f"Count vectorizer fitted, with {len(self.count_vectorizer.vocabulary_)} features.")

        return self.count_vectorizer.transform(train_msg), self.count_vectorizer.transform(test_msg)