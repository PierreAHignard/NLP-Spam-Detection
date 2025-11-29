"""
Feature engineering module for Air Quality ML Pipeline.

This module handles feature extraction and selection.
Students need to complete the sections.
"""

import re
import string
import nltk
import mlflow
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from utils.config import NB_FEATURES, TOKEN_REGEX
from utils.logger import get_logger


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

        if self.vectorizer is None:

            # Define common arguments to avoid repetition
            vectorizer_args = {
                'input': 'content',
                'max_features': NB_FEATURES,
                'token_pattern': TOKEN_REGEX,
                'preprocessor': self.preprocess,
                'stop_words': self._stop_words_list
            }

            # Initialize the specific vectorizer based on the init parameter
            if self.vectorizer_type == 'tfidf':
                self.vectorizer = TfidfVectorizer(**vectorizer_args)
            else:
                self.vectorizer = CountVectorizer(**vectorizer_args)

            self.vectorizer.fit(train_msg)

            if mlflow.active_run():
                # Dynamically log the class name (CountVectorizer or TfidfVectorizer)
                vec_name = self.vectorizer.__class__.__name__
                mlflow.log_params({
                    f'{vec_name}.vocabulary_size': len(self.vectorizer.vocabulary_),
                    f'{vec_name}.token_pattern': TOKEN_REGEX,
                    f'{vec_name}.stop_words': self.stop_words,
                    f'{vec_name}.lowercase': self.lowercase,
                    f'{vec_name}.number_placeholder': self.number_placeholder,
                    f'{vec_name}.remove_punctuation': self.remove_punctuation,
                    'vectorizer_type': self.vectorizer_type
                })

            logger.info(f"{self.vectorizer.__class__.__name__} fitted, with {len(self.vectorizer.vocabulary_)} features.")

        return self.vectorizer.transform(train_msg), self.vectorizer.transform(test_msg)