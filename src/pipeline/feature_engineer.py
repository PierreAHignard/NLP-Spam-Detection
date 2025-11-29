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

from utils.config import TOKEN_REGEX
from utils.logger import get_logger


class FeatureEngineer:
    """
    Feature engineer for air quality prediction.

    Handles temporal feature extraction, geographic feature creation,
    categorical encoding, and feature selection.
    """

    def __init__(
            self,
            stop_words: str | None = None,
            number_placeholder: bool = False,
            lowercase: bool = False,
            remove_punctuation: bool = False,
            vectorizer_type: str = 'count',
            max_features: int = 5000
        ):
        """
        Initialize the feature engineer.

        Args:
            vectorizer_type (str): 'count' for CountVectorizer or 'tfidf' for TfidfVectorizer
        """

        self.vectorizer = None  # Renamed from count_vectorizer to be generic
        self.vectorizer_type = vectorizer_type # Store the type
        self.stop_words = stop_words
        self.number_placeholder = number_placeholder
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.max_features = max_features

        self._stop_words_list = None

        if self.stop_words == "nltk":
            nltk.download('stopwords')
            self._stop_words_list = list(stopwords.words('english'))

            # Sklearn tells me I should add these too
            self._stop_words_list += ['arent', 'couldnt', 'didnt', 'doesnt', 'dont', 'hadnt', 'hasnt', 'havent', 'hed', 'hell', 'hes', 'id', 'ill', 'im', 'isnt', 'itd', 'itll', 'ive', 'mightnt', 'mustnt', 'neednt', 'shant', 'shed', 'shell', 'shes', 'shouldnt', 'shouldve', 'thatll', 'theyd', 'theyll', 'theyre', 'theyve', 'wasnt', 'wed', 'well', 'werent', 'weve', 'wont', 'wouldnt', 'youd', 'youll', 'youre', 'youve']

        if self.remove_punctuation:
            # Create pattern (which keeps the <NUM> placeholder)
            punctuations = string.punctuation.replace('<', '').replace('>', '')
            self._remove_punctuation_pattern = r'[' + re.escape(punctuations) + r']'

    def preprocess(self, txt: str):
        """
        Replace all numbers in text with <NUM> token.

        Args:
            txt (str): Input text containing numbers

        Returns:
            str: Text with numbers replaced by <NUM>
        """

        if self.lowercase:
            txt = txt.lower()

        if self.number_placeholder:
            txt = re.sub(r'\d+', '<NUM>', txt)

        if self.remove_punctuation:
            txt = re.sub(self._remove_punctuation_pattern, '', txt)

        return txt

    def tokeniser(self, train_msg, test_msg):
        logger = get_logger()

        if self.vectorizer is None:

            # Define common arguments to avoid repetition
            vectorizer_args = {
                'input': 'content',
                'max_features': self.max_features,
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