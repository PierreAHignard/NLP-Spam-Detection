import argparse
from utils.config import MODEL_TYPES

def parse_arguments():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Run Air Quality ML Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--model', type=str, default='Logistic_Regression',
        choices=MODEL_TYPES,
        help='Model type to train'
    )

    parser.add_argument(
        '--optimize', action='store_true',
        help='Enable hyperparameter optimization using GridSearchCV'
    )

    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose output (deprecated, use --log-level verbose)'
    )

    parser.add_argument(
        '--log-level', type=str, default='normal',
        choices=['silent', 'normal', 'verbose'],
        help='Logging level: silent (no output), normal (main steps), verbose (all details)'
    )

    parser.add_argument(
        '--run_name', type=str, default='',
        help='Name to give to the run. Defaults to general parameters and date'
    )

    parser.add_argument(
        '--train_datasets',
        nargs='+',  # Allows 0, 1, or more datasets (e.g., 'SMS', 'EMAIL', 'SMS EMAIL')
        default=['SMS'],
        choices=['SMS', 'EMAIL'],
        help='Datasets to include in the training set. Options: SMS, EMAIL.'
    )

    parser.add_argument(
        '--test_datasets',
        nargs='+',  # Allows 0, 1, or more datasets (e.g., 'SMS', 'EMAIL', 'SMS EMAIL')
        default=['SMS'],
        choices=['SMS', 'EMAIL'],
        help='Datasets to include in the testing set. Options: SMS, EMAIL.'
    )

    parser.add_argument(
        '--stop_words', type=str, default=None,
        help='Stop word list to use for tokenisation'
    )

    parser.add_argument(
        '--vectorizer_type', type=str, default=None,
        choices=['count', 'tfidf'],
        help='Stop word list to use for tokenisation'
    )

    parser.add_argument(
        '--lowercase', action='store_true',
        help='Lowercase all tokens'
    )

    parser.add_argument(
        '--remove_punctuation', action='store_true',
        help='Remove punctuation from tokens'
    )

    parser.add_argument(
        '--number_placeholder', action='store_true',
        help='Replace all numbers by the <NUM> placeholder token'
    )

    return parser.parse_args()