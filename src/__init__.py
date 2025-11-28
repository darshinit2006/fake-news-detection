"""
Fake News Detection Package.
"""

from .preprocessing import (
    clean_text,
    remove_stopwords,
    stem_text,
    preprocess_text,
    preprocess_dataframe
)

from .train import FakeNewsClassifier, train_model
from .evaluate import evaluate_model, get_classification_report, print_evaluation

__all__ = [
    'clean_text',
    'remove_stopwords',
    'stem_text',
    'preprocess_text',
    'preprocess_dataframe',
    'FakeNewsClassifier',
    'train_model',
    'evaluate_model',
    'get_classification_report',
    'print_evaluation'
]
