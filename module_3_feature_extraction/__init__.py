"""Module 3: Feature Extraction (TF-IDF)"""
from .tfidf import (
    TFIDFVectorizer,
    create_tfidf_vectors,
    save_tfidf_vectors,
    load_tfidf_vectors
)

__all__ = [
    'TFIDFVectorizer',
    'create_tfidf_vectors',
    'save_tfidf_vectors',
    'load_tfidf_vectors'
]
