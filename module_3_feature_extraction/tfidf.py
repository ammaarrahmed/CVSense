"""
Module 3: Feature Extraction (TF-IDF)
Converts preprocessed text into TF-IDF vectors for similarity computation.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from scipy.sparse import spmatrix

from sklearn.feature_extraction.text import TfidfVectorizer


class TFIDFVectorizer:
    """
    TF-IDF Vectorizer for resume and job description matching.
    
    Optimized for small corpus sizes (upload feature) and large corpus (batch processing).
    """
    
    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        use_idf: bool = True,
        stop_words: str = 'english'
    ):
        """
        Initialize the vectorizer.
        
        Args:
            max_features: Maximum number of features to extract
            ngram_range: Range of n-grams to consider (1,2) = unigrams and bigrams
            use_idf: Whether to use inverse document frequency weighting
            stop_words: Stop words to remove ('english' or None)
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=stop_words,
            use_idf=use_idf,
            norm='l2'
        )
        self.is_fitted = False
        
    def fit_transform(
        self,
        resume_texts: List[str],
        job_texts: List[str]
    ) -> Tuple[spmatrix, spmatrix]:
        """
        Fit vectorizer on all texts and transform to TF-IDF vectors.
        
        Args:
            resume_texts: List of preprocessed resume texts
            job_texts: List of preprocessed job description texts
            
        Returns:
            Tuple of (resume_vectors, job_vectors) as sparse matrices
        """
        # Combine all texts for fitting
        all_texts = resume_texts + job_texts
        
        # Fit and transform
        tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        self.is_fitted = True
        
        # Split back into resume and job vectors
        resume_vectors = tfidf_matrix[:len(resume_texts)]  # type: ignore
        job_vectors = tfidf_matrix[len(resume_texts):]  # type: ignore
        
        return resume_vectors, job_vectors
    
    def get_feature_names(self) -> np.ndarray:
        """Get the feature names (vocabulary) after fitting"""
        if not self.is_fitted:
            raise ValueError("Vectorizer not fitted. Call fit_transform first.")
        return self.vectorizer.get_feature_names_out()
    
    def save(self, filepath: str) -> None:
        """Save vectorizer and vectors to pickle file"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'is_fitted': self.is_fitted
            }, f)
    
    @classmethod
    def load(cls, filepath: str) -> 'TFIDFVectorizer':
        """Load vectorizer from pickle file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = cls()
        instance.vectorizer = data['vectorizer']
        instance.is_fitted = data['is_fitted']
        return instance


def create_tfidf_vectors(
    resume_texts: List[str],
    job_texts: List[str],
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    use_idf: bool = True
) -> Dict[str, Any]:
    """
    Create TF-IDF vectors for resumes and jobs.
    
    Args:
        resume_texts: List of preprocessed resume texts
        job_texts: List of preprocessed job description texts
        max_features: Maximum vocabulary size
        ngram_range: N-gram range for feature extraction
        use_idf: Whether to use IDF weighting (set False for small corpus)
        
    Returns:
        Dictionary with:
        - resume_vectors: Sparse matrix of resume TF-IDF vectors
        - jd_vectors: Sparse matrix of job TF-IDF vectors
        - feature_names: Array of feature names
        - vectorizer: Fitted TfidfVectorizer instance
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english',
        use_idf=use_idf,
        norm='l2'
    )
    
    # Combine and fit
    all_texts = resume_texts + job_texts
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Split
    resume_vectors = tfidf_matrix[:len(resume_texts)]  # type: ignore
    job_vectors = tfidf_matrix[len(resume_texts):]  # type: ignore
    
    return {
        'resume_vectors': resume_vectors,
        'jd_vectors': job_vectors,
        'feature_names': vectorizer.get_feature_names_out(),
        'vectorizer': vectorizer
    }


def save_tfidf_vectors(
    output_path: str,
    resume_vectors: spmatrix,
    job_vectors: spmatrix,
    feature_names: np.ndarray
) -> None:
    """
    Save TF-IDF vectors to pickle file.
    
    Args:
        output_path: Path to save pickle file
        resume_vectors: Resume TF-IDF vectors
        job_vectors: Job description TF-IDF vectors
        feature_names: Vocabulary feature names
    """
    output = {
        'resume_vectors': resume_vectors,
        'jd_vectors': job_vectors,
        'feature_names': feature_names
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(output, f)


def load_tfidf_vectors(filepath: str) -> Dict[str, Any]:
    """
    Load TF-IDF vectors from pickle file.
    
    Args:
        filepath: Path to pickle file
        
    Returns:
        Dictionary with resume_vectors, jd_vectors, feature_names
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    # Test the module
    sample_resumes = [
        "python developer machine learning tensorflow keras",
        "data scientist nlp deep learning pytorch",
        "software engineer java spring backend"
    ]
    sample_jobs = [
        "machine learning engineer python tensorflow required",
        "backend developer java spring boot experience"
    ]
    
    result = create_tfidf_vectors(sample_resumes, sample_jobs)
    print(f"Resume vectors shape: {result['resume_vectors'].shape}")
    print(f"Job vectors shape: {result['jd_vectors'].shape}")
    print(f"Feature count: {len(result['feature_names'])}")
    print(f"Sample features: {result['feature_names'][:10]}")
