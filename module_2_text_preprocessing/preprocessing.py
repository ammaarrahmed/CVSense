"""
Module 2: Text Preprocessing
Cleans and preprocesses resume and job description text for vectorization.
"""

import re
import string
import pandas as pd
from typing import Optional, Set

# Try to import NLTK components
NLTK_AVAILABLE = False
nltk = None
stopwords = None
WordNetLemmatizer = None
word_tokenize = None

try:
    import nltk as _nltk
    from nltk.corpus import stopwords as _stopwords
    from nltk.stem import WordNetLemmatizer as _WordNetLemmatizer
    from nltk.tokenize import word_tokenize as _word_tokenize
    
    nltk = _nltk
    stopwords = _stopwords
    WordNetLemmatizer = _WordNetLemmatizer
    word_tokenize = _word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    pass


def ensure_nltk_data():
    """Download required NLTK data if not present"""
    if not NLTK_AVAILABLE:
        return False
    
    packages = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']
    for package in packages:
        try:
            if 'punkt' in package:
                nltk.data.find(f'tokenizers/{package}')
            elif package == 'stopwords':
                nltk.data.find('corpora/stopwords')
            elif package == 'wordnet':
                nltk.data.find('corpora/wordnet')
            else:
                nltk.data.find(f'taggers/{package}')
        except LookupError:
            nltk.download(package, quiet=True)
    return True


def clean_text(text: str) -> str:
    """
    Clean raw text by removing noise while preserving meaningful content.
    
    Keeps:
    - Alphanumeric characters (preserves esp32, n8n, python3)
    - Spaces
    
    Removes:
    - Special characters and punctuation
    - Extra whitespace
    """
    text = str(text).lower()
    # Replace punctuation with space (preserves word boundaries)
    text = re.sub(r'[^\w\s]', ' ', text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def preprocess_text_simple(text: str) -> str:
    """
    Simple preprocessing without NLTK.
    Just tokenizes and filters short words.
    """
    tokens = text.split()
    # Keep words with length > 1 (preserves AI, ML, etc.)
    tokens = [word for word in tokens if len(word) > 1]
    return ' '.join(tokens)


def preprocess_text_nltk(text: str, lemmatizer, stop_words: set) -> str:
    """
    Full preprocessing with NLTK lemmatization and stopword removal.
    """
    tokens = word_tokenize(str(text))
    # Remove stopwords but keep technical terms
    tokens = [t for t in tokens if t.lower() not in stop_words and len(t) > 1]
    # Lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)


def preprocess_dataframe(df: pd.DataFrame, text_column: str, use_nltk: bool = True) -> pd.DataFrame:
    """
    Preprocess a DataFrame with text content.
    
    Args:
        df: DataFrame with text column
        text_column: Name of column containing text to preprocess
        use_nltk: Whether to use NLTK for advanced preprocessing
        
    Returns:
        DataFrame with 'cleaned_text' and 'preprocessed_text' columns added
    """
    df = df.copy()
    
    # Step 1: Clean text
    df['cleaned_text'] = df[text_column].apply(clean_text)
    
    # Step 2: Preprocess
    if use_nltk and NLTK_AVAILABLE:
        ensure_nltk_data()
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        df['preprocessed_text'] = df['cleaned_text'].apply(
            lambda x: preprocess_text_nltk(x, lemmatizer, stop_words)
        )
    else:
        df['preprocessed_text'] = df['cleaned_text'].apply(preprocess_text_simple)
    
    return df


def preprocess_resumes(resumes_df: pd.DataFrame, text_column: str = 'cleaned_resume') -> pd.DataFrame:
    """Preprocess resumes DataFrame"""
    return preprocess_dataframe(resumes_df, text_column)


def preprocess_jobs(jobs_df: pd.DataFrame, text_column: str = 'cleaned_description') -> pd.DataFrame:
    """Preprocess job descriptions DataFrame"""
    return preprocess_dataframe(jobs_df, text_column)


# For direct text preprocessing (used by app.py upload feature)
def preprocess_text(text: str, use_nltk: bool = False) -> str:
    """
    Preprocess a single text string.
    
    Args:
        text: Raw text to preprocess
        use_nltk: Whether to use NLTK (default False for simplicity)
        
    Returns:
        Preprocessed text string
    """
    cleaned = clean_text(text)
    
    if use_nltk and NLTK_AVAILABLE:
        ensure_nltk_data()
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))
        return preprocess_text_nltk(cleaned, lemmatizer, stop_words)
    else:
        return preprocess_text_simple(cleaned)


if __name__ == "__main__":
    # Test the module
    sample_text = "Senior Python Developer with 5+ years in ML/AI, FastAPI, and ESP32 systems!"
    print(f"Original: {sample_text}")
    print(f"Cleaned: {clean_text(sample_text)}")
    print(f"Preprocessed: {preprocess_text(sample_text)}")
