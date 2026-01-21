"""Module 4: Similarity & Ranking"""
from .ranking import (
    keyword_match_score,
    compute_tfidf_similarity,
    compute_hybrid_scores,
    rank_resumes,
    rank_resumes_for_job,
    extract_keywords,
    extract_phrases,
    STOP_WORDS,
    TECH_PHRASES
)

__all__ = [
    'keyword_match_score',
    'compute_tfidf_similarity',
    'compute_hybrid_scores',
    'rank_resumes',
    'rank_resumes_for_job',
    'extract_keywords',
    'extract_phrases',
    'STOP_WORDS',
    'TECH_PHRASES'
]
