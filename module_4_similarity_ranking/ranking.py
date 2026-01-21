"""
Module 4: Similarity & Ranking
Computes similarity between resumes and job descriptions, ranks candidates.

Implements hybrid scoring combining:
1. Keyword matching (like Jobscan)
2. TF-IDF cosine similarity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Set
from scipy.sparse import spmatrix
from sklearn.metrics.pairwise import cosine_similarity


# Common words to exclude from keyword matching
COMMON_WORDS = {
    'the', 'a', 'an', 'and', 'or', 'is', 'are', 'was', 'were', 'be', 'been',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used',
    'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'between',
    'this', 'that', 'these', 'those', 'it', 'its', 'we', 'our', 'you', 'your',
    'they', 'their', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there', 'when',
    'where', 'why', 'how', 'what', 'which', 'who', 'whom', 'if', 'then', 'else',
    'any', 'about', 'over', 'under', 'again', 'further', 'once', 'including',
    'able', 'across', 'almost', 'along', 'already', 'although', 'always',
    'among', 'another', 'around', 'away', 'back', 'become', 'becomes', 'being',
    'best', 'better', 'big', 'come', 'comes', 'coming', 'consider', 'considered',
    'could', 'day', 'days', 'different', 'done', 'down', 'either', 'end', 'enough',
    'even', 'ever', 'example', 'experience', 'find', 'first', 'full', 'get',
    'give', 'given', 'go', 'going', 'good', 'got', 'great', 'help', 'high',
    'however', 'keep', 'know', 'last', 'least', 'less', 'let', 'like', 'likely',
    'long', 'look', 'looking', 'made', 'make', 'makes', 'making', 'many', 'much',
    'must', 'need', 'needed', 'needs', 'never', 'new', 'next', 'number', 'often',
    'old', 'one', 'open', 'order', 'part', 'place', 'point', 'possible', 'put',
    'rather', 'really', 'right', 'said', 'say', 'see', 'seem', 'seems', 'set',
    'show', 'shown', 'since', 'small', 'something', 'still', 'sure', 'take',
    'taken', 'tell', 'thing', 'things', 'think', 'three', 'time', 'today',
    'together', 'top', 'toward', 'try', 'trying', 'turn', 'two', 'under',
    'understand', 'until', 'upon', 'use', 'used', 'uses', 'using', 'want',
    'way', 'ways', 'well', 'went', 'whether', 'while', 'within', 'without',
    'work', 'working', 'works', 'world', 'would', 'year', 'years', 'yet',
    'role', 'team', 'join', 'company', 'position', 'opportunity', 'responsibilities',
    'requirements', 'qualifications', 'skills', 'required', 'preferred', 'must',
    'strong', 'excellent', 'ability', 'candidate', 'ideal', 'location', 'salary'
}


def extract_keywords(text: str, min_length: int = 2) -> Set[str]:
    """
    Extract meaningful keywords from text.
    
    Args:
        text: Input text
        min_length: Minimum word length to include
        
    Returns:
        Set of keywords
    """
    words = set(text.lower().split())
    keywords = {w for w in words if len(w) >= min_length and w not in COMMON_WORDS}
    return keywords


def keyword_match_score(job_text: str, resume_text: str) -> float:
    """
    Calculate keyword overlap percentage between job and resume.
    
    This is similar to how Jobscan calculates match scores:
    - Extract keywords from job description
    - Check what percentage appear in resume
    
    Args:
        job_text: Job description text
        resume_text: Resume text
        
    Returns:
        Match score between 0.0 and 1.0
    """
    job_keywords = extract_keywords(job_text)
    resume_keywords = extract_keywords(resume_text)
    
    if not job_keywords:
        return 0.0
    
    # How many job keywords are found in resume?
    matched = job_keywords.intersection(resume_keywords)
    return len(matched) / len(job_keywords)


def compute_tfidf_similarity(
    job_vectors: spmatrix,
    resume_vectors: spmatrix
) -> np.ndarray:
    """
    Compute cosine similarity between job and resume TF-IDF vectors.
    
    Args:
        job_vectors: TF-IDF vectors for jobs (n_jobs x n_features)
        resume_vectors: TF-IDF vectors for resumes (n_resumes x n_features)
        
    Returns:
        Similarity matrix (n_jobs x n_resumes)
    """
    return cosine_similarity(job_vectors, resume_vectors)


def compute_hybrid_scores(
    job_texts: List[str],
    resume_texts: List[str],
    job_vectors: spmatrix,
    resume_vectors: spmatrix,
    keyword_weight: float = 0.7,
    tfidf_weight: float = 0.3
) -> np.ndarray:
    """
    Compute hybrid similarity scores combining keyword matching and TF-IDF.
    
    Args:
        job_texts: List of job description texts
        resume_texts: List of resume texts
        job_vectors: TF-IDF vectors for jobs
        resume_vectors: TF-IDF vectors for resumes
        keyword_weight: Weight for keyword matching (default 0.7)
        tfidf_weight: Weight for TF-IDF similarity (default 0.3)
        
    Returns:
        Hybrid similarity matrix (n_jobs x n_resumes)
    """
    # TF-IDF similarity
    tfidf_sim = compute_tfidf_similarity(job_vectors, resume_vectors)
    
    # Keyword matching scores
    keyword_sim = np.zeros((len(job_texts), len(resume_texts)))
    for job_idx, job_text in enumerate(job_texts):
        for resume_idx, resume_text in enumerate(resume_texts):
            keyword_sim[job_idx, resume_idx] = keyword_match_score(job_text, resume_text)
    
    # Combine scores
    hybrid_scores = keyword_weight * keyword_sim + tfidf_weight * tfidf_sim
    
    return hybrid_scores


def rank_resumes(
    similarity_matrix: np.ndarray,
    top_n: int = 5,
    job_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Rank resumes based on similarity scores.
    
    Args:
        similarity_matrix: Similarity scores (n_jobs x n_resumes)
        top_n: Number of top resumes to return per job
        job_names: Optional list of job names/IDs
        
    Returns:
        DataFrame with columns: Job, Rank, Resume_ID, Similarity_Score
    """
    rankings = []
    
    for job_idx in range(similarity_matrix.shape[0]):
        scores = similarity_matrix[job_idx]
        
        # Rank by score (descending)
        ranked_indices = np.argsort(scores)[::-1]
        
        # Get top N
        top_resumes = ranked_indices[:top_n]
        top_scores = scores[top_resumes]
        
        job_name = job_names[job_idx] if job_names else f"Job_{job_idx + 1}"
        
        for rank, (resume_id, score) in enumerate(zip(top_resumes, top_scores), 1):
            rankings.append({
                'Job': job_name,
                'Rank': rank,
                'Resume_ID': int(resume_id),
                'Similarity_Score': float(score)
            })
    
    return pd.DataFrame(rankings)


def rank_resumes_for_job(
    job_text: str,
    resume_texts: List[str],
    job_vector: spmatrix,
    resume_vectors: spmatrix,
    top_n: int = 10,
    keyword_weight: float = 0.7,
    tfidf_weight: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Rank all resumes for a single job.
    
    Args:
        job_text: Job description text
        resume_texts: List of resume texts
        job_vector: TF-IDF vector for the job (1 x n_features)
        resume_vectors: TF-IDF vectors for resumes (n_resumes x n_features)
        top_n: Number of results to return
        keyword_weight: Weight for keyword matching
        tfidf_weight: Weight for TF-IDF
        
    Returns:
        List of dicts with resume_id, score, rank
    """
    # TF-IDF similarity
    tfidf_scores = cosine_similarity(job_vector, resume_vectors)[0]
    
    # Keyword scores
    keyword_scores = np.array([
        keyword_match_score(job_text, resume_text)
        for resume_text in resume_texts
    ])
    
    # Combine
    hybrid_scores = keyword_weight * keyword_scores + tfidf_weight * tfidf_scores
    
    # Rank
    ranked_indices = np.argsort(hybrid_scores)[::-1][:top_n]
    
    results = []
    for rank, idx in enumerate(ranked_indices, 1):
        results.append({
            'resume_id': int(idx),
            'score': float(hybrid_scores[idx]),
            'rank': rank,
            'keyword_score': float(keyword_scores[idx]),
            'tfidf_score': float(tfidf_scores[idx])
        })
    
    return results


if __name__ == "__main__":
    # Test the module
    sample_job = "python machine learning engineer tensorflow keras deep learning"
    sample_resumes = [
        "python developer machine learning tensorflow keras neural networks",
        "java spring boot backend developer microservices",
        "data scientist python pandas numpy deep learning pytorch"
    ]
    
    print("Testing keyword matching:")
    for i, resume in enumerate(sample_resumes):
        score = keyword_match_score(sample_job, resume)
        print(f"  Resume {i}: {score:.2%}")
    
    print("\nJob keywords:", extract_keywords(sample_job))
