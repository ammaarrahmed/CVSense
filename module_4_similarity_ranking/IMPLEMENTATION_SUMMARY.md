# Module 4: Resume Similarity & Ranking

## Objective
This module computes the similarity between job descriptions and resumes using TF-IDF vectors, and ranks the resumes based on relevance to each job. The goal is to identify the top matching resumes for each job description.

## Inputs
- `tfidf_vectors.pkl` generated in Module 3, containing:
  - `jd_vectors`: TF-IDF vectors for 10 job descriptions (shape: 10x5000)
  - `resume_vectors`: TF-IDF vectors for 100 resumes (shape: 100x5000)

## Process
1. Load the TF-IDF vectors from the pickle file.
2. Compute cosine similarity between each job and all resumes.
3. Rank resumes for each job in descending order of similarity score.
4. Handle ties using stable sorting.
5. Select the top N resumes (default N=5) for each job.

## Outputs
- `module5_resume_ranking.csv`: A CSV file containing the rankings with columns:
  - `Job`: Job identifier (Job_1 to Job_10)
  - `Rank`: Rank of the resume for that job (1 = most relevant)
  - `Resume_ID`: Resume identifier
  - `Similarity_Score`: Cosine similarity score with the job

