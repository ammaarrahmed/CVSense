# Module 5: Evaluation & Manual Validation

## 1. Overview
The Evaluation Module is designed to assess the performance of the Intelligent Resume Screening system. It analyzes the similarity scores generated in Module 4 and provides insights into how the ranking varies across different job descriptions.

## 2. Similarity Score Analysis
This section analyzes the **Cosine Similarity** values:
- **Mean Score:** Represents the average match quality.
- **Spread (STD):** High standard deviation indicates that the system is effectively distinguishing between "Excellent" and "Poor" matches.
- **Outliers:** Identification of resumes with exceptionally high scores (>0.8) which are prioritized.

## 3. Evaluation Metrics
- **Job Relevance Check:** Manual comparison of the Top-3 ranked resumes against the Job Description.
- **Score Distribution:** Visualizing the histogram to identify the "sweet spot" for shortlisting thresholds.
- **System Filtering Efficiency:** Calculating what percentage of resumes are automatically filtered out using a similarity threshold (e.g., 0.50).

## 4. Manual Validation (Human-in-the-loop)
To validate the system, recruiters should use the following rubric:
| Rank | Resume ID | AI Score | Human Accuracy (Y/N) | Notes |
|------|-----------|----------|----------------------|-------|
| 1    | RES_001   | 0.78     |                      |       |
| 2    | RES_045   | 0.72     |                      |       |
| 3    | RES_012   | 0.65     |                      |       |

## 5. Conclusion
Module 5 ensures that the ML pipeline isn't just a "black box" but provides actionable data for HR professionals to trust and verify.
