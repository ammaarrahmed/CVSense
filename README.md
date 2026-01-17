# CVSense - Resume Ranking System

A modular resume-job description matching system using TF-IDF and cosine similarity.

## 🚀 Quick Start

1. **Clone the repository**
2. **Set up Kaggle API credentials** - See [SETUP_KAGGLE_API.md](SETUP_KAGGLE_API.md)
3. **Install dependencies:** `pip install -r requirements.txt`
4. **Run Module 1:** Open `module_1_data_ingestion/data_ingestion.ipynb`

## 🔐 Important: API Security

This project requires Kaggle API credentials to download datasets. **Your credentials are private!**

- ✅ Copy `.env.example` to `.env` and add your credentials
- ✅ `.env` is in `.gitignore` (never committed to git)
- ✅ Each team member uses their own API key
- ❌ Never commit `kaggle.json` or `.env` files

**See [SETUP_KAGGLE_API.md](SETUP_KAGGLE_API.md) for detailed setup instructions.**

---

## Project Structure

### Module 1: Data Ingestion & Resume Handling
**Assigned to: Person 1**

Handles data collection, PDF extraction, and dataset organization.

**Location:** `module_1_data_ingestion/`

### Module 2: Text Preprocessing
**Assigned to: Person 2 & Person 3**

Combines text cleaning, normalization, and linguistic preprocessing.

**Location:** `module_2_text_preprocessing/`

**Includes:**
- Text cleaning & normalization (lowercase, punctuation removal, PDF noise handling)
- Linguistic preprocessing (tokenization, stopword removal, lemmatization)

### Module 3: Feature Extraction (TF-IDF)
**Assigned to: Person 4**

Implements TF-IDF vectorization to convert text into numerical vectors.

**Location:** `module_3_feature_extraction/`

### Module 4: Similarity Computation & Ranking
**Assigned to: Person 5**

Computes cosine similarity and ranks resumes based on relevance.

**Location:** `module_4_similarity_ranking/`

### Module 5: Evaluation, Validation & Documentation
**Assigned to: Person 6**

Analyzes results, validates rankings, and documents system limitations.

**Location:** `module_5_evaluation_documentation/`

## Data Directory

`data/` - Contains resumes and job descriptions
- `data/resumes/` - PDF/text resumes
- `data/job_descriptions/` - Job description files

## Setup

```bash
pip install -r requirements.txt
```

## Usage

To be implemented by respective module owners.
