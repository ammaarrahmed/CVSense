# CVSense - Resume Ranking System

A modular resume-job description matching system using TF-IDF and cosine similarity.

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone <repo-url>
cd CVSense
pip install -r requirements.txt
```

### 2. Set Up Kaggle API (Required for data download)
```bash
# Copy template
cp .env.example .env

# Get your Kaggle API credentials:
# 1. Go to https://www.kaggle.com/account
# 2. Click "Create New API Token" (downloads kaggle.json)
# 3. Open kaggle.json and copy username & key values
# 4. Paste them into .env file
```

**Your `.env` file should look like:**
```
KAGGLE_USERNAME=your_username
KAGGLE_KEY=abc123def456...
```

**Note:** `.env` is in `.gitignore` - your credentials stay private!

### 3. Run Module 1
```bash
jupyter notebook module_1_data_ingestion/data_ingestion.ipynb
# Or open in VS Code
```

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
