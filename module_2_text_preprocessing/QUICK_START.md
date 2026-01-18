# Quick Start Guide - Module 2: Text Preprocessing

**CVSense Resume Classifier - Module 2 Implementation**

---

## 🚀 Getting Started (5 Minutes)

### Step 1: Ensure Module 1 Data Exists
Make sure you have the processed data from Module 1:
- ✅ `data/processed_resumes.csv` (100 resumes)
- ✅ `data/processed_job_descriptions.csv` (10 job descriptions)

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
# Includes: nltk, pandas, scikit-learn, numpy
```

### Step 3: Open the Notebook
```bash
# Option 1: Jupyter Notebook
jupyter notebook module_2_text_preprocessing/module_2_text_preprocessing.ipynb

# Option 2: VS Code
# Open the .ipynb file directly in VS Code
```

### Step 4: Run All Cells
- Click "Run All" or execute cells sequentially
- NLTK resources will auto-download (~1-2 minutes first time)
- Processing takes ~2-3 minutes for 100 resumes + 10 jobs
- Wait for completion

### Step 5: Verify Output
Check that these files were created in `data/linguistically_preprocessed_files/`:
- ✅ `preprocessed_resumes_final.csv`
- ✅ `preprocessed_jobs_final.csv`

---

## 📊 What You Get

### Preprocessed Resumes
- **File:** `data/linguistically_preprocessed_files/preprocessed_resumes_final.csv`
- **Rows:** 100 resumes
- **Columns:**
  - `Category`: Resume category/field
  - `cleaned_resume`: Original resume text
  - `is_valid`: Validation flag
  - `cleaned_text`: After Phase A (text cleaning)
  - `preprocessed_text`: After Phase B (linguistic preprocessing) ← **USE THIS FOR MODULE 3**

### Preprocessed Jobs
- **File:** `data/linguistically_preprocessed_files/preprocessed_jobs_final.csv`
- **Rows:** 10 job descriptions
- **Columns:**
  - `job_id`: Job identifier
  - `title`: Job title
  - `category`: Job category
  - `description`: Original description
  - `cleaned_description`: Cleaned text
  - `cleaned_text`: After Phase A (text cleaning)
  - `preprocessed_text`: After Phase B (linguistic preprocessing) ← **USE THIS FOR MODULE 3**

### Data Quality
- **100% valid preprocessing** - All 100 resumes + 10 jobs successfully processed
- **Average tokens per resume:** 642
- **Average tokens per job:** 39
- **No null values** in output

---

## 🔍 What Preprocessing Does

### Phase A: Text Cleaning
1. **Lowercase conversion** - Standardizes case
2. **Remove numbers** - Eliminates phone numbers, years, zip codes
3. **Remove punctuation** - Strips special characters
4. **Remove newlines** - Handles PDF formatting
5. **Normalize whitespace** - Cleans spacing

**Result:** Standardized, noise-free text

### Phase B: Linguistic Preprocessing
1. **Tokenization** - Splits text into words
2. **Stopword removal** - Removes "the", "is", "in", etc. (~30-40% reduction)
3. **Lemmatization** - Reduces to base forms (engineer, engineers → engineer)

**Result:** Clean, normalized, lemmatized tokens ready for vectorization

---

## 💾 For Module 3: Feature Extraction

Team 3 should use the **`preprocessed_text` column** from the output files:

```python
import pandas as pd

# Load preprocessed data
resumes = pd.read_csv('data/linguistically_preprocessed_files/preprocessed_resumes_final.csv')
jobs = pd.read_csv('data/linguistically_preprocessed_files/preprocessed_jobs_final.csv')

# Use preprocessed_text for vectorization
resume_texts = resumes['preprocessed_text']  # Ready for TF-IDF, embeddings, etc.
job_texts = jobs['preprocessed_text']        # Ready for vectorization
```

---

## 🔧 Configuration (Optional)

Edit these variables in the notebook to customize:

```python
# File paths
resumes_path = r"C:\...\processed_resumes.csv"
jobs_path = r"C:\...\processed_job_descriptions.csv"
output_folder = r"C:\...\linguistically_preprocessed_files"

# NLTK resources (auto-downloaded)
nltk.download('punkt_tab')      # Tokenizer
nltk.download('stopwords')      # Stopword list
nltk.download('wordnet')        # Lemmatizer dictionary
```

---

## 🐛 Common Issues

### 1. NLTK Resource Not Found Error
**Error:** "Resource punkt_tab not found"

**Fix:**
```python
import nltk
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
```

### 2. File Not Found
**Error:** "processed_resumes.csv not found"

**Fix:**
- Ensure Module 1 was run first
- Check file paths are correct
- Verify `data/` directory exists

### 3. Out of Memory
**For large datasets:** Process in batches instead of all at once

```python
chunk_size = 20
for i in range(0, len(df), chunk_size):
    chunk = df.iloc[i:i+chunk_size]
    # Process chunk
```

---

## 📚 Documentation

For detailed information, see:
- **[MODULE_2_DOCUMENTATION.md](MODULE_2_DOCUMENTATION.md)** - Complete technical reference
- **[IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)** - What was implemented
- **[module_2_text_preprocessing.ipynb](module_2_text_preprocessing.ipynb)** - Full code with explanations

---

## ✨ Key Insights

✓ **Why preprocessing matters:**
- Raw text has noise (numbers, symbols, formatting)
- Creates high-dimensional, unreliable vectors
- Preprocessing reduces dimensionality by ~40%

✓ **Why lemmatization over stemming:**
- Lemmatization: "engineer", "engineers" → "engineer" (real word)
- Stemming: "engineer" → "engin" (not a real word)
- Better for semantic matching in CV domain

✓ **Preprocessing order is critical:**
- ❌ WRONG: Tokenize → Lemmatize → Clean
- ✓ RIGHT: Clean → Tokenize → Remove Stopwords → Lemmatize

---

## 🎯 Next Steps

**For Team 3 (Module 3 - Feature Extraction):**

1. Load the preprocessed data from `data/linguistically_preprocessed_files/`
2. Use the `preprocessed_text` column
3. Apply TF-IDF vectorization
4. Extract semantic embeddings
5. Combine features for matching

The data is clean, normalized, and ready to go! 🚀

---

## 📞 Questions?

Refer to the module documentation or check the notebook cells for detailed explanations of each preprocessing step.
