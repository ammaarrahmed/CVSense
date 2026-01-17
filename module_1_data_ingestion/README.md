# Module 1: Data Ingestion & Resume Handling

**Assigned to: Person 1**  
**Status: ✅ COMPLETE**  
**Implementation Date: January 2026**

---

## Overview

Module 1 handles the critical first step of the CVSense pipeline: collecting, extracting, cleaning, and validating resume and job description data. This module ensures high-quality input for all downstream processing modules.

## Responsibilities

- ✅ Collect sample resumes and job descriptions from Kaggle
- ✅ Handle PDF → text extraction
- ✅ Organize dataset and folder structure
- ✅ Ensure consistent input format for preprocessing
- ✅ Validate extracted text quality
- ✅ Document data format specifications

---

## Implementation Files

### Main Notebook
- **`data_ingestion.ipynb`** - Complete implementation of data ingestion pipeline
  - Kaggle dataset download
  - PDF text extraction utilities
  - Data cleaning and validation
  - Quality report generation

### Output Files
- **`data_quality_report.json`** - Metrics and statistics about ingested data
- **`DATA_FORMAT_SPECIFICATION.md`** - Detailed format specifications for downstream modules

---

## Generated Datasets

### 1. Processed Resumes
**Location:** `../data/processed_resumes.csv`

**Columns:**
- `Category` - Resume category (e.g., 'Data Science', 'Software Engineering')
- `cleaned_resume` - Cleaned resume text ready for preprocessing
- `is_valid` - Boolean indicating if resume passed quality validation

**Volume:** Up to 100 resumes (configurable)

### 2. Processed Job Descriptions
**Location:** `../data/processed_job_descriptions.csv`

**Columns:**
- `job_id` - Unique identifier (e.g., 'JD001')
- `title` - Job title
- `category` - Job category
- `description` - Original job description
- `cleaned_description` - Cleaned text ready for preprocessing

**Volume:** 10 sample job descriptions across multiple categories

**Individual Files:** Also saved in `../data/job_descriptions/` for easy access

---

## Key Features

### 1. Data Collection
- Downloads resume dataset from Kaggle using `opendatasets` library
- Creates sample job descriptions for common tech roles
- Limits dataset to specified maximum (100 resumes by default)

### 2. Text Extraction
- **PDF Extraction Function:** `extract_text_from_pdf()`
  - Supports both `pdfplumber` and `PyPDF2` libraries
  - Handles multi-page PDFs
  - Automatic error handling and fallback

### 3. Data Cleaning
- **Text Cleaning Function:** `clean_text()`
  - Removes excessive whitespace
  - Handles encoding errors
  - Normalizes text format
  
### 4. Data Validation
- **Validation Function:** `validate_resume_text()`
  - Minimum length check (50 characters)
  - Alphabetic content ratio validation (>50%)
  - Encoding error detection
  - Detailed issue reporting

### 5. Quality Reporting
- Generates comprehensive quality metrics
- Tracks valid vs. invalid resumes
- Provides statistics (average length, ranges, etc.)
- Saved as JSON for programmatic access

---

## Data Quality Standards

All resumes must meet these criteria to be marked as valid:
- ✓ Minimum 50 characters
- ✓ At least 50% alphabetic content
- ✓ No excessive encoding errors (< 5 '�' characters)
- ✓ Proper text extraction (not corrupted)

**Why This Matters:**
- Poor quality data → Poor feature extraction → Inaccurate matching
- Clean data ensures reliable TF-IDF vectorization
- Validation prevents garbage data from affecting model performance

---

## Usage Instructions

### Quick Start

1. **Setup Kaggle API:**
   ```bash
   cd /home/ammaar/CODE/CVSense
   cp .env.example .env
   # Edit .env with your Kaggle credentials from https://www.kaggle.com/account
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r ../requirements.txt
   ```

3. **Run the Notebook:**
   ```bash
   jupyter notebook data_ingestion.ipynb
   ```
   Or open in VS Code and run all cells

4. **Check Outputs:**
   - Data files in `../data/`
   - Quality report in `data_quality_report.json`

### For Other Team Members (Modules 2-5)

```python
import pandas as pd

# Load the processed data from Module 1
resumes_df = pd.read_csv('../data/processed_resumes.csv')
jobs_df = pd.read_csv('../data/processed_job_descriptions.csv')

# Use only valid resumes
valid_resumes = resumes_df[resumes_df['is_valid'] == True]

# Your preprocessing/feature extraction code here
```

---

## Technical Challenges & Solutions

### Challenge 1: PDF Text Extraction
**Problem:** PDFs can have varying layouts, encodings, and formats
**Solution:** 
- Dual library support (pdfplumber + PyPDF2)
- Text cleaning pipeline to handle common issues
- Validation to catch extraction failures

### Challenge 2: Data Quality Consistency
**Problem:** Raw text may contain noise, formatting issues
**Solution:**
- Standardized cleaning function
- Validation with multiple criteria
- Quality reporting for transparency

### Challenge 3: Dataset Size Management
**Problem:** Large datasets can slow down development
**Solution:**
- Configurable maximum resume count (100 default)
- Random sampling for representative subset
- Scalable architecture for production

---

## Must Be Able to Explain

### 1. Why Data Quality Impacts Model Output
- **TF-IDF Sensitivity:** Noisy text creates spurious features
- **Matching Accuracy:** Corrupted words won't match job descriptions
- **Feature Space:** Poor quality increases dimensionality unnecessarily
- **Example:** "Pythn" vs "Python" - same skill, different features

### 2. Challenges with PDF Resume Extraction
- **Layout Issues:** Multi-column resumes can scramble text order
- **Encoding Problems:** Special characters (™, ©, bullets) may corrupt
- **Images:** Text in images requires OCR (not included)
- **Tables:** Formatting often lost during extraction
- **Hidden Text:** Some PDFs have invisible layers or metadata

---

## Integration with Other Modules

### Module 2 (Text Preprocessing)
**Receives:**
- `data/processed_resumes.csv`
- `data/processed_job_descriptions.csv`

**Expected to:**
- Load cleaned text columns
- Filter by `is_valid == True`
- Apply tokenization, stopword removal, lemmatization
- Output preprocessed text for feature extraction

### Module 3 (Feature Extraction)
**Will receive from Module 2:**
- Preprocessed tokens/text
- Should apply TF-IDF vectorization

### Module 4 (Similarity & Ranking)
**Will receive from Module 3:**
- TF-IDF vectors for resumes and job descriptions
- Should compute cosine similarity and rank

### Module 5 (Evaluation)
**Will receive from Module 4:**
- Ranked resume lists per job description
- Should evaluate and validate results

---

## Deliverables Checklist

- [x] Collect sample resumes (PDFs/text) - ✅ Kaggle dataset
- [x] Collect sample job descriptions - ✅ 10 curated samples
- [x] Implement PDF text extraction - ✅ `extract_text_from_pdf()`
- [x] Create data validation functions - ✅ `validate_resume_text()`
- [x] Organize data in proper folder structure - ✅ `data/resumes/`, `data/job_descriptions/`
- [x] Document data format specifications - ✅ `DATA_FORMAT_SPECIFICATION.md`
- [x] Generate quality report - ✅ `data_quality_report.json`

---

## Configuration

Key parameters in the notebook:

```python
MAX_RESUMES = 100              # Maximum resumes to process
MAX_JOB_DESCRIPTIONS = 50      # Maximum job descriptions
MIN_TEXT_LENGTH = 50           # Minimum valid text length
MIN_ALPHA_RATIO = 0.5          # Minimum alphabetic content ratio
```

---

## Troubleshooting

### Issue: Kaggle Download Fails
**Solution:** 
1. Create Kaggle account at kaggle.com
2. Go to Account → API → Create New API Token
3. Place `kaggle.json` in `~/.kaggle/`
4. Run notebook again

### Issue: PDF Extraction Returns Empty Text
**Solution:**
- Check if PDF is image-based (needs OCR)
- Try alternative library (switch between pdfplumber/PyPDF2)
- Verify PDF is not password-protected

### Issue: Most Resumes Marked Invalid
**Solution:**
- Check validation criteria (might be too strict)
- Adjust `MIN_TEXT_LENGTH` or `MIN_ALPHA_RATIO`
- Review sample invalid resumes for patterns

---

## Contact & Support

For questions about:
- **Data format:** Check `DATA_FORMAT_SPECIFICATION.md`
- **Quality issues:** Review `data_quality_report.json`
- **Implementation details:** See `data_ingestion.ipynb`
- **Module 1 specif Module 1:
- **Data format:** Check `DATA_FORMAT_SPECIFICATION.md`
- **Quality issues:** Review `data_quality_report.json`
- **Implementation:** See `data_ingestion.ipynb`
**Version:** 1.0  
**Status:** Production Ready ✅
