# Quick Start Guide - Module 1: Data Ingestion

**CVSense Resume Classifier - Module 1 Implementation**

---

## 🚀 Getting Started (5 Minutes)

### Step 1: Set Up Kaggle API Credentials
```bash
cd /home/ammaar/CODE/CVSense

# Copy the example environment file
cp .env.example .env

# Edit .env and add your Kaggle credentials
# Get them from: https://www.kaggle.com/account
```

**Detailed instructions:** See [SETUP_KAGGLE_API.md](../SETUP_KAGGLE_API.md)

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Open the Notebook
```bash
# Option 1: Jupyter Notebook
jupyter notebook module_1_data_ingestion/data_ingestion.ipynb

# Option 2: VS Code
# Open the .ipynb file directly in VS Code
```

### Step 4: Run All Cells
- Click "Run All" or execute cells sequentially
- Follow prompts for Kaggle credentials if needed
- Wait for data download and processing (~2-3 minutes)

### Step 5: Verify Output
Check that these files were created:
- ✅ `data/processed_resumes.csv`
- ✅ `data/processed_job_descriptions.csv`
- ✅ `module_1_data_ingestion/data_quality_report.json`

---

## 📊 What You Get

### Resume Dataset
- Up to 100 resumes from Kaggle
- Cleaned and validated text
- Quality metrics included

### Job Descriptions
- 10 sample job postings
- Multiple tech categories
- Ready for matching

### Documentation
- Data quality report (JSON)
- Format specification
- Integration guide for other modules

---

## 🔧 Configuration (Optional)

Edit these variables in the notebook to customize:

```python
MAX_RESUMES = 100              # Change number of resumes
MAX_JOB_DESCRIPTIONS = 50      # Change number of job descriptions
MIN_TEXT_LENGTH = 50           # Validation: minimum text length
MIN_ALPHA_RATIO = 0.5          # Validation: minimum alphabetic ratio
```

---

## 🐛 Common Issues

### 1. Kaggle Authentication Error
**Error:** "Could not authenticate with Kaggle" or "Kaggle credentials not found"

**Fix:** Follow the setup guide: [SETUP_KAGGLE_API.md](../SETUP_KAGGLE_API.md)

**Quick fix:**
```bash
# Option 1: Create .env file
cp .env.example .env
# Then edit .env with your credentials

# Option 2: System-wide setup
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 2. Import Error
**Error:** "ModuleNotFoundError: No module named 'pdfplumber'"

**Fix:**
```bash
pip install pdfplumber PyPDF2 opendatasets kaggle pandas
```

### 3. No Data Downloaded
**Solution:** The notebook will create sample data automatically if Kaggle download fails. You can proceed with the sample data or manually download a dataset.

---

## 📝 For Other Team Members

### Module 2 (Text Preprocessing)
```python
# Load the data Module 1 created
import pandas as pd

resumes = pd.read_csv('data/processed_resumes.csv')
jobs = pd.read_csv('data/processed_job_descriptions.csv')

# Use only valid resumes
valid_resumes = resumes[resumes['is_valid'] == True]

# Your preprocessing code here
# Use: valid_resumes['cleaned_resume']
#      jobs['cleaned_description']
```

### Module 3, 4, 5
- Load output from Module 2
- Follow the data pipeline: Module 1 → 2 → 3 → 4 → 5

---

## ✅ Success Criteria

You've successfully completed Module 1 if:
- [ ] Notebook runs without errors
- [ ] `data/processed_resumes.csv` exists with 50+ resumes
- [ ] `data/processed_job_descriptions.csv` exists with 10 jobs
- [ ] Quality report shows >80% valid resumes
- [ ] Data format matches specification

---

## 🆘 Need Help?

1. **Check the full README:** `module_1_data_ingestion/README.md`
2. **Review data spec:** `module_1_data_ingestion/DATA_FORMAT_SPECIFICATION.md`
3. **Inspect quality report:** `module_1_data_ingestion/data_quality_report.json`
4. **Contact:** Module 1 owner (Person 1)

---

**Estimated Time:** 5-10 minutes  
**Difficulty:** Beginner  
**Prerequisites:** Python 3.8+, pip

---

## Next Steps After Module 1

1. **Module 2 Team:** Start text preprocessing using Module 1 outputs
2. **Module 3 Team:** Wait for Module 2 completion
3. **Module 4 Team:** Wait for Module 3 completion
4. **Module 5 Team:** Plan evaluation metrics

**Happy Coding! 🎉**
