# CVSense - Intelligent Resume Screening System

End-to-end automated resume screening using NLP and Machine Learning to match resumes with job descriptions.

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone <repo-url>
cd CVSense
pip install -r requirements.txt
```

### 2. Set Up Kaggle API (Required for Module 1)
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

### 3. Run Module 1 (Data Ingestion)
```bash
jupyter notebook module_1_data_ingestion/data_ingestion.ipynb
# Or open in VS Code and run all cells
```

### 4. Run Complete Pipeline
```bash
# Run all modules automatically
python main.py

# Or run from a specific module
python main.py --module 2

# Force re-run all modules
python main.py --force

# Run evaluation only
python main.py --evaluate
```

### 5. Launch Web Interface

**Option A: Automated Upload (Recommended)** ✨ NEW
```bash
streamlit run app.py
# Go to "Upload & Process" page
# Upload your resume PDFs
# Add job descriptions
# Click "Process & Match"
# See results instantly!
```

**Option B: View Pre-Processed Results**
```bash
streamlit run app.py
# Browse dashboard, analytics, job search
```

Then open your browser to `http://localhost:8501`

[📖 Upload Guide](UPLOAD_GUIDE.md) - Complete guide for the automated upload feature

---

## 📊 System Architecture

### Complete Pipeline Flow

```
Module 1: Data Ingestion
    ↓
Module 2: Text Preprocessing
    ↓
Module 3: TF-IDF Vectorization
    ↓
Module 4: Similarity Ranking
    ↓
Module 5: Evaluation & Validation
```

---

## 📁 Project Structure

### Module 1: Data Ingestion & Resume Handling

Handles data collection, PDF extraction, and dataset organization.

**Location:** `module_1_data_ingestion/`

**Key Features:**
- Downloads resume dataset from Kaggle (100 resumes)
- Extracts text from PDF files
- Creates 10 sample job descriptions
- Validates data quality
- Outputs: `data/processed_resumes.csv`, `data/processed_job_descriptions.csv`

**Run:**
```bash
jupyter notebook module_1_data_ingestion/data_ingestion.ipynb
```

### Module 2: Text Preprocessing

Combines text cleaning, normalization, and linguistic preprocessing.

**Location:** `module_2_text_preprocessing/`

**Key Features:**
- Phase A: Text cleaning (lowercase, remove numbers/punctuation)
- Phase B: Linguistic preprocessing (tokenization, stopword removal, lemmatization)
- NLTK-based processing
- Outputs: `data/linguistically_preprocessed_files/preprocessed_*.csv`

**Run:**
```bash
python main.py --module 2
```

### Module 3: Feature Extraction (TF-IDF)

Implements TF-IDF vectorization to convert text into numerical vectors.

**Location:** `module_3_feature_extraction/`

**Key Features:**
- TF-IDF vectorization with 5000 max features
- Bi-gram support (1-2 word combinations)
- Consistent vocabulary across resumes and jobs
- Outputs: `module_3_feature_extraction/tfidf_vectors.pkl`

**Run:**
```bash
python main.py --module 3
```

### Module 4: Similarity Computation & Ranking

Computes cosine similarity and ranks resumes based on relevance.

**Location:** `module_4_similarity_ranking/`

**Key Features:**
- Cosine similarity calculation
- Top-5 resume ranking per job
- Similarity score matrix
- Outputs: `module_4_similarity_ranking/module5_resume_ranking.csv`

**Run:**
```bash
python main.py --module 4
```

### Module 5: Evaluation, Validation & Documentation

Analyzes results, validates rankings, and documents system performance.

**Location:** Root directory files

**Key Features:**
- Statistical analysis of similarity scores
- Score distribution visualization
- Manual validation template generation
- Performance metrics
- Outputs: `evaluation_score_distribution.png`, `manual_validation.csv`

**Run:**
```bash
python main.py --evaluate
```

---

## 🛠️ Command-Line Interface

### Main Pipeline Script (`main.py`)

```bash
# Run complete pipeline (Modules 2-5)
python main.py

# Force re-run all modules (ignore cached data)
python main.py --force

# Start from specific module
python main.py --module 2  # Runs 2→3→4→5
python main.py --module 3  # Runs 3→4→5
python main.py --module 4  # Runs 4→5

# Run evaluation only
python main.py --evaluate

# Get help
python main.py --help
```

**Pipeline Intelligence:**
- Automatically detects which modules have already run
- Skips completed modules (unless `--force` is used)
- Validates prerequisites before execution
- Provides progress updates and status checks

---

## 🌐 Web Interface (`app.py`)

### Launch the App

```bash
streamlit run app.py
```

### Features

**🏠 Dashboard**
- Quick overview of system status
- Score distribution visualization
- Top matches per job
- Key performance metrics

**🔍 Job Search**
- Select job description
- View top matching resumes
- Adjust number of results
- Expandable resume previews

**📊 Analytics**
- Overall performance statistics
- Score distribution by job
- Top performing resumes
- Category-wise analysis

**📄 Resume Explorer**
- Browse individual resumes
- View original, cleaned, and preprocessed text
- See ranking performance across all jobs
- Match score statistics

**ℹ️ About**
- System documentation
- Pipeline explanation
- Technology stack
- Usage instructions

---

## 📂 Data Directory Structure

```
data/
├── resumes/                          # Raw resume PDFs (from Kaggle)
├── job_descriptions/                 # Sample job description files
├── processed_resumes.csv             # Module 1 output
├── processed_job_descriptions.csv    # Module 1 output
└── linguistically_preprocessed_files/
    ├── preprocessed_resumes_final.csv    # Module 2 output
    └── preprocessed_jobs_final.csv       # Module 2 output
```

**Note:** CSV files and raw data are in `.gitignore` - they're generated by running the pipeline.

---

## 🔧 Dependencies

### Core Libraries
- **Python 3.8+**: Base language
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: TF-IDF, cosine similarity

### NLP & Text Processing
- **NLTK**: Tokenization, lemmatization, stopwords
- **PyPDF2/pdfplumber**: PDF text extraction

### Visualization & UI
- **matplotlib**: Static plots
- **seaborn**: Statistical visualizations
- **plotly**: Interactive charts
- **streamlit**: Web interface

### Data Collection
- **kaggle**: API for dataset download
- **python-dotenv**: Environment variable management

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## 🎯 Usage Examples

### Example 1: Complete First-Time Setup

```bash
# 1. Setup credentials
cp .env.example .env
# Edit .env with your Kaggle credentials

# 2. Run Module 1 (data ingestion)
jupyter notebook module_1_data_ingestion/data_ingestion.ipynb
# Run all cells in the notebook

# 3. Run complete pipeline
python main.py

# 4. Launch web interface
streamlit run app.py
```

### Example 2: Re-run After Changes

```bash
# Re-run from Module 2 onwards (if you changed preprocessing)
python main.py --module 2

# Re-run from Module 3 onwards (if you changed vectorization)
python main.py --module 3

# Force complete re-run (ignore cached data)
python main.py --force
```

### Example 3: Quick Evaluation

```bash
# Only run evaluation (Module 5)
python main.py --evaluate
```

---

## 📊 Expected Outputs

### Module 1
- ✅ `data/processed_resumes.csv` (100 resumes)
- ✅ `data/processed_job_descriptions.csv` (10 jobs)
- ✅ `module_1_data_ingestion/data_quality_report.json`

### Module 2
- ✅ `data/linguistically_preprocessed_files/preprocessed_resumes_final.csv`
- ✅ `data/linguistically_preprocessed_files/preprocessed_jobs_final.csv`

### Module 3
- ✅ `module_3_feature_extraction/tfidf_vectors.pkl`

### Module 4
- ✅ `module_4_similarity_ranking/module5_resume_ranking.csv`

### Module 5
- ✅ `evaluation_score_distribution.png`
- ✅ `manual_validation.csv`

---

## 🔍 Troubleshooting

### Module 1 Issues

**Problem:** Kaggle API not working
```bash
# Solution: Verify .env file
cat .env
# Should show KAGGLE_USERNAME and KAGGLE_KEY

# Alternative: Check kaggle config
cat ~/.kaggle/kaggle.json
```

**Problem:** PDF extraction errors
- Some PDFs may fail extraction (images, scanned documents)
- Module 1 automatically handles errors and continues
- Check `data_quality_report.json` for validation results

### Pipeline Issues

**Problem:** "Module 1 data not found"
```bash
# Solution: Run Module 1 first
jupyter notebook module_1_data_ingestion/data_ingestion.ipynb
```

**Problem:** Missing NLTK data
```python
# Solution: Download NLTK resources
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
```

### Web Interface Issues

**Problem:** Streamlit not found
```bash
# Solution: Install streamlit
pip install streamlit plotly
```

**Problem:** "Rankings not available"
```bash
# Solution: Run pipeline first
python main.py
```

---

## 👥 Team Contributions

- **Module 1**: Data Ingestion & Resume Handling
- **Module 2**: Text Preprocessing (Person 2 & 3)
- **Module 3**: Feature Extraction - TF-IDF (Person 4)
- **Module 4**: Similarity Computation & Ranking (Person 5)
- **Module 5**: Evaluation & Documentation

---

## 📝 License

This project was created for educational purposes as part of an internship assignment.

---

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ End-to-end ML pipeline development
- ✅ Natural Language Processing techniques
- ✅ TF-IDF vectorization and cosine similarity
- ✅ Modular code architecture
- ✅ Web application development
- ✅ Data visualization
- ✅ Team collaboration via Git

---

## 🚀 Future Enhancements

Potential improvements:
- [ ] Add custom job description input
- [ ] Implement resume upload functionality
- [ ] Use deep learning embeddings (Word2Vec, BERT)
- [ ] Add multi-language support
- [ ] Implement user authentication
- [ ] Export rankings to PDF reports
- [ ] Add email notification for top candidates

---

**For detailed module documentation, see individual `README.md` files in each module directory.**
