# CVSense - Intelligent Resume Screening System

Automated resume screening using NLP and Machine Learning to match resumes with job descriptions.

## 🚀 Quick Start

### 1. Install
```bash
git clone https://github.com/ammaarrahmed/CVSense.git
cd CVSense
pip install -r requirements.txt
```

### 2. Launch Web Interface
```bash
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

---

## 💡 Main Use Case: Google Forms Job Applications

Perfect for screening resumes submitted through Google Forms!

### How It Works

1. **Create a Google Form** with resume file upload
2. **Submitted resumes** are stored in a Google Drive folder
3. **Share the folder** (anyone with link can view)
4. **Paste the link in CVSense** → Instant ranking!

### Step-by-Step

1. Go to **📁 Google Drive Import** page
2. Paste your shared Drive folder URL
3. Click "Download Resumes from Drive"
4. Add your job description(s)
5. Click "Match Resumes to Jobs"
6. View ranked results on Dashboard!

---

## 📊 Features

### 🔗 Google Drive Integration
- Import resumes from shared Drive folders
- Supports PDF, DOCX, and TXT files
- Batch processing for multiple resumes

### ⬆️ Direct Upload
- Upload resume PDFs directly
- Add job descriptions via text or file upload
- Instant processing and ranking

### 🎯 Hybrid Matching Algorithm
- **Keyword matching** (like Jobscan)
- **TF-IDF cosine similarity**
- **Phrase matching** for technical terms
- **Synonym expansion** (ML ↔ machine learning)

### 📈 Results Dashboard
- Score distribution visualization
- Top candidates per job
- Export rankings to CSV

---

## 📁 Project Structure

```
CVSense/
├── app.py                          # Streamlit web interface
├── main.py                         # CLI pipeline runner
├── requirements.txt                # Dependencies
│
├── module_1_data_ingestion/        # Google Drive integration
│   ├── google_drive.py             # Download & extract resumes
│   └── README.md
│
├── module_2_text_preprocessing/    # Text cleaning
│   ├── preprocessing.py            # clean_text(), preprocess_text()
│   └── README.md
│
├── module_3_feature_extraction/    # TF-IDF vectorization
│   ├── tfidf.py                    # create_tfidf_vectors()
│   └── README.md
│
├── module_4_similarity_ranking/    # Matching algorithm
│   ├── ranking.py                  # Hybrid scoring, ranking
│   └── README.md
│
├── module_5_evaluation_documentation/
│   └── README.md
│
└── data/                           # Sample data (optional)
    ├── resumes/
    └── job_descriptions/
```

---

## 🛠️ Modules

### Module 1: Data Ingestion
**Google Drive integration for importing resumes**

```python
from module_1_data_ingestion import download_resumes_from_drive, process_resume_files

# Download from Google Drive
result = download_resumes_from_drive("https://drive.google.com/drive/folders/...")
resumes = process_resume_files(result['files'])
```

### Module 2: Text Preprocessing
**Clean and normalize text**

```python
from module_2_text_preprocessing import clean_text, preprocess_text

cleaned = clean_text(raw_text)
processed = preprocess_text(cleaned)
```

### Module 3: Feature Extraction
**TF-IDF vectorization**

```python
from module_3_feature_extraction import create_tfidf_vectors

vectors = create_tfidf_vectors(resume_texts, job_texts)
```

### Module 4: Similarity & Ranking
**Hybrid matching with keyword + TF-IDF scoring**

```python
from module_4_similarity_ranking import compute_hybrid_scores, rank_resumes

scores = compute_hybrid_scores(job_texts, resume_texts, job_vectors, resume_vectors)
rankings = rank_resumes(scores, top_n=5)
```

---

## 🌐 Web Interface Pages

| Page | Description |
|------|-------------|
| 🏠 **Dashboard** | View results, score distribution, top matches |
| ⬆️ **Upload & Process** | Upload PDFs and job descriptions directly |
| 📁 **Google Drive Import** | Import resumes from shared Drive folder |
| ℹ️ **About** | Documentation and system info |

---

## 🔧 Dependencies

```
# Core
pandas, numpy, scikit-learn

# NLP
nltk

# PDF/Document Processing
pdfplumber, PyPDF2, python-docx

# Google Drive
gdown, requests

# Web Interface
streamlit, plotly
```

Install all:
```bash
pip install -r requirements.txt
```

---

## 🎯 Matching Algorithm

CVSense uses a **hybrid approach** combining multiple techniques:

### 1. Keyword Matching (70%)
- Extracts meaningful words from job description
- Checks overlap with resume keywords
- Expands synonyms (AI ↔ artificial intelligence, ML ↔ machine learning)

### 2. TF-IDF Similarity (30%)
- Vectorizes text with n-grams (1-2 words)
- Computes cosine similarity
- Captures semantic patterns

### 3. Phrase Matching (Bonus)
- Detects technical phrases ("machine learning", "data science", etc.)
- Boosts score for matching phrases

### Synonym/Abbreviation Support
Automatically matches:
- `ML` ↔ `machine learning`
- `AI` ↔ `artificial intelligence`
- `JS` ↔ `JavaScript`
- `k8s` ↔ `kubernetes`
- And 40+ more mappings

---

## 📊 Example Output

```
Job: Machine Learning Engineer

Rank | Resume              | Match Score
-----|---------------------|------------
1    | john_doe_ml.pdf     | 67.3%
2    | jane_smith_ds.pdf   | 54.1%
3    | bob_wilson_dev.pdf  | 42.8%
4    | alice_chen_eng.pdf  | 38.5%
5    | mike_jones_sw.pdf   | 31.2%
```

---

## 🔍 Troubleshooting

### Google Drive Issues

**"gdown not installed"**
```bash
pip install gdown
```

**"Could not download folder"**
- Make sure folder is shared as "Anyone with link can view"
- Check that the URL is correct

### PDF Extraction Issues

**"Could not extract text"**
- Some scanned PDFs (images) don't have extractable text
- Try DOCX or TXT format instead

### NLTK Data Missing

```python
import nltk
nltk.download('punkt')
nltk.download('wordnet')
```

---

## 👥 Team

- **Module 1**: Data Ingestion & Google Drive Integration
- **Module 2**: Text Preprocessing
- **Module 3**: Feature Extraction (TF-IDF)
- **Module 4**: Similarity Computation & Ranking
- **Module 5**: Evaluation & Documentation

---

## 📝 License

Educational project for internship assignment.

---

## 🚀 Future Enhancements

- [ ] Add deep learning embeddings (sentence-transformers)
- [ ] Email notifications for top candidates
- [ ] PDF report generation
- [ ] Multi-language support
- [ ] OAuth for private Drive folders
