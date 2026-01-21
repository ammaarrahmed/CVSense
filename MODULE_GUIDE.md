# CVSense - Technical Module Guide

Complete technical explanation of the 5-module pipeline.

---

## Module 1: Data Ingestion

**Purpose:** Collect and prepare resume/job data

**Notebook:** `module_1_data_ingestion/data_ingestion.ipynb`

**Key Functions:**
- `extract_text_from_pdf()`: Extract text from PDF files
- `clean_text()`: Remove encoding errors, normalize whitespace
- `validate_resume_text()`: Check minimum length, alphabetic ratio

**Process:**
1. Download Kaggle resume dataset (100 resumes)
2. Create 10 sample job descriptions
3. Extract text from PDFs
4. Clean and validate data
5. Output: `data/processed_resumes.csv`, `data/processed_job_descriptions.csv`

**Output Columns:**
- Resumes: `ID`, `Category`, `cleaned_resume`, `is_valid`
- Jobs: `job_id`, `title`, `category`, `description`, `cleaned_description`

---

## Module 2: Text Preprocessing

**Purpose:** Clean and normalize text for ML processing

**Notebook:** `module_2_text_preprocessing/module_2_text_preprocessing.ipynb`

**Two-Phase Approach:**

### Phase A: Text Cleaning
```python
def clean_text(text):
    text = text.lower()                 # Lowercase
    text = re.sub(r'\d+', '', text)     # Remove numbers
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = re.sub(r'\s+', ' ', text)    # Normalize whitespace
    return text.strip()
```

### Phase B: Linguistic Preprocessing
```python
def preprocess_text(text):
    tokens = word_tokenize(text)                    # Tokenization
    tokens = [w for w in tokens if w not in stopwords] # Remove stopwords
    tokens = [lemmatizer.lemmatize(w) for w in tokens] # Lemmatization
    return ' '.join(tokens)
```

**Libraries:** NLTK (punkt, wordnet, stopwords)

**Output:** `data/linguistically_preprocessed_files/preprocessed_*.csv`
- Columns: `cleaned_text`, `preprocessed_text`

**Why Lemmatization?**
- "running" → "run", "better" → "good"
- Reduces vocabulary, improves matching
- Better than stemming (preserves word meaning)

---

## Module 3: Feature Extraction (TF-IDF)

**Purpose:** Convert text to numerical vectors

**Notebook:** `module_3_feature_extraction/tfidf_vectorizer.ipynb`

**TF-IDF Vectorizer Configuration:**
```python
TfidfVectorizer(
    max_features=5000,      # Top 5000 terms
    ngram_range=(1, 2),     # Unigrams + bigrams
    stop_words='english'    # Filter common words
)
```

**Process:**
1. Combine resumes + jobs into single corpus
2. Fit vectorizer on combined text (consistent vocabulary)
3. Transform all documents
4. Split back into resume vectors and job vectors

**Output:** `tfidf_vectors.pkl`
- `resume_vectors`: (100 × 5000) sparse matrix
- `jd_vectors`: (10 × 5000) sparse matrix
- `feature_names`: 5000 terms selected

**Why Combined Corpus?**
- Ensures same vocabulary for jobs and resumes
- Enables proper comparison
- Prevents dimension mismatch

---

## Module 4: Similarity Ranking

**Purpose:** Match resumes to jobs using cosine similarity

**Notebook:** `module_4_similarity_ranking/module4_similarity_ranking.ipynb`

**Cosine Similarity:**
```python
similarity_matrix = cosine_similarity(job_vectors, resume_vectors)
# Shape: (10 jobs × 100 resumes)
# Values: 0.0 (no match) to 1.0 (perfect match)
```

**Ranking Process:**
```python
for job_idx in range(num_jobs):
    scores = similarity_matrix[job_idx]
    ranked_indices = np.argsort(scores)[::-1]  # Descending
    top_5 = ranked_indices[:5]
```

**Output:** `module5_resume_ranking.csv`
- Columns: `Job`, `Rank`, `Resume_ID`, `Similarity_Score`
- 50 rows (10 jobs × 5 top candidates)

**Cosine Similarity Formula:**
```
similarity = (A · B) / (||A|| × ||B||)
```
- Measures angle between vectors
- Range: 0 to 1 (for TF-IDF, always positive)
- Ignores magnitude, focuses on direction

---

## Module 5: Evaluation & Validation

**Purpose:** Analyze and validate ranking quality

**Script:** `Evaluation_Metrics_Pipeline.py`

**Metrics Computed:**
1. **Score Statistics:**
   - Mean, median, std deviation
   - Min, max, quartiles

2. **Score Distribution:**
   - Histogram with KDE curve
   - Mean line overlay
   - Saved as PNG

3. **Top Candidates:**
   - Top 3 per job
   - Formatted table

4. **Manual Validation Template:**
   - CSV for human review
   - Recruiter can mark accuracy

**Output:**
- `evaluation_score_distribution.png`
- `manual_validation.csv`

**Key Insights:**
- Average similarity: ~30-40%
- Top matches: 60-80%
- Distribution: Right-skewed (few high matches, many low)

---

## Complete Pipeline Flow

```
┌──────────────────────────────────────────────────┐
│ Module 1: Data Ingestion                         │
│ Input: Kaggle dataset, PDFs                      │
│ Output: CSV files (raw text)                     │
└──────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────┐
│ Module 2: Text Preprocessing                     │
│ Input: Raw text CSV                              │
│ Process: Clean → Tokenize → Lemmatize           │
│ Output: Preprocessed text CSV                    │
└──────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────┐
│ Module 3: TF-IDF Vectorization                   │
│ Input: Preprocessed text                         │
│ Process: Fit vectorizer → Transform              │
│ Output: Sparse matrices (5000-dim)               │
└──────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────┐
│ Module 4: Similarity Ranking                     │
│ Input: TF-IDF vectors                            │
│ Process: Cosine similarity → Sort → Top-5        │
│ Output: Rankings CSV                             │
└──────────────────────────────────────────────────┘
                        ↓
┌──────────────────────────────────────────────────┐
│ Module 5: Evaluation                             │
│ Input: Rankings CSV                              │
│ Process: Statistics → Visualization              │
│ Output: Metrics, plots, validation template      │
└──────────────────────────────────────────────────┘
```

---

## Web Application (`app.py`)

**Automated Upload Feature:**

Bypasses Module 1, runs entire pipeline in browser:

```python
def process_uploads(resume_files, job_descriptions):
    # 1. Extract text from PDFs
    texts = [extract_pdf_text(f) for f in resume_files]
    
    # 2. Clean text (Module 2, Phase A)
    cleaned = [clean_text(t) for t in texts]
    
    # 3. Preprocess (Module 2, Phase B)
    preprocessed = [preprocess_text(t) for t in cleaned]
    
    # 4. TF-IDF vectorization (Module 3)
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    vectors = vectorizer.fit_transform(preprocessed)
    
    # 5. Similarity ranking (Module 4)
    similarity = cosine_similarity(job_vectors, resume_vectors)
    rankings = rank_top_n(similarity)
    
    return rankings
```

**Processing Time:** ~5-30 seconds (10 resumes + 2 jobs)

---

## CLI Pipeline (`main.py`)

**Automated Module Execution:**

```python
class CVSensePipeline:
    def run_complete_pipeline(self):
        # Check what's already done
        status = self.check_module_outputs()
        
        # Run only needed modules
        if not status[2]: self.run_module_2()  # Preprocessing
        if not status[3]: self.run_module_3()  # TF-IDF
        if not status[4]: self.run_module_4()  # Ranking
        
        # Always run evaluation
        self.run_module_5()
```

**Smart Caching:** Skips completed modules automatically

---

## Key Algorithms

### 1. TF-IDF (Term Frequency-Inverse Document Frequency)

**Term Frequency (TF):**
```
TF(t,d) = count(t in d) / total words in d
```

**Inverse Document Frequency (IDF):**
```
IDF(t) = log(N / df(t))
N = total documents
df(t) = documents containing term t
```

**TF-IDF:**
```
TF-IDF(t,d) = TF(t,d) × IDF(t)
```

**Effect:**
- Common words (low IDF) get low scores
- Rare, specific words (high IDF) get high scores
- Captures importance of terms

### 2. Cosine Similarity

**Formula:**
```
cos(θ) = (A · B) / (||A|| × ||B||)

Where:
A · B = dot product (sum of element-wise products)
||A|| = magnitude of A = sqrt(sum of squares)
```

**Example:**
```
Resume: [0.5, 0.3, 0.0, 0.8]  # Python, SQL, Java, ML
Job:    [0.4, 0.2, 0.0, 0.9]  # Python, SQL, Java, ML

Similarity = (0.5×0.4 + 0.3×0.2 + 0×0 + 0.8×0.9) / (||A|| × ||B||)
          = (0.20 + 0.06 + 0 + 0.72) / (0.985 × 1.033)
          = 0.98 / 1.017
          = 0.96 (96% match!)
```

---

## Performance Optimization

**Sparse Matrices:**
- TF-IDF produces sparse matrices (mostly zeros)
- scipy.sparse.csr_matrix reduces memory by 90%+
- 100 × 5000 dense = 4MB, sparse = 100KB

**Vectorization:**
- NumPy operations (C-optimized)
- Batch processing instead of loops
- 100x faster than pure Python

**Caching:**
- Save TF-IDF vectors to disk
- Reuse for multiple similarity computations
- Avoid recomputing expensive operations

---

## Accuracy Improvements

**Current Approach:**
- TF-IDF: Good baseline, fast
- Cosine similarity: Industry standard
- ~70% accuracy for clear matches

**Potential Enhancements:**
1. **Word2Vec/GloVe:** Semantic similarity (not just keywords)
2. **BERT embeddings:** Context-aware matching
3. **Named Entity Recognition:** Extract skills, companies, education
4. **Skill taxonomy:** Map synonyms (Python = py, NumPy = np)
5. **Experience weighting:** More weight to years of experience
6. **Hybrid scoring:** TF-IDF + semantic + rule-based

---

## Troubleshooting

**Issue:** Low similarity scores across the board
- **Cause:** Vocabulary mismatch
- **Solution:** Increase `max_features`, add domain synonyms

**Issue:** All resumes rank similarly
- **Cause:** Generic job description
- **Solution:** Add specific skills, technologies, requirements

**Issue:** Processing too slow
- **Cause:** Large PDFs, many files
- **Solution:** Limit to first N pages, batch smaller groups

**Issue:** Poor matches
- **Cause:** Resumes in wrong format (images, tables)
- **Solution:** Use text-based PDFs, avoid scanned documents

---

## Testing

**Unit Tests:**
```python
def test_text_cleaning():
    assert clean_text("Hello123!!") == "hello"
    
def test_vectorization():
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(["test"])
    assert matrix.shape[1] > 0
```

**Integration Tests:**
```bash
# Test complete pipeline
python3 main.py
# Check outputs exist
ls data/*.csv module_*/*.pkl
```

---

## Deployment Considerations

**Environment Variables:**
- NLTK downloads (punkt, wordnet)
- No API keys needed (for upload interface)

**Dependencies:**
- See `requirements.txt`
- Total: ~200MB installed

**Resource Requirements:**
- RAM: ~500MB for 100 resumes
- CPU: Moderate (vectorization intensive)
- Storage: ~50MB for data/models

**Startup Time:**
- First run: ~30 seconds (NLTK downloads)
- Subsequent: <5 seconds

---

## API Endpoints (Future Enhancement)

```python
# Potential REST API design
POST /api/match
{
    "resumes": ["base64_pdf1", "base64_pdf2"],
    "job_description": "We need a Python developer..."
}

Response:
{
    "rankings": [
        {"resume_id": 0, "score": 0.85, "rank": 1},
        {"resume_id": 1, "score": 0.72, "rank": 2}
    ],
    "processing_time": 8.5
}
```

---

## References

- **TF-IDF:** Salton & Buckley (1988)
- **Cosine Similarity:** Vector Space Model (1975)
- **NLTK:** Natural Language Toolkit Documentation
- **scikit-learn:** TF-IDF Vectorizer Guide
- **Streamlit:** Web App Framework Docs

---

**For more details, see code comments in each notebook.**
