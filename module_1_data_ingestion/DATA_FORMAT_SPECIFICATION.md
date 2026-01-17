
# Data Format Specification for CVSense Pipeline

## Module 1 Output Format

### Processed Resumes (`data/processed_resumes.csv`)

**Columns:**
- `Category` (optional): The category/field of the resume (e.g., 'Data Science', 'Software Engineering')
- `cleaned_resume`: Cleaned and validated resume text ready for preprocessing
- `is_valid`: Boolean flag indicating if the resume passed quality validation

**Data Quality Standards:**
- Minimum text length: 50 characters
- Minimum alphabetic content ratio: 50%
- Encoding errors removed
- Excessive whitespace normalized

### Processed Job Descriptions (`data/processed_job_descriptions.csv`)

**Columns:**
- `job_id`: Unique identifier for the job posting (e.g., 'JD001')
- `title`: Job title
- `category`: Job category/field
- `description`: Original job description text
- `cleaned_description`: Cleaned job description ready for preprocessing

**Individual Files:** Each job description is also saved as a separate text file in `data/job_descriptions/`

## Expected Input for Module 2 (Text Preprocessing)

Module 2 should:
1. Load `data/processed_resumes.csv` and `data/processed_job_descriptions.csv`
2. Use only rows where `is_valid == True` for resumes
3. Apply text preprocessing to `cleaned_resume` and `cleaned_description` columns
4. Output format should maintain the same structure with additional preprocessed columns

## Data Validation Guidelines

### Why Data Quality Matters:
- **Poor PDF Extraction:** Corrupted characters, formatting issues can reduce matching accuracy
- **Text Quality:** Low-quality text leads to poor feature extraction and inaccurate similarity scores
- **Consistency:** Standardized format ensures all modules work correctly

### Common PDF Extraction Challenges:
1. **Encoding Issues:** Special characters may not extract correctly
2. **Layout Problems:** Multi-column resumes can have scrambled text
3. **Images as Text:** Text in images cannot be extracted without OCR
4. **Tables:** Table formatting often gets lost in extraction

## File Locations

```
CVSense/
├── data/
│   ├── processed_resumes.csv          # Main resume dataset
│   ├── processed_job_descriptions.csv # Main job descriptions dataset
│   ├── resumes/                       # Individual resume files (if any)
│   └── job_descriptions/              # Individual job description files
└── module_1_data_ingestion/
    ├── data_ingestion.ipynb           # Main implementation notebook
    └── data_quality_report.json       # Quality metrics and statistics
```

## Contact

For questions about data format or quality issues, contact the Module 1 owner.
