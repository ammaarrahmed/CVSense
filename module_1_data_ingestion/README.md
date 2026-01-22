# Module 1: Data Ingestion & Google Drive Integration

**Status: ✅ COMPLETE**  
**Updated: January 2026**

---

## Overview

Module 1 handles data ingestion for the CVSense pipeline. The primary method is **Google Drive integration** - allowing users to import resumes directly from a shared Google Drive folder (perfect for Google Forms job applications).

## Features

### 🔗 Google Drive Integration
- Import resumes from a shared Google Drive folder
- Perfect for processing Google Forms job applications
- Supports public folders (anyone with link can view)

### 📄 File Support
- **PDF** - via pdfplumber or PyPDF2
- **DOCX** - via python-docx
- **TXT** - plain text files

### 🛠️ Text Extraction
- Automatic PDF text extraction
- DOCX paragraph extraction
- Multi-page document support
- Error handling with fallbacks

---

## Usage

### In Streamlit App

1. Go to **📁 Google Drive Import** page
2. Paste your Google Drive folder URL
3. Click "Download Resumes from Drive"
4. Add job description(s)
5. Click "Match Resumes to Jobs"
6. View results on Dashboard!

### Programmatic Usage

```python
from module_1_data_ingestion import (
    download_resumes_from_drive,
    process_resume_files,
    extract_text_from_pdf
)

# Download from Google Drive
result = download_resumes_from_drive(
    "https://drive.google.com/drive/folders/YOUR_FOLDER_ID"
)

print(f"Downloaded {result['count']} files")

# Extract text from downloaded files
resumes = process_resume_files(result['files'])

for resume in resumes:
    print(f"{resume['filename']}: {len(resume['text'])} chars")
```

---

## API Reference

### `download_resumes_from_drive(folder_url, output_dir=None, file_types=['.pdf', '.docx', '.txt'])`

Download resume files from a Google Drive folder.

**Args:**
- `folder_url`: Google Drive folder URL or ID
- `output_dir`: Directory to save files (uses temp dir if None)
- `file_types`: List of file extensions to download

**Returns:**
```python
{
    'files': [Path, ...],    # List of downloaded file paths
    'count': int,            # Number of files downloaded
    'errors': [str, ...],    # List of error messages
    'output_dir': Path       # Where files were saved
}
```

### `process_resume_files(files)`

Extract text from a list of resume files.

**Args:**
- `files`: List of file paths

**Returns:**
```python
[
    {
        'filename': 'resume.pdf',
        'path': '/path/to/resume.pdf',
        'text': 'Extracted resume content...',
        'error': None  # or error message
    },
    ...
]
```

### `extract_text_from_pdf(pdf_path)`

Extract text from a PDF file.

### `extract_text_from_docx(docx_path)`

Extract text from a DOCX file.

---

## Google Drive Setup

### For Google Forms Applications

1. **Create a Google Form** with:
   - Name field
   - Email field
   - File upload field for resume (PDF/DOCX)

2. **Configure Form Settings**:
   - Responses → Create spreadsheet
   - File uploads go to a linked Drive folder

3. **Share the Folder**:
   - Open the folder in Google Drive
   - Click "Share"
   - Change to "Anyone with the link can view"
   - Copy the link

4. **Use in CVSense**:
   - Paste the folder link
   - Download and process resumes
   - Get instant rankings!

### URL Formats Supported

All these formats work:
- `https://drive.google.com/drive/folders/FOLDER_ID`
- `https://drive.google.com/drive/folders/FOLDER_ID?usp=sharing`
- `https://drive.google.com/drive/u/0/folders/FOLDER_ID`
- Just the folder ID: `1ABC123xyz...`

---

## Dependencies

```
gdown>=4.7.0        # Google Drive downloads
pdfplumber>=0.9.0   # PDF extraction (primary)
PyPDF2>=3.0.0       # PDF extraction (fallback)
python-docx>=0.8.11 # DOCX extraction
requests>=2.28.0    # HTTP requests
```

---

## Files

- `google_drive.py` - Main module with all functions
- `__init__.py` - Module exports
- `README.md` - This documentation
