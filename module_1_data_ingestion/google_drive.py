"""
Module 1: Google Drive Integration
Download resumes from a shared Google Drive folder.

Works with public or shared folders (anyone with the link can view).
"""

import os
import re
import tempfile
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import requests


def extract_folder_id(drive_url: str) -> Optional[str]:
    """
    Extract Google Drive folder ID from various URL formats.
    
    Supports:
    - https://drive.google.com/drive/folders/FOLDER_ID
    - https://drive.google.com/drive/folders/FOLDER_ID?usp=sharing
    - https://drive.google.com/drive/u/0/folders/FOLDER_ID
    - Just the folder ID itself
    
    Args:
        drive_url: Google Drive folder URL or ID
        
    Returns:
        Folder ID string or None if not found
    """
    # If it's already just an ID (no slashes or dots)
    if re.match(r'^[\w-]+$', drive_url) and len(drive_url) > 20:
        return drive_url
    
    # Extract from various URL formats
    patterns = [
        r'drive\.google\.com/drive/(?:u/\d+/)?folders/([a-zA-Z0-9_-]+)',
        r'drive\.google\.com/folderview\?id=([a-zA-Z0-9_-]+)',
        r'id=([a-zA-Z0-9_-]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, drive_url)
        if match:
            return match.group(1)
    
    return None


def extract_file_id(drive_url: str) -> Optional[str]:
    """
    Extract Google Drive file ID from various URL formats.
    
    Args:
        drive_url: Google Drive file URL or ID
        
    Returns:
        File ID string or None if not found
    """
    # If it's already just an ID
    if re.match(r'^[\w-]+$', drive_url) and len(drive_url) > 20:
        return drive_url
    
    patterns = [
        r'drive\.google\.com/file/d/([a-zA-Z0-9_-]+)',
        r'drive\.google\.com/open\?id=([a-zA-Z0-9_-]+)',
        r'id=([a-zA-Z0-9_-]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, drive_url)
        if match:
            return match.group(1)
    
    return None


def get_direct_download_url(file_id: str) -> str:
    """Get direct download URL for a Google Drive file."""
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def download_file(file_id: str, filename: str, output_dir: Path) -> Optional[Path]:
    """
    Download a single file from Google Drive.
    
    Args:
        file_id: Google Drive file ID
        filename: Name to save the file as
        output_dir: Directory to save the file
        
    Returns:
        Path to downloaded file or None if failed
    """
    url = get_direct_download_url(file_id)
    
    try:
        # First request - may get confirmation page for large files
        session = requests.Session()
        response = session.get(url, stream=True, timeout=30)
        
        # Check for virus scan warning (large files)
        if 'confirm' in response.text[:1000]:
            # Extract confirm token
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    url = f"{url}&confirm={value}"
                    response = session.get(url, stream=True, timeout=30)
                    break
        
        # Save file
        output_path = output_dir / filename
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
        
        # Verify it's not an error page
        if output_path.stat().st_size < 100:
            with open(output_path, 'r', errors='ignore') as f:
                content = f.read()
                if 'html' in content.lower() or 'error' in content.lower():
                    output_path.unlink()
                    return None
        
        return output_path
        
    except Exception as e:
        print(f"Error downloading {filename}: {e}")
        return None


def list_folder_files_api(folder_id: str, api_key: Optional[str] = None) -> List[Dict]:
    """
    List files in a Google Drive folder using the API.
    
    Note: Requires either:
    1. API key (for public folders)
    2. Folder to be publicly accessible
    
    Args:
        folder_id: Google Drive folder ID
        api_key: Optional Google API key
        
    Returns:
        List of file info dicts with 'id', 'name', 'mimeType'
    """
    if api_key:
        url = f"https://www.googleapis.com/drive/v3/files"
        params = {
            'q': f"'{folder_id}' in parents",
            'key': api_key,
            'fields': 'files(id,name,mimeType)',
            'pageSize': 100
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response.json().get('files', [])
        except Exception as e:
            print(f"API error: {e}")
    
    return []


def download_folder_with_gdown(folder_url: str, output_dir: Path) -> Tuple[List[Path], List[str]]:
    """
    Download all files from a Google Drive folder using gdown.
    
    This is the most reliable method for public folders.
    
    Args:
        folder_url: Google Drive folder URL
        output_dir: Directory to save files
        
    Returns:
        Tuple of (list of downloaded file paths, list of error messages)
    """
    try:
        import gdown
    except ImportError:
        return [], ["gdown not installed. Run: pip install gdown"]
    
    folder_id = extract_folder_id(folder_url)
    if not folder_id:
        return [], ["Could not extract folder ID from URL"]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_files = []
    errors = []
    
    try:
        # Download entire folder
        url = f"https://drive.google.com/drive/folders/{folder_id}"
        gdown.download_folder(url, output=str(output_dir), quiet=False, use_cookies=False)
        
        # Find downloaded files
        for file_path in output_dir.iterdir():
            if file_path.is_file():
                downloaded_files.append(file_path)
                
    except Exception as e:
        errors.append(f"Error downloading folder: {str(e)}")
    
    return downloaded_files, errors


def download_resumes_from_drive(
    folder_url: str,
    output_dir: Optional[Path] = None,
    file_types: List[str] = ['.pdf', '.docx', '.doc', '.txt']
) -> Dict:
    """
    Download resume files from a Google Drive folder.
    
    Main entry point for the Google Drive integration.
    
    Args:
        folder_url: Google Drive folder URL or ID
        output_dir: Directory to save files (uses temp dir if None)
        file_types: List of file extensions to download
        
    Returns:
        Dict with:
        - 'files': List of downloaded file paths
        - 'count': Number of files downloaded
        - 'errors': List of error messages
        - 'output_dir': Path where files were saved
    """
    # Create output directory
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix='cvsense_resumes_'))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Try gdown first (most reliable for public folders)
    downloaded_files, errors = download_folder_with_gdown(folder_url, output_dir)
    
    # Filter by file type
    resume_files = [
        f for f in downloaded_files 
        if f.suffix.lower() in file_types
    ]
    
    return {
        'files': resume_files,
        'count': len(resume_files),
        'errors': errors,
        'output_dir': output_dir
    }


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text string
    """
    text = ""
    
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except ImportError:
        try:
            import PyPDF2
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except ImportError:
            raise ImportError("Install pdfplumber or PyPDF2: pip install pdfplumber")
    
    return text.strip()


def extract_text_from_docx(docx_path: Path) -> str:
    """
    Extract text from a DOCX file.
    
    Args:
        docx_path: Path to DOCX file
        
    Returns:
        Extracted text string
    """
    try:
        from docx import Document
        doc = Document(docx_path)
        return "\n".join([para.text for para in doc.paragraphs])
    except ImportError:
        raise ImportError("Install python-docx: pip install python-docx")


def process_resume_files(files: List[Path]) -> List[Dict]:
    """
    Extract text from a list of resume files.
    
    Args:
        files: List of file paths
        
    Returns:
        List of dicts with 'filename', 'text', 'error' keys
    """
    results = []
    
    for file_path in files:
        result = {
            'filename': file_path.name,
            'path': str(file_path),
            'text': '',
            'error': None
        }
        
        try:
            suffix = file_path.suffix.lower()
            
            if suffix == '.pdf':
                result['text'] = extract_text_from_pdf(file_path)
            elif suffix in ['.docx', '.doc']:
                result['text'] = extract_text_from_docx(file_path)
            elif suffix == '.txt':
                result['text'] = file_path.read_text(encoding='utf-8', errors='ignore')
            else:
                result['error'] = f"Unsupported file type: {suffix}"
                
        except Exception as e:
            result['error'] = str(e)
        
        results.append(result)
    
    return results
