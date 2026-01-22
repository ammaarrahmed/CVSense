#!/usr/bin/env python3
"""
CVSense - Web Interface
Interactive Streamlit application for resume screening
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import tempfile
import os
import io
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="CVSense - Resume Screening",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check for gdown availability
try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .resume-card {
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
    }
    </style>
""", unsafe_allow_html=True)

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'


class CVSenseApp:
    """Main application class"""
    
    def __init__(self):
        self.data_dir = DATA_DIR
        # No static data loading - all processing is dynamic
    
    def render_sidebar(self):
        """Render sidebar navigation"""
        st.sidebar.markdown("## 🎯 Navigation")
        
        pages = {
            "🏠 Dashboard": "dashboard",
            "⬆️ Upload & Process": "upload",
            "📁 Google Drive Import": "google_drive",
            "ℹ️ About": "about"
        }
        
        selection = st.sidebar.radio("Go to", list(pages.keys()))
        
        # Session state status
        st.sidebar.markdown("---")
        st.sidebar.markdown("## 📊 Session Status")
        
        if 'processed_data' in st.session_state and st.session_state.processed_data:
            st.sidebar.text("✅ Data Processed")
            results = st.session_state.processed_data
            st.sidebar.metric("Resumes", results['num_resumes'])
            st.sidebar.metric("Jobs", results['num_jobs'])
        else:
            st.sidebar.text("❌ No data processed yet")
            st.sidebar.info("Upload resumes or import from Google Drive to get started")
        
        return pages[selection]
    
    def render_dashboard(self):
        """Render main dashboard"""
        st.markdown('<div class="main-header">📄 CVSense - Intelligent Resume Screening</div>', 
                   unsafe_allow_html=True)
        
        st.markdown("### 🎯 Welcome to CVSense!")
        
        # Check if we have processed data
        if 'processed_data' not in st.session_state or not st.session_state.processed_data:
            st.info("👋 No resumes processed yet. Get started by:")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                ### ⬆️ Upload Files
                Upload resume PDFs and job descriptions directly.
                
                Best for: Small batches, quick testing
                """)
                if st.button("Go to Upload Page", key="go_upload"):
                    st.session_state.nav_to = "upload"
                    st.rerun()
            
            with col2:
                st.markdown("""
                ### 📁 Google Drive Import
                Connect to a Google Drive folder containing resumes from form submissions.
                
                Best for: Batch processing from Google Forms
                """)
                if st.button("Go to Google Drive Import", key="go_gdrive"):
                    st.session_state.nav_to = "google_drive"
                    st.rerun()
            
            st.markdown("---")
            st.markdown("""
            ### 💡 Use Case: Google Forms Job Applications
            
            1. Create a Google Form for job applications with a file upload field
            2. Responses are stored in a Google Drive folder
            3. Share the folder link (anyone with link can view)
            4. Paste the link in CVSense → We download and rank all resumes!
            """)
            return
        
        # Display results if we have processed data
        results = st.session_state.processed_data
        
        st.markdown("### 📊 Processing Results")
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Resumes Processed", results['num_resumes'])
        with col2:
            st.metric("Job Positions", results['num_jobs'])
        with col3:
            avg_score = results['rankings']['Similarity_Score'].mean()
            st.metric("Avg Match Score", f"{avg_score:.1%}")
        with col4:
            high_matches = len(results['rankings'][results['rankings']['Similarity_Score'] > 0.5])
            st.metric("High Matches (>50%)", high_matches)
        
        # Score distribution
        st.markdown("### 📊 Score Distribution")
        fig = px.histogram(
            results['rankings'],
            x='Similarity_Score',
            nbins=20,
            title='Resume Match Score Distribution',
            labels={'Similarity_Score': 'Match Score', 'count': 'Frequency'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        
        # Top matches per job
        st.markdown("### 🏆 Top Matches by Job")
        
        for job_idx in range(results['num_jobs']):
            job_name = f"Job_{job_idx+1}"
            job_info = results['jobs'].iloc[job_idx]
            job_rankings = results['rankings'][results['rankings']['Job'] == job_name].head(5)
            
            with st.expander(f"**{job_info['title']}** - Top {len(job_rankings)} candidates"):
                for _, row in job_rankings.iterrows():
                    resume_id = int(row['Resume_ID'])
                    resume_data = results['resumes'].iloc[resume_id]
                    score = row['Similarity_Score']
                    
                    color = "🟢" if score > 0.5 else "🟡" if score > 0.3 else "🔴"
                    
                    st.markdown(f"""
                    <div class="resume-card">
                        {color} <b>Rank {int(row['Rank'])}</b> - {resume_data.get('filename', f'Resume {resume_id}')} 
                        - Match: <b>{score*100:.1f}%</b>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Download results
        st.markdown("### 💾 Export Results")
        
        csv_data = results['rankings'].to_csv(index=False)
        st.download_button(
            label="📥 Download Rankings CSV",
            data=csv_data,
            file_name=f"resume_rankings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    def render_google_drive(self):
        """Render Google Drive import page"""
        st.markdown("## 📁 Import Resumes from Google Drive")
        
        st.markdown("""
        Import resumes directly from a Google Drive folder. Perfect for processing
        applications submitted through Google Forms!
        
        ### 📋 Requirements
        - The Google Drive folder must be **shared** (anyone with link can view)
        - Supported file types: PDF, DOCX, TXT
        """)
        
        if not GDOWN_AVAILABLE:
            st.error("⚠️ gdown library not installed. Please install it: `pip install gdown`")
            st.code("pip install gdown", language="bash")
            return
        
        # Initialize session state
        if 'gdrive_resumes' not in st.session_state:
            st.session_state.gdrive_resumes = None
        
        # Google Drive URL input
        st.markdown("### 🔗 Step 1: Enter Google Drive Folder Link")
        
        drive_url = st.text_input(
            "Google Drive Folder URL",
            placeholder="https://drive.google.com/drive/folders/YOUR_FOLDER_ID",
            help="Paste the shared folder link from Google Drive"
        )
        
        if st.button("📥 Download Resumes from Drive", type="primary"):
            if not drive_url:
                st.error("Please enter a Google Drive folder URL")
                return
            
            with st.spinner("⏳ Downloading resumes from Google Drive..."):
                try:
                    from module_1_data_ingestion.google_drive import (
                        download_resumes_from_drive,
                        process_resume_files
                    )
                    
                    # Download files
                    result = download_resumes_from_drive(drive_url)
                    
                    if result['errors']:
                        for error in result['errors']:
                            st.warning(f"⚠️ {error}")
                    
                    if result['count'] == 0:
                        st.error("❌ No resume files found in the folder. Make sure the folder contains PDF, DOCX, or TXT files.")
                        return
                    
                    st.success(f"✅ Downloaded {result['count']} resume file(s)")
                    
                    # Extract text
                    with st.spinner("📄 Extracting text from resumes..."):
                        resumes = process_resume_files(result['files'])
                        
                        # Filter successful extractions
                        valid_resumes = [r for r in resumes if r['text'] and not r['error']]
                        failed = [r for r in resumes if r['error']]
                        
                        if failed:
                            with st.expander(f"⚠️ {len(failed)} file(s) could not be processed"):
                                for r in failed:
                                    st.text(f"- {r['filename']}: {r['error']}")
                        
                        st.session_state.gdrive_resumes = valid_resumes
                        st.success(f"✅ Successfully extracted text from {len(valid_resumes)} resume(s)")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.exception(e)
        
        # Show downloaded resumes and job input
        if st.session_state.gdrive_resumes:
            resumes = st.session_state.gdrive_resumes
            
            st.markdown("---")
            st.markdown(f"### 📄 {len(resumes)} Resumes Ready for Processing")
            
            with st.expander("View downloaded resumes"):
                for i, resume in enumerate(resumes):
                    st.markdown(f"**{i+1}. {resume['filename']}**")
                    st.text_area("Preview", resume['text'][:500] + "...", height=100, key=f"gdrive_preview_{i}")
            
            # Job description input
            st.markdown("### 💼 Step 2: Add Job Description(s)")
            
            num_jobs = st.number_input("Number of job positions", min_value=1, max_value=10, value=1)
            
            job_descriptions = []
            for i in range(num_jobs):
                with st.expander(f"Job Description {i+1}", expanded=(i==0)):
                    job_title = st.text_input(f"Job Title", key=f"gdrive_title_{i}")
                    job_category = st.text_input(f"Category", value="General", key=f"gdrive_cat_{i}")
                    job_desc = st.text_area(f"Description", height=150, key=f"gdrive_desc_{i}")
                    
                    if job_title and job_desc:
                        job_descriptions.append({
                            'title': job_title,
                            'category': job_category,
                            'description': job_desc
                        })
            
            # Process button
            st.markdown("### 🚀 Step 3: Process & Rank")
            
            if st.button("🎯 Match Resumes to Jobs", type="primary"):
                if not job_descriptions:
                    st.error("Please add at least one job description")
                    return
                
                with st.spinner("⏳ Processing and ranking resumes..."):
                    try:
                        results = self.process_gdrive_resumes(resumes, job_descriptions)
                        st.session_state.processed_data = results
                        st.success("✅ Processing complete! View results on the Dashboard.")
                        st.balloons()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.exception(e)
    
    def process_gdrive_resumes(self, resumes, job_descriptions):
        """Process Google Drive resumes and rank against jobs"""
        from module_2_text_preprocessing.preprocessing import clean_text, preprocess_text
        from module_3_feature_extraction.tfidf import create_tfidf_vectors
        from module_4_similarity_ranking.ranking import compute_hybrid_scores, rank_resumes
        
        # Create DataFrames
        resumes_df = pd.DataFrame(resumes)
        resumes_df['cleaned_text'] = resumes_df['text'].apply(clean_text)
        resumes_df['preprocessed_text'] = resumes_df['cleaned_text'].apply(
            lambda x: preprocess_text(x, use_nltk=False)
        )
        
        jobs_df = pd.DataFrame(job_descriptions)
        jobs_df['job_id'] = [f"JOB_{i+1}" for i in range(len(jobs_df))]
        jobs_df['cleaned_text'] = jobs_df['description'].apply(clean_text)
        jobs_df['preprocessed_text'] = jobs_df['cleaned_text'].apply(
            lambda x: preprocess_text(x, use_nltk=False)
        )
        
        # Get text lists
        resume_texts = resumes_df['preprocessed_text'].tolist()
        job_texts = jobs_df['preprocessed_text'].tolist()
        
        # TF-IDF
        tfidf_result = create_tfidf_vectors(
            resume_texts, job_texts,
            max_features=5000, ngram_range=(1, 2), use_idf=False
        )
        
        # Compute scores
        similarity_matrix = compute_hybrid_scores(
            job_texts=job_texts,
            resume_texts=resume_texts,
            job_vectors=tfidf_result['jd_vectors'],
            resume_vectors=tfidf_result['resume_vectors'],
            keyword_weight=0.7,
            tfidf_weight=0.3
        )
        
        # Rank
        rankings_df = rank_resumes(
            similarity_matrix=similarity_matrix,
            top_n=min(10, len(resume_texts))
        )
        
        return {
            'resumes': resumes_df,
            'jobs': jobs_df,
            'rankings': rankings_df,
            'num_resumes': len(resumes_df),
            'num_jobs': len(jobs_df)
        }
    
    def render_upload(self):
        """Render upload and processing page"""
        st.markdown("## ⬆️ Upload & Process Resumes")
        
        st.markdown("""
        Upload your own resumes and job descriptions to get instant matching results!
        No need to run Module 1 or the command-line pipeline.
        """)
        
        # Initialize session state
        if 'processed_data' not in st.session_state:
            st.session_state.processed_data = None
        
        # Upload section
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📄 Upload Resumes")
            uploaded_resumes = st.file_uploader(
                "Upload resume files (PDF or TXT)",
                type=['pdf', 'txt'],
                accept_multiple_files=True,
                help="Upload multiple resume files at once"
            )
            
            if uploaded_resumes:
                st.success(f"✅ {len(uploaded_resumes)} resume(s) uploaded")
        
        with col2:
            st.markdown("### 💼 Add Job Descriptions")
            job_input_method = st.radio(
                "Choose input method:",
                ["Text Input", "Upload File"]
            )
            
            job_descriptions = []
            
            if job_input_method == "Text Input":
                num_jobs = st.number_input("Number of job descriptions", min_value=1, max_value=20, value=1)
                
                for i in range(num_jobs):
                    with st.expander(f"Job Description {i+1}"):
                        job_title = st.text_input(f"Job Title {i+1}", key=f"title_{i}")
                        job_category = st.text_input(f"Category {i+1}", key=f"category_{i}", value="General")
                        job_desc = st.text_area(f"Description {i+1}", key=f"desc_{i}", height=150)
                        
                        if job_title and job_desc:
                            job_descriptions.append({
                                'title': job_title,
                                'category': job_category,
                                'description': job_desc
                            })
            else:
                uploaded_jobs = st.file_uploader(
                    "Upload job description files (TXT)",
                    type=['txt'],
                    accept_multiple_files=True
                )
                
                if uploaded_jobs:
                    for idx, job_file in enumerate(uploaded_jobs):
                        content = job_file.read().decode('utf-8')
                        job_descriptions.append({
                            'title': job_file.name.replace('.txt', ''),
                            'category': 'Uploaded',
                            'description': content
                        })
                    st.success(f"✅ {len(uploaded_jobs)} job description(s) uploaded")
        
        # Process button
        st.markdown("---")
        
        if st.button("🚀 Process & Match Resumes", type="primary", width='stretch'):
            if not uploaded_resumes:
                st.error("❌ Please upload at least one resume")
                return
            
            if not job_descriptions:
                st.error("❌ Please add at least one job description")
                return
            
            # Process the uploads
            with st.spinner("⏳ Processing resumes and matching with jobs..."):
                try:
                    results = self.process_uploads(uploaded_resumes, job_descriptions)
                    st.session_state.processed_data = results
                    st.success("✅ Processing complete!")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Error during processing: {str(e)}")
                    st.exception(e)
                    return
        
        # Display results
        if st.session_state.processed_data:
            st.markdown("---")
            st.markdown("## 📊 Results")
            
            results = st.session_state.processed_data
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Resumes Processed", results['num_resumes'])
            with col2:
                st.metric("Job Descriptions", results['num_jobs'])
            with col3:
                st.metric("Total Matches", len(results['rankings']))
            with col4:
                avg_score = results['rankings']['Similarity_Score'].mean()
                st.metric("Avg Match Score", f"{avg_score:.1%}")
            
            # Rankings by job
            st.markdown("### 🎯 Top Matches by Job")
            
            for job_idx in range(results['num_jobs']):
                job_name = f"Job_{job_idx+1}"
                job_info = results['jobs'].iloc[job_idx]
                job_rankings = results['rankings'][results['rankings']['Job'] == job_name].head(5)
                
                with st.expander(f"**{job_info['title']}** ({job_info['category']}) - {len(job_rankings)} matches"):
                    st.markdown(f"**Description:** {job_info['description'][:200]}...")
                    
                    st.markdown("**Top Matches:**")
                    
                    for _, row in job_rankings.iterrows():
                        resume_id = int(row['Resume_ID'])
                        resume_data = results['resumes'].iloc[resume_id]
                        score = row['Similarity_Score']
                        
                        color = "🟢" if score > 0.6 else "🟡" if score > 0.4 else "🔴"
                        
                        st.markdown(f"""
                        <div class="resume-card">
                            {color} <b>Rank {int(row['Rank'])}</b> - {resume_data.get('filename', f'Resume {resume_id}')} 
                            - Match: <b>{score*100:.1f}%</b>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        with st.expander(f"View Resume {resume_id}"):
                            st.text_area("Resume Content", resume_data['text'][:1000] + "...", height=200, key=f"view_{job_idx}_{resume_id}")
            
            # Download results
            st.markdown("### 💾 Export Results")
            
            csv_data = results['rankings'].to_csv(index=False)
            st.download_button(
                label="📥 Download Rankings CSV",
                data=csv_data,
                file_name=f"resume_rankings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    def process_uploads(self, resume_files, job_descriptions):
        """Process uploaded resumes and job descriptions using modular pipeline"""
        
        # Import modules
        from module_2_text_preprocessing.preprocessing import clean_text, preprocess_text
        from module_3_feature_extraction.tfidf import create_tfidf_vectors
        from module_4_similarity_ranking.ranking import compute_hybrid_scores, rank_resumes
        
        # Import PDF processing
        try:
            import pdfplumber
            pdf_library = 'pdfplumber'
        except ImportError:
            try:
                import PyPDF2
                pdf_library = 'PyPDF2'
            except ImportError:
                raise Exception("Please install pdfplumber or PyPDF2: pip install pdfplumber")
        
        # Extract text from resumes
        resumes_data = []
        
        for resume_file in resume_files:
            filename = resume_file.name
            
            try:
                if filename.endswith('.pdf'):
                    text = self.extract_pdf_text(resume_file, pdf_library)
                else:
                    text = resume_file.read().decode('utf-8', errors='ignore')
                
                resumes_data.append({
                    'filename': filename,
                    'text': text
                })
            except Exception as e:
                st.warning(f"⚠️ Error processing {filename}: {str(e)}")
        
        if not resumes_data:
            raise Exception("No resumes could be processed successfully")
        
        # Create DataFrames
        resumes_df = pd.DataFrame(resumes_data)
        jobs_df = pd.DataFrame(job_descriptions)
        jobs_df['job_id'] = [f"JOB_{i+1}" for i in range(len(jobs_df))]
        
        # Preprocess using Module 2
        resumes_df['cleaned_text'] = resumes_df['text'].apply(clean_text)
        resumes_df['preprocessed_text'] = resumes_df['cleaned_text'].apply(
            lambda x: preprocess_text(x, use_nltk=False)
        )
        
        jobs_df['cleaned_text'] = jobs_df['description'].apply(clean_text)
        jobs_df['preprocessed_text'] = jobs_df['cleaned_text'].apply(
            lambda x: preprocess_text(x, use_nltk=False)
        )
        
        # Get text lists
        resume_texts = resumes_df['preprocessed_text'].tolist()
        job_texts = jobs_df['preprocessed_text'].tolist()
        
        # TF-IDF using Module 3 (disable IDF for small corpus)
        tfidf_result = create_tfidf_vectors(
            resume_texts,
            job_texts,
            max_features=5000,
            ngram_range=(1, 2),
            use_idf=False  # Better for small corpus
        )
        
        resume_vectors = tfidf_result['resume_vectors']
        job_vectors = tfidf_result['jd_vectors']
        
        # Compute hybrid scores using Module 4
        similarity_matrix = compute_hybrid_scores(
            job_texts=job_texts,
            resume_texts=resume_texts,
            job_vectors=job_vectors,
            resume_vectors=resume_vectors,
            keyword_weight=0.7,
            tfidf_weight=0.3
        )
        
        # Rank resumes using Module 4
        rankings_df = rank_resumes(
            similarity_matrix=similarity_matrix,
            top_n=min(5, len(resume_texts))
        )
        
        return {
            'resumes': resumes_df,
            'jobs': jobs_df,
            'rankings': rankings_df,
            'num_resumes': len(resumes_df),
            'num_jobs': len(jobs_df)
        }
    
    def extract_pdf_text(self, pdf_file, library='pdfplumber'):
        """Extract text from PDF file"""
        text = ""
        
        if library == 'pdfplumber':
            import pdfplumber
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        else:
            import PyPDF2
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        
        return text.strip()
    
    def render_about(self):
        """Render about page"""
        st.markdown("## ℹ️ About CVSense")
        
        st.markdown("""
        ### 🎯 Intelligent Resume Screening System
        
        CVSense is an automated resume screening system that uses Natural Language Processing (NLP) 
        and Machine Learning to match resumes with job descriptions.
        
        #### 📊 How It Works
        
        1. **Import Resumes**
           - Upload PDF/DOCX files directly, OR
           - Import from Google Drive folder (perfect for Google Forms submissions!)
        
        2. **Text Processing**
           - Extracts text from documents
           - Cleans and normalizes text
           - Tokenization and preprocessing
        
        3. **Feature Extraction**
           - TF-IDF vectorization with n-grams
           - Keyword extraction with synonym matching
           - Technical phrase detection
        
        4. **Hybrid Matching**
           - Keyword overlap scoring (like Jobscan)
           - TF-IDF cosine similarity
           - Phrase matching for technical terms
           - Synonym/abbreviation expansion (ML ↔ machine learning)
        
        #### 🛠️ Technologies Used
        
        - **Python**: Core programming language
        - **scikit-learn**: TF-IDF vectorization, cosine similarity
        - **NLTK**: Natural language processing
        - **gdown**: Google Drive integration
        - **pdfplumber**: PDF text extraction
        - **Streamlit**: Web interface
        - **Plotly**: Interactive visualizations
        
        #### 💡 Use Case: Google Forms Job Applications
        
        1. Create a Google Form with file upload for resumes
        2. Form responses go to a Google Drive folder
        3. Share the folder link
        4. Paste link in CVSense → Instant ranking!
        
        ---
        
        **Version**: 2.0.0  
        **Project**: CVSense - Intelligent Resume Screening
        """)


def main():
    """Main application entry point"""
    app = CVSenseApp()
    
    # Check for navigation override
    if 'nav_to' in st.session_state:
        page = st.session_state.nav_to
        del st.session_state.nav_to
    else:
        # Render sidebar and get selected page
        page = app.render_sidebar()
    
    # Render selected page
    if page == "dashboard":
        app.render_dashboard()
    elif page == "upload":
        app.render_upload()
    elif page == "google_drive":
        app.render_google_drive()
    elif page == "about":
        app.render_about()


if __name__ == "__main__":
    main()
