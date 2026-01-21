#!/usr/bin/env python3
"""
CVSense - Web Interface
Interactive Streamlit application for resume screening
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
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
        self.preprocessed_dir = self.data_dir / 'linguistically_preprocessed_files'
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load all necessary data files"""
        try:
            # Raw data - try processed files first, fallback to preprocessed
            if (self.data_dir / 'processed_resumes.csv').exists():
                self.raw_resumes = pd.read_csv(self.data_dir / 'processed_resumes.csv')
                self.raw_jobs = pd.read_csv(self.data_dir / 'processed_job_descriptions.csv')
            elif (self.preprocessed_dir / 'preprocessed_resumes_final.csv').exists():
                # Use preprocessed files as raw data if processed files don't exist
                self.raw_resumes = pd.read_csv(self.preprocessed_dir / 'preprocessed_resumes_final.csv')
                self.raw_jobs = pd.read_csv(self.preprocessed_dir / 'preprocessed_jobs_final.csv')
            else:
                self.raw_resumes = None
                self.raw_jobs = None
            
            # Preprocessed data
            if (self.preprocessed_dir / 'preprocessed_resumes_final.csv').exists():
                self.preprocessed_resumes = pd.read_csv(self.preprocessed_dir / 'preprocessed_resumes_final.csv')
                self.preprocessed_jobs = pd.read_csv(self.preprocessed_dir / 'preprocessed_jobs_final.csv')
            else:
                self.preprocessed_resumes = None
                self.preprocessed_jobs = None
            
            # Rankings
            rankings_path = PROJECT_ROOT / 'module_4_similarity_ranking' / 'module5_resume_ranking.csv'
            if rankings_path.exists():
                self.rankings = pd.read_csv(rankings_path)
            else:
                self.rankings = None
            
            # TF-IDF vectors
            vectors_path = PROJECT_ROOT / 'module_3_feature_extraction' / 'tfidf_vectors.pkl'
            if vectors_path.exists():
                with open(vectors_path, 'rb') as f:
                    self.tfidf_data = pickle.load(f)
            else:
                self.tfidf_data = None
                
        except Exception as e:
            st.warning(f"Some data files are missing. Upload & Process page will be available for custom processing.")
            self.raw_resumes = None
            self.raw_jobs = None
            self.preprocessed_resumes = None
            self.preprocessed_jobs = None
            self.rankings = None
            self.tfidf_data = None
    
    def render_sidebar(self):
        """Render sidebar navigation"""
        st.sidebar.markdown("## 🎯 Navigation")
        
        pages = {
            "🏠 Dashboard": "dashboard",
            "⬆️ Upload & Process": "upload",
            "🔍 Job Search": "job_search",
            "📊 Analytics": "analytics",
            "📄 Resume Explorer": "resume_explorer",
            "ℹ️ About": "about"
        }
        
        selection = st.sidebar.radio("Go to", list(pages.keys()))
        
        # System status
        st.sidebar.markdown("---")
        st.sidebar.markdown("## 📊 System Status")
        
        status = {
            "Module 1": self.raw_resumes is not None,
            "Module 2": self.preprocessed_resumes is not None,
            "Module 3": self.tfidf_data is not None,
            "Module 4": self.rankings is not None
        }
        
        for module, is_complete in status.items():
            icon = "✅" if is_complete else "❌"
            st.sidebar.text(f"{icon} {module}")
        
        # Stats
        if self.raw_resumes is not None:
            st.sidebar.markdown("---")
            st.sidebar.markdown("## 📈 Dataset Stats")
            st.sidebar.metric("Total Resumes", len(self.raw_resumes))
            if self.raw_jobs is not None:
                st.sidebar.metric("Job Descriptions", len(self.raw_jobs))
            
            if self.rankings is not None:
                st.sidebar.metric("Total Rankings", len(self.rankings))
        
        return pages[selection]
    
    def render_dashboard(self):
        """Render main dashboard"""
        st.markdown('<div class="main-header">📄 CVSense - Intelligent Resume Screening</div>', 
                   unsafe_allow_html=True)
        
        st.markdown("### 🎯 Quick Overview")
        
        if self.rankings is None:
            st.info("👋 Welcome to CVSense! No pre-processed data available.")
            st.markdown("""
            **Get Started:**
            - Use the **⬆️ Upload & Process** page to upload resumes and job descriptions
            - The system will automatically process and rank candidates
            - Or run the full pipeline locally: `python main.py`
            """)
            return
        
        # Metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Resumes", len(self.raw_resumes) if self.raw_resumes is not None else 0)
        with col2:
            st.metric("Job Positions", len(self.raw_jobs) if self.raw_jobs is not None else 0)
        with col3:
            avg_score = self.rankings['Similarity_Score'].mean()
            st.metric("Avg Match Score", f"{avg_score:.2%}")
        with col4:
            high_matches = len(self.rankings[self.rankings['Similarity_Score'] > 0.5])
            st.metric("High Matches (>50%)", high_matches)
        
        # Score distribution
        st.markdown("### 📊 Score Distribution")
        fig = px.histogram(
            self.rankings,
            x='Similarity_Score',
            nbins=30,
            title='Resume Similarity Score Distribution',
            labels={'Similarity_Score': 'Similarity Score', 'count': 'Frequency'}
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width='stretch')
        
        # Top matches per job
        st.markdown("### 🏆 Top Matches by Job")
        
        job_options = sorted(self.rankings['Job'].unique())
        selected_job = st.selectbox("Select Job Description", job_options)
        
        job_rankings = self.rankings[self.rankings['Job'] == selected_job].head(10)
        
        # Get job details
        if selected_job:
            job_idx = int(selected_job.split('_')[1]) - 1
        else:
            job_idx = 0
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### 📋 Job Details")
            if self.raw_jobs is not None and job_idx < len(self.raw_jobs):
                job_info = self.raw_jobs.iloc[job_idx]
                st.markdown(f"**Title:** {job_info['title']}")
                st.markdown(f"**Category:** {job_info['category']}")
            else:
                st.markdown(f"**Job:** {selected_job}")
        
        with col2:
            st.markdown("#### 🎯 Top Candidates")
            for _, row in job_rankings.iterrows():
                score_pct = row['Similarity_Score'] * 100
                color = "🟢" if score_pct > 60 else "🟡" if score_pct > 40 else "🔴"
                
                resume_id = int(row['Resume_ID'])
                if self.raw_resumes is not None and resume_id < len(self.raw_resumes):
                    resume_category = self.raw_resumes.iloc[resume_id].get('Category', 'N/A')
                else:
                    resume_category = "N/A"
                
                st.markdown(f"""
                <div class="resume-card">
                    {color} <b>Rank {int(row['Rank'])}</b> - Resume #{resume_id} 
                    ({resume_category}) - Match: <b>{score_pct:.1f}%</b>
                </div>
                """, unsafe_allow_html=True)
    
    def render_job_search(self):
        """Render job search interface"""
        st.markdown("## 🔍 Resume Search by Job Description")
        
        if self.rankings is None:
            st.warning("⚠️ Rankings not available. Run the pipeline first: `python main.py`")
            return
        
        # Job selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            max_jobs = len(self.raw_jobs) if self.raw_jobs is not None else 10
            job_idx = st.number_input(
                "Select Job ID (1-10)",
                min_value=1,
                max_value=max_jobs,
                value=1
            )
        
        with col2:
            top_n = st.slider("Number of Results", 3, 20, 5)
        
        # Get job info
        if self.raw_jobs is not None and job_idx <= len(self.raw_jobs):
            job_info = self.raw_jobs.iloc[job_idx - 1]
        else:
            job_info = None
        
        # Display job description
        st.markdown("### 📋 Job Description")
        
        if job_info is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Title:** {job_info['title']}")
                st.markdown(f"**Category:** {job_info['category']}")
            
            with col2:
                st.markdown(f"**ID:** {job_info['job_id']}")
            
            with st.expander("📄 View Full Description"):
                st.text(job_info['description'])
        else:
            st.warning("Job information not available.")
        
        # Get rankings
        job_name = f"Job_{job_idx}"
        job_rankings = self.rankings[self.rankings['Job'] == job_name].head(top_n)
        
        st.markdown(f"### 🎯 Top {top_n} Matching Resumes")
        
        for _, row in job_rankings.iterrows():
            resume_id = int(row['Resume_ID'])
            if self.raw_resumes is not None and resume_id < len(self.raw_resumes):
                resume_data = self.raw_resumes.iloc[resume_id]
            else:
                resume_data = pd.Series({'Category': 'N/A', 'ID': resume_id, 'cleaned_resume': 'N/A', 'Resume_str': 'N/A'})
            
            score = row['Similarity_Score']
            
            # Score color
            if score > 0.6:
                score_color = "success"
            elif score > 0.4:
                score_color = "warning"
            else:
                score_color = "error"
            
            # Resume card
            with st.expander(f"#{int(row['Rank'])} - Resume {resume_id} - {resume_data.get('Category', 'N/A')} - {score:.1%} Match"):
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.metric("Match Score", f"{score:.1%}")
                    st.markdown(f"**Category:** {resume_data.get('Category', 'N/A')}")
                    st.markdown(f"**Resume ID:** {resume_data.get('ID', resume_id)}")
                
                with col2:
                    st.markdown("**Resume Text (Preview):**")
                    preview_text = str(resume_data.get('cleaned_resume', str(resume_data.get('Resume_str', 'N/A'))))[:500] + "..."
                    st.text_area("", preview_text, height=150, key=f"resume_{resume_id}")
    
    def render_analytics(self):
        """Render analytics dashboard"""
        st.markdown("## 📊 System Analytics")
        
        if self.rankings is None:
            st.warning("⚠️ Analytics not available. Run the pipeline first.")
            return
        
        # Overall statistics
        st.markdown("### 📈 Overall Performance")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Mean Similarity", f"{self.rankings['Similarity_Score'].mean():.2%}")
        with col2:
            st.metric("Median Similarity", f"{self.rankings['Similarity_Score'].median():.2%}")
        with col3:
            st.metric("Max Similarity", f"{self.rankings['Similarity_Score'].max():.2%}")
        
        # Score distribution by job
        st.markdown("### 📊 Score Distribution by Job")
        
        fig = px.box(
            self.rankings,
            x='Job',
            y='Similarity_Score',
            title='Similarity Score Distribution Across Jobs',
            labels={'Similarity_Score': 'Similarity Score', 'Job': 'Job Description'}
        )
        st.plotly_chart(fig, width='stretch')
        
        # Top performers
        st.markdown("### 🏆 Top Performing Resumes")
        
        # Get average score per resume
        resume_avg_scores = self.rankings.groupby('Resume_ID')['Similarity_Score'].agg(['mean', 'max', 'count']).reset_index()
        resume_avg_scores = resume_avg_scores.sort_values('mean', ascending=False).head(10)
        
        # Add category information
        if self.raw_resumes is not None:
            def get_category(x):
                idx = int(x)
                if self.raw_resumes is not None and idx < len(self.raw_resumes):
                    return self.raw_resumes.iloc[idx].get('Category', 'N/A')
                return 'N/A'
            resume_avg_scores['Category'] = resume_avg_scores['Resume_ID'].apply(get_category)
        else:
            resume_avg_scores['Category'] = 'N/A'
        
        fig = px.bar(
            resume_avg_scores,
            x='Resume_ID',
            y='mean',
            title='Top 10 Resumes by Average Match Score',
            labels={'mean': 'Average Score', 'Resume_ID': 'Resume ID'},
            color='Category'
        )
        st.plotly_chart(fig, width='stretch')
        
        # Category analysis
        if self.raw_resumes is not None and 'Category' in self.raw_resumes.columns:
            st.markdown("### 📂 Performance by Resume Category")
            
            # Merge rankings with resume categories
            rankings_with_cat = self.rankings.copy()
            def get_category_rank(x):
                idx = int(x)
                if self.raw_resumes is not None and idx < len(self.raw_resumes):
                    return self.raw_resumes.iloc[idx].get('Category', 'Unknown')
                return 'Unknown'
            rankings_with_cat['Category'] = rankings_with_cat['Resume_ID'].apply(get_category_rank)
            
            category_stats = rankings_with_cat.groupby('Category')['Similarity_Score'].agg(['mean', 'median', 'count']).reset_index()
            category_stats = category_stats.sort_values('mean', ascending=False)
            
            fig = px.bar(
                category_stats,
                x='Category',
                y='mean',
                title='Average Similarity Score by Resume Category',
                labels={'mean': 'Average Score', 'Category': 'Resume Category'},
                color='mean',
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, width='stretch')
    
    def render_resume_explorer(self):
        """Render resume explorer"""
        st.markdown("## 📄 Resume Explorer")
        
        if self.raw_resumes is None or len(self.raw_resumes) == 0:
            st.info("No resumes available. Use the Upload & Process page to add resumes.")
            return
        
        # Resume selection
        resume_id = st.number_input(
            "Select Resume ID",
            min_value=0,
            max_value=len(self.raw_resumes) - 1,
            value=0
        )
        
        resume_data = self.raw_resumes.iloc[resume_id]
        
        # Resume details
        st.markdown("### 📋 Resume Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Resume ID", resume_data.get('ID', resume_id))
        with col2:
            st.metric("Category", resume_data.get('Category', 'N/A'))
        with col3:
            is_valid = "✅ Valid" if resume_data.get('is_valid', True) else "❌ Invalid"
            st.markdown(f"**Status:** {is_valid}")
        
        # Resume text
        st.markdown("### 📄 Resume Content")
        
        tab1, tab2, tab3 = st.tabs(["Original", "Cleaned", "Preprocessed"])
        
        with tab1:
            st.text_area("Original Resume", resume_data.get('Resume_str', resume_data.get('cleaned_resume', 'N/A')), height=300)
        
        with tab2:
            st.text_area("Cleaned Resume", resume_data.get('cleaned_resume', 'N/A'), height=300)
        
        with tab3:
            if self.preprocessed_resumes is not None:
                preprocessed = self.preprocessed_resumes.iloc[resume_id]['preprocessed_text']
                st.text_area("Preprocessed Text", preprocessed, height=300)
            else:
                st.info("Run Module 2 to see preprocessed text")
        
        # Ranking performance
        if self.rankings is not None:
            st.markdown("### 🎯 Ranking Performance")
            
            resume_rankings = self.rankings[self.rankings['Resume_ID'] == resume_id]
            
            if len(resume_rankings) > 0:
                # Performance metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Best Match", f"{resume_rankings['Similarity_Score'].max():.2%}")
                with col2:
                    st.metric("Avg Match", f"{resume_rankings['Similarity_Score'].mean():.2%}")
                with col3:
                    best_rank = resume_rankings['Rank'].min()
                    st.metric("Best Rank", f"#{int(best_rank)}")
                
                # Rankings table
                st.markdown("**All Rankings:**")
                display_rankings = resume_rankings[['Job', 'Rank', 'Similarity_Score']].copy()
                display_rankings['Similarity_Score'] = display_rankings['Similarity_Score'].apply(lambda x: f"{x:.2%}")
                st.dataframe(display_rankings, width='stretch')
            else:
                st.info("This resume has no rankings")
    
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
        """Process uploaded resumes and job descriptions"""
        import re
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
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
        
        # Import NLTK
        try:
            import nltk
            from nltk.tokenize import word_tokenize
            from nltk.stem import WordNetLemmatizer
            from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
            
            # Download required NLTK data
            for package in ['punkt', 'wordnet', 'punkt_tab']:
                try:
                    nltk.data.find(f'tokenizers/{package}' if 'punkt' in package else f'corpora/{package}')
                except LookupError:
                    nltk.download(package, quiet=True)
        except ImportError:
            raise Exception("Please install NLTK: pip install nltk")
        
        # Extract text from resumes
        resumes_data = []
        
        for resume_file in resume_files:
            filename = resume_file.name
            
            try:
                if filename.endswith('.pdf'):
                    # Extract from PDF
                    text = self.extract_pdf_text(resume_file, pdf_library)
                else:
                    # Read text file
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
        
        # Preprocessing functions
        def clean_text(text):
            text = str(text).lower()
            # DON'T remove numbers - preserves esp32, n8n, python3, etc.
            text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with space
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
        
        def preprocess_text(text):
            # Simple tokenization - keep it minimal to avoid losing keywords
            tokens = text.split()
            # Only filter very short tokens (keep AI, ML, UI, etc.)
            tokens = [word for word in tokens if len(word) > 1]
            return ' '.join(tokens)
        
        # Process resumes
        resumes_df['cleaned_text'] = resumes_df['text'].apply(clean_text)
        resumes_df['preprocessed_text'] = resumes_df['cleaned_text'].apply(preprocess_text)
        
        # Process jobs
        jobs_df['cleaned_text'] = jobs_df['description'].apply(clean_text)
        jobs_df['preprocessed_text'] = jobs_df['cleaned_text'].apply(preprocess_text)
        
        # TF-IDF Vectorization
        resume_texts = resumes_df['preprocessed_text'].tolist()
        job_texts = jobs_df['preprocessed_text'].tolist()
        all_texts = resume_texts + job_texts
        
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english',
            use_idf=False,  # Disable IDF - doesn't work well with small corpus
            norm='l2',
            sublinear_tf=False
        )
        
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        resume_vectors = tfidf_matrix[:len(resume_texts)]  # type: ignore
        job_vectors = tfidf_matrix[len(resume_texts):]  # type: ignore
        
        # Compute TF-IDF cosine similarity
        tfidf_similarity = cosine_similarity(job_vectors, resume_vectors)
        
        # KEYWORD MATCHING (like Jobscan)
        def keyword_match_score(job_text, resume_text):
            """Calculate keyword overlap percentage"""
            job_words = set(job_text.lower().split())
            resume_words = set(resume_text.lower().split())
            
            # Filter to meaningful keywords (remove very common words)
            common_words = {'the', 'a', 'an', 'and', 'or', 'is', 'are', 'was', 'were', 'be', 'been', 
                           'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
                           'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used',
                           'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                           'through', 'during', 'before', 'after', 'above', 'below', 'between',
                           'this', 'that', 'these', 'those', 'it', 'its', 'we', 'our', 'you', 'your',
                           'they', 'their', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
                           'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                           'than', 'too', 'very', 'just', 'also', 'now', 'here', 'there', 'when',
                           'where', 'why', 'how', 'what', 'which', 'who', 'whom', 'if', 'then', 'else',
                           'any', 'about', 'over', 'under', 'again', 'further', 'once', 'including'}
            
            job_keywords = {w for w in job_words if len(w) > 2 and w not in common_words}
            resume_keywords = {w for w in resume_words if len(w) > 2 and w not in common_words}
            
            if not job_keywords:
                return 0.0
            
            # How many job keywords are found in resume?
            matched = job_keywords.intersection(resume_keywords)
            return len(matched) / len(job_keywords)
        
        # Calculate hybrid scores
        keyword_scores = np.zeros_like(tfidf_similarity)
        for job_idx, job_text in enumerate(job_texts):
            for resume_idx, resume_text in enumerate(resume_texts):
                keyword_scores[job_idx, resume_idx] = keyword_match_score(job_text, resume_text)
        
        # Combine: 70% keyword match + 30% TF-IDF (keyword matching is more intuitive)
        similarity_matrix = 0.7 * keyword_scores + 0.3 * tfidf_similarity
        
        # Create rankings
        top_n = 5
        rankings = []
        
        for job_idx in range(similarity_matrix.shape[0]):
            scores = similarity_matrix[job_idx]
            ranked_indices = np.argsort(scores)[::-1]
            top_resumes = ranked_indices[:top_n]
            top_scores = scores[top_resumes]
            
            for rank, (resume_id, score) in enumerate(zip(top_resumes, top_scores), 1):
                rankings.append({
                    'Job': f'Job_{job_idx+1}',
                    'Rank': rank,
                    'Resume_ID': resume_id,
                    'Similarity_Score': score
                })
        
        rankings_df = pd.DataFrame(rankings)
        
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
        
        #### 📊 System Pipeline
        
        1. **Module 1: Data Ingestion**
           - Downloads resumes from Kaggle
           - Extracts text from PDFs
           - Validates data quality
        
        2. **Module 2: Text Preprocessing**
           - Cleans and normalizes text
           - Tokenization and lemmatization
           - Stopword removal
        
        3. **Module 3: Feature Extraction**
           - TF-IDF vectorization
           - Converts text to numerical features
           - Creates consistent vocabulary
        
        4. **Module 4: Similarity Ranking**
           - Computes cosine similarity
           - Ranks resumes by relevance
           - Generates top candidates list
        
        5. **Module 5: Evaluation**
           - Analyzes performance metrics
           - Generates validation templates
           - Creates visualizations
        
        #### 🛠️ Technologies Used
        
        - **Python**: Core programming language
        - **scikit-learn**: TF-IDF vectorization, cosine similarity
        - **NLTK**: Natural language processing
        - **pandas**: Data manipulation
        - **Streamlit**: Web interface
        - **Plotly**: Interactive visualizations
        
        #### 👥 Team
        
        - **Module 1**: Data Ingestion & Resume Handling
        - **Module 2**: Text Preprocessing
        - **Module 3**: Feature Extraction (TF-IDF)
        - **Module 4**: Similarity Computation & Ranking
        - **Module 5**: Evaluation & Documentation
        
        #### 📝 Usage
        
        ```bash
        # Run complete pipeline
        python main.py
        
        # Launch web interface
        streamlit run app.py
        ```
        
        #### 📊 Dataset
        
        - **Source**: Kaggle Resume Dataset
        - **Resumes**: 100 samples
        - **Job Descriptions**: 10 curated positions
        - **Categories**: Multiple tech roles
        
        ---
        
        **Version**: 1.0.0  
        **Project**: CVSense - Intelligent Resume Screening
        """)


def main():
    """Main application entry point"""
    app = CVSenseApp()
    
    # Render sidebar and get selected page
    page = app.render_sidebar()
    
    # Render selected page
    if page == "dashboard":
        app.render_dashboard()
    elif page == "upload":
        app.render_upload()
    elif page == "job_search":
        app.render_job_search()
    elif page == "analytics":
        app.render_analytics()
    elif page == "resume_explorer":
        app.render_resume_explorer()
    elif page == "about":
        app.render_about()


if __name__ == "__main__":
    main()
