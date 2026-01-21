#!/usr/bin/env python3
"""
CVSense - Intelligent Resume Screening System
Main Pipeline Execution Script

This script orchestrates all 5 modules to run the complete resume ranking pipeline:
Module 1: Data Ingestion → Module 2: Text Preprocessing → Module 3: TF-IDF Vectorization →
Module 4: Similarity Ranking → Module 5: Evaluation
"""

import os
import sys
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


class CVSensePipeline:
    """Complete pipeline for intelligent resume screening"""
    
    def __init__(self):
        self.project_root = PROJECT_ROOT
        self.data_dir = self.project_root / 'data'
        self.preprocessed_dir = self.data_dir / 'linguistically_preprocessed_files'
        
        # Module directories
        self.module_dirs = {
            1: self.project_root / 'module_1_data_ingestion',
            2: self.project_root / 'module_2_text_preprocessing',
            3: self.project_root / 'module_3_feature_extraction',
            4: self.project_root / 'module_4_similarity_ranking',
            5: self.project_root / 'module_5_evaluation_documentation'
        }
        
        # Data files
        self.files = {
            'raw_resumes': self.data_dir / 'processed_resumes.csv',
            'raw_jobs': self.data_dir / 'processed_job_descriptions.csv',
            'preprocessed_resumes': self.preprocessed_dir / 'preprocessed_resumes_final.csv',
            'preprocessed_jobs': self.preprocessed_dir / 'preprocessed_jobs_final.csv',
            'tfidf_vectors': self.module_dirs[3] / 'tfidf_vectors.pkl',
            'rankings': self.module_dirs[4] / 'module5_resume_ranking.csv'
        }
        
    def print_banner(self, text, char='='):
        """Print formatted banner"""
        print(f"\n{char * 80}")
        print(f"{text.center(80)}")
        print(f"{char * 80}\n")
        
    def check_module_outputs(self):
        """Check which modules have already been run"""
        status = {}
        
        # Module 1
        status[1] = self.files['raw_resumes'].exists() and self.files['raw_jobs'].exists()
        
        # Module 2
        status[2] = self.files['preprocessed_resumes'].exists() and self.files['preprocessed_jobs'].exists()
        
        # Module 3
        status[3] = self.files['tfidf_vectors'].exists()
        
        # Module 4
        status[4] = self.files['rankings'].exists()
        
        return status
        
    def run_module_2(self):
        """Run text preprocessing (Module 2)"""
        from module_2_text_preprocessing import preprocess_resumes, preprocess_jobs, ensure_nltk_data
        
        print("📥 Loading raw data from Module 1...")
        resumes_df = pd.read_csv(self.files['raw_resumes'])
        jobs_df = pd.read_csv(self.files['raw_jobs'])
        
        # Ensure NLTK data is available
        ensure_nltk_data()
        
        print("🧹 Preprocessing text (cleaning + linguistic processing)...")
        
        # Use module functions
        resumes_df = preprocess_resumes(resumes_df, text_column='cleaned_resume')
        jobs_df = preprocess_jobs(jobs_df, text_column='cleaned_description')
        
        # Save outputs
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        resumes_df.to_csv(self.files['preprocessed_resumes'], index=False)
        jobs_df.to_csv(self.files['preprocessed_jobs'], index=False)
        
        print(f"✅ Preprocessed {len(resumes_df)} resumes and {len(jobs_df)} job descriptions")
        print(f"📁 Saved to: {self.preprocessed_dir}/")
        
    def run_module_3(self):
        """Run TF-IDF vectorization (Module 3)"""
        from module_3_feature_extraction import create_tfidf_vectors, save_tfidf_vectors
        
        print("📥 Loading preprocessed data from Module 2...")
        resumes_df = pd.read_csv(self.files['preprocessed_resumes'])
        jobs_df = pd.read_csv(self.files['preprocessed_jobs'])
        
        # Extract text columns
        resume_texts = resumes_df['preprocessed_text'].astype(str).tolist()
        job_texts = jobs_df['preprocessed_text'].astype(str).tolist()
        
        print("🔢 Creating TF-IDF vectors...")
        
        # Use module function
        result = create_tfidf_vectors(
            resume_texts=resume_texts,
            job_texts=job_texts,
            max_features=5000,
            ngram_range=(1, 2),
            use_idf=True  # Use IDF for large corpus
        )
        
        # Save output
        save_tfidf_vectors(
            output_path=str(self.files['tfidf_vectors']),
            resume_vectors=result['resume_vectors'],
            job_vectors=result['jd_vectors'],
            feature_names=result['feature_names']
        )
        
        print(f"✅ Created TF-IDF vectors: {result['resume_vectors'].shape[0]} resumes × {result['resume_vectors'].shape[1]} features")
        print(f"✅ Created TF-IDF vectors: {result['jd_vectors'].shape[0]} jobs × {result['jd_vectors'].shape[1]} features")
        print(f"📁 Saved to: {self.files['tfidf_vectors']}")
        
    def run_module_4(self):
        """Run similarity ranking (Module 4)"""
        from module_3_feature_extraction import load_tfidf_vectors
        from module_4_similarity_ranking import compute_hybrid_scores, rank_resumes
        
        print("📥 Loading TF-IDF vectors from Module 3...")
        tfidf_data = load_tfidf_vectors(str(self.files['tfidf_vectors']))
        
        job_vectors = tfidf_data['jd_vectors']
        resume_vectors = tfidf_data['resume_vectors']
        
        # Load preprocessed texts for keyword matching
        print("📥 Loading preprocessed texts for hybrid scoring...")
        resumes_df = pd.read_csv(self.files['preprocessed_resumes'])
        jobs_df = pd.read_csv(self.files['preprocessed_jobs'])
        
        resume_texts = resumes_df['preprocessed_text'].astype(str).tolist()
        job_texts = jobs_df['preprocessed_text'].astype(str).tolist()
        
        print(f"📊 Computing hybrid similarity scores (70% keyword + 30% TF-IDF)...")
        
        # Use hybrid scoring for better results
        similarity_matrix = compute_hybrid_scores(
            job_texts=job_texts,
            resume_texts=resume_texts,
            job_vectors=job_vectors,
            resume_vectors=resume_vectors,
            keyword_weight=0.7,
            tfidf_weight=0.3
        )
        
        print(f"   Matrix shape: {similarity_matrix.shape} (jobs × resumes)")
        
        # Rank resumes
        top_n = 5
        df_results = rank_resumes(
            similarity_matrix=similarity_matrix,
            top_n=top_n
        )
        
        # Save rankings
        df_results.to_csv(self.files['rankings'], index=False)
        
        print(f"✅ Ranked top {top_n} resumes for {similarity_matrix.shape[0]} job descriptions")
        print(f"📁 Saved to: {self.files['rankings']}")
        
        # Show sample rankings
        print("\n📋 Sample Rankings (Job_1):")
        job1_results = df_results[df_results['Job'] == 'Job_1']
        for _, row in job1_results.iterrows():
            print(f"   Rank {int(row['Rank'])}: Resume {int(row['Resume_ID'])} — Score: {row['Similarity_Score']:.4f}")
        
    def run_module_5(self):
        """Run evaluation (Module 5)"""
        print("📊 Running evaluation metrics...")
        
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError as e:
            print(f"⚠️  Warning: Optional visualization libraries not installed: {e}")
            print("   Install with: pip install matplotlib seaborn")
            print("   Skipping visualization...")
            return
        
        # Import evaluation function here to avoid early import errors
        from module_5_evaluation_documentation.Evaluation_Metrics_Pipeline import run_evaluation
        
        # Change to project root for evaluation script
        original_dir = os.getcwd()
        os.chdir(self.project_root)
        
        try:
            run_evaluation()
        finally:
            os.chdir(original_dir)
        
        print("✅ Evaluation complete!")
        
    def run_complete_pipeline(self, force_rerun=False):
        """Run complete pipeline from start to finish"""
        self.print_banner("CVSense - Intelligent Resume Screening System", "=")
        
        # Check current status
        status = self.check_module_outputs()
        
        print("📦 Checking module outputs...")
        for module_num, completed in status.items():
            status_icon = "✅" if completed else "❌"
            print(f"   Module {module_num}: {status_icon}")
        
        # Module 1 check
        if not status[1]:
            print("\n⚠️  Module 1 data not found!")
            print("   Please run the Module 1 notebook first:")
            print("   → module_1_data_ingestion/data_ingestion.ipynb")
            return False
        
        # Module 2
        if not status[2] or force_rerun:
            self.print_banner("Module 2: Text Preprocessing", "-")
            self.run_module_2()
        else:
            print("\n✓ Module 2: Already completed (preprocessed files found)")
        
        # Module 3
        if not status[3] or force_rerun:
            self.print_banner("Module 3: TF-IDF Vectorization", "-")
            self.run_module_3()
        else:
            print("✓ Module 3: Already completed (TF-IDF vectors found)")
        
        # Module 4
        if not status[4] or force_rerun:
            self.print_banner("Module 4: Similarity Ranking", "-")
            self.run_module_4()
        else:
            print("✓ Module 4: Already completed (rankings found)")
        
        # Module 5
        self.print_banner("Module 5: Evaluation & Validation", "-")
        self.run_module_5()
        
        self.print_banner("Pipeline Complete!", "=")
        print("📊 All outputs saved in respective module directories")
        print("📈 Evaluation plots and manual validation template generated")
        print("\n✨ Resume screening system ready to use!")
        
        return True
    
    def get_top_candidates(self, job_id, top_n=5):
        """Get top N candidates for a specific job"""
        if not self.files['rankings'].exists():
            print("❌ Rankings not found. Please run the pipeline first.")
            return None
        
        rankings = pd.read_csv(self.files['rankings'])
        job_rankings = rankings[rankings['Job'] == f'Job_{job_id}'].head(top_n)
        
        return job_rankings[['Rank', 'Resume_ID', 'Similarity_Score']]


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='CVSense - Intelligent Resume Screening System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run complete pipeline
  python main.py --force            # Force re-run all modules
  python main.py --module 3         # Run from specific module onwards
  python main.py --evaluate         # Run evaluation only
        """
    )
    
    parser.add_argument('--force', action='store_true',
                       help='Force re-run all modules (ignore cached outputs)')
    parser.add_argument('--module', type=int, choices=[2, 3, 4, 5],
                       help='Start from specific module (assumes previous modules completed)')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run evaluation only (Module 5)')
    
    args = parser.parse_args()
    
    pipeline = CVSensePipeline()
    
    # Run evaluation only
    if args.evaluate:
        pipeline.print_banner("Running Evaluation (Module 5)", "=")
        pipeline.run_module_5()
        return
    
    # Run from specific module
    if args.module:
        status = pipeline.check_module_outputs()
        
        pipeline.print_banner(f"Running from Module {args.module}", "=")
        
        if args.module <= 2:
            pipeline.run_module_2()
        if args.module <= 3:
            pipeline.run_module_3()
        if args.module <= 4:
            pipeline.run_module_4()
        
        pipeline.run_module_5()
        return
    
    # Run complete pipeline
    pipeline.run_complete_pipeline(force_rerun=args.force)


if __name__ == "__main__":
    main()
