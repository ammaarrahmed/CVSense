import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_evaluation():
    print("START: Starting Module 5: Evaluation...")
    
    file_path = 'module5_resume_ranking.csv'
    
    if not os.path.exists(file_path):
        print(f"❌ Error: {file_path} not found.")
        print("Please ensure Module 4 has been executed successfully.")
        return

    # Load data
    df = pd.read_csv(file_path)
    
    # 1. Basic Stats
    print("\n--- Similarity Score Statistics ---")
    print(df['Similarity_Score'].describe())
    
    # 2. Score Distribution Plot
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Similarity_Score'], bins=15, kde=True, color='skyblue')
    plt.axvline(df['Similarity_Score'].mean(), color='red', linestyle='--', label='Mean')
    plt.title('Resume Similarity Score Distribution')
    plt.legend()
    plt.savefig('evaluation_score_distribution.png')
    print("DONE: Distribution plot saved as 'evaluation_score_distribution.png'")
    
    # 3. Top Candidates
    print("\n--- Top 3 Candidates per Job ---")
    top_3 = df[df['Rank'] <= 3].sort_values(by=['Job', 'Rank'])
    print(top_3[['Job', 'Rank', 'Resume_ID', 'Similarity_Score']])
    
    # 4. Export Validation Template
    val_file = 'manual_validation.csv'
    top_3.to_csv(val_file, index=False)
    print(f"DONE: Validation template exported to '{val_file}'")
    
    print("\nDONE: Evaluation Module complete.")

if __name__ == "__main__":
    run_evaluation()
