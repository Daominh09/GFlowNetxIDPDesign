import pandas as pd
import numpy as np
import tempfile
import subprocess
import os
import matplotlib.pyplot as plt

def predict_disorder_iupred(seq, iupred_path):
    """
    Run IUPred2A to get per-residue disorder scores.
    Returns average disorder score, or np.nan on failure.
    """
    if not iupred_path or not seq or len(seq) == 0:
        return np.nan
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.fasta')
    try:
        with open(tmp.name, 'w') as f:
            f.write(f'>seq\n{seq}')
        
        result = subprocess.run(
            ['python', iupred_path, tmp.name, 'long'],
            capture_output=True, text=True, timeout=60
        )
        
        # Parse IUPred output
        lines = result.stdout.strip().split('\n')
        scores = []
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    scores.append(float(parts[2]))
                except ValueError:
                    continue
        
        if scores:
            return np.mean(scores)
        return np.nan
        
    except Exception as e:
        print(f"IUPred error for seq length {len(seq)}: {e}")
        return np.nan
    finally:
        try:
            os.unlink(tmp.name)
        except:
            pass




def plot_scores(df):
    """
    Visualize csat and iupred_score on the same line chart sorted by iupred_score.
    """
    # Sort by iupred_score to see patterns clearly
    df_sorted = df.sort_values('iupred_score').reset_index(drop=True)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Plot both scores on the same graph
    ax.plot(range(len(df_sorted)), df_sorted['csat'], marker='o', linestyle='-', 
            linewidth=2, markersize=5, color='steelblue', label='CSAT Score', alpha=0.8)
    ax.plot(range(len(df_sorted)), df_sorted['iupred_score'], marker='s', linestyle='-', 
            linewidth=2, markersize=5, color='coral', label='IUPred Score', alpha=0.8)
    
    ax.set_xlabel('Sequence Index (sorted by IUPred Score)', fontsize=12)
    ax.set_ylabel('Score Value', fontsize=12)
    ax.set_title('CSAT vs IUPred Scores - Correlation Comparison', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('iupred_csat_correlation.png', dpi=300, bbox_inches='tight')
    print("Chart saved as 'iupred_csat_correlation.png'")
    
    # Calculate correlation
    correlation = df['csat'].corr(df['iupred_score'])
    print(f"\nPearson Correlation between CSAT and IUPred: {correlation:.4f}")
    
    plt.show()


def compute_iupred_scores(csv_path, iupred_path, output_path=None):
    """
    Compute IUPred scores for each sequence in a CSV dataset.
    
    Args:
        csv_path: Path to input CSV file
        iupred_path: Path to IUPred2A script
        output_path: Path to save output CSV (optional, defaults to input path with _scored suffix)
    """
    # Load dataset
    df = pd.read_csv(csv_path)
    
    # Verify required columns
    if 'sequence' not in df.columns:
        raise ValueError("CSV must contain 'sequence' column")
    
    print(f"Processing {len(df)} sequences...")
    
    # Compute IUPred scores
    iupred_scores = []
    for idx, row in df.iterrows():
        seq = row['sequence']
        score = predict_disorder_iupred(seq, iupred_path)
        iupred_scores.append(score)
        
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(df)} sequences")
    
    # Add scores to dataframe
    df['iupred_score'] = iupred_scores
    
    # Save output
    if output_path is None:
        output_path = csv_path.replace('.csv', '_scored.csv')
    
    df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    return df


if __name__ == "__main__":
    # Configuration
    csv_file = "datasets/de_dataset.csv"  # Change to your CSV file path
    iupred_script = "src/gfnxidp/tools/iupred2a.py"  # Change to your IUPred2A script path
    output_file = "datasets/de_dataset_scored.csv"  # Optional: specify output file
    
    # Compute scores
    df_scored = compute_iupred_scores(csv_file, iupred_script, output_file)
    
    # Display results
    print("\nFirst few rows:")
    print(df_scored.head())
    
    # Print statistics for iupred_score
    valid_scores = df_scored['iupred_score'].dropna()
    print(f"\n--- IUPred Score Statistics ---")
    print(f"Range: {valid_scores.min():.6f} to {valid_scores.max():.6f}")
    print(f"Mean: {valid_scores.mean():.6f}")
    print(f"Std: {valid_scores.std():.6f}")
    
    # Visualize csat and iupred_score
    plot_scores(df_scored)