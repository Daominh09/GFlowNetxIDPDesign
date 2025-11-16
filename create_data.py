import pandas as pd
import argparse
from lib.tokenizer import get_tokenizer
from lib.oracle import IDPOracle, ProtVec
from lib.args import get_default_args
import numpy as np

# Valid amino acids
VALID_ALPHABET = set(['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                      'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'])

def is_valid_sequence(seq):
    """Check if sequence contains only valid amino acids"""
    return all(aa in VALID_ALPHABET for aa in seq)

def process_tsv_with_oracle(input_file, output_file):
    """
    Process TSV file through DeePhase oracle and save results
    
    Args:
        input_file: Path to input TSV file
        output_file: Path to output CSV file
    """
    print("="*60)
    print("Processing sequences with DeePhase Oracle")
    print("="*60)
    
    # Read TSV file
    print(f"\nReading input file: {input_file}")
    df = pd.read_csv(input_file, sep='\t')
    print(f"Found {len(df)} total sequences")
    
    # Filter valid sequences
    print("\nFiltering sequences...")
    df['is_valid'] = df['Full.seq'].apply(is_valid_sequence)
    valid_df = df[df['is_valid']].copy()
    invalid_df = df[~df['is_valid']].copy()
    
    print(f"Valid sequences: {len(valid_df)}")
    print(f"Invalid sequences (contain non-standard AAs): {len(invalid_df)}")
    
    if len(invalid_df) > 0:
        print("\nInvalid sequences (showing UniProt IDs):")
        for idx, row in invalid_df.iterrows():
            invalid_chars = set(row['Full.seq']) - VALID_ALPHABET
            print(f"  {row['UniProt.Acc']}: contains {invalid_chars}")
    
    if len(valid_df) == 0:
        print("\nNo valid sequences to process!")
        return
    
    # Initialize tokenizer and oracle
    print("\nInitializing oracle...")
    args = get_default_args()
    tokenizer = get_tokenizer(args)
    oracle = IDPOracle(args, tokenizer)
    
    
    # Extract sequences
    sequences = valid_df['Full.seq'].tolist()
    uniprot_ids = valid_df['UniProt.Acc'].tolist()
    
    print(f"\nProcessing {len(sequences)} valid sequences...")
    print("This may take a while for large datasets...\n")
    
    # Process sequences in batch
    batch_tokens = [tokenizer.tokenize(seq) for seq in sequences]
    scores = oracle.batch_predict(batch_tokens)
    
    # Verify we got the right number of scores
    assert len(scores) == len(sequences), f"Score count mismatch: {len(scores)} != {len(sequences)}"
    
    # Create output dataframe
    results_df = pd.DataFrame({
        'uniprot_id': uniprot_ids,
        'sequence': sequences,
        'score': scores
    })
    
    # Save to CSV
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics:")
    print(f"Total sequences processed: {len(results_df)}")
    print(f"Mean score: {scores.mean():.4f}")
    print(f"Std score: {scores.std():.4f}")
    print(f"Min score: {scores.min():.4f}")
    print(f"Max score: {scores.max():.4f}")
    print("="*60)
    
    # Show first few results
    print("\nFirst 5 results:")
    for i in range(min(5, len(results_df))):
        row = results_df.iloc[i]
        print(f"{row['uniprot_id']}: {row['score']:.4f} (length={len(row['sequence'])})")

if __name__ == "__main__":
    # Configuration
    input_tsv = "lib/data/dataset.tsv"  # Change this
    output_csv = "lib/data/dataset.csv"  # Change this
    
    # Process
    process_tsv_with_oracle(input_tsv, output_csv)