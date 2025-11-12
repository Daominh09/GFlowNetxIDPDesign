"""
Test script for IDPOracle class
Reads sequences from example.fasta and makes predictions
Modified to work with tokenizer-based oracle
"""

import sys
import os
import numpy as np

from lib.args import get_default_args
from lib.oracle import IDPOracle
from lib.tokenizer import get_tokenizer
from lib.utils import Model, AttrSetter

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


def read_fasta(filename):
    """
    Read sequences from FASTA file
    Returns dict of {header: sequence}
    """
    sequences = {}
    current_header = None
    current_seq = []
    
    try:
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    # Save previous sequence if exists
                    if current_header:
                        sequences[current_header] = ''.join(current_seq)
                    # Start new sequence
                    current_header = line[1:]  # Remove '>'
                    current_seq = []
                elif line:
                    current_seq.append(line)
            
            # Save last sequence
            if current_header:
                sequences[current_header] = ''.join(current_seq)
                
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found!")
        sys.exit(1)
    
    return sequences


def test_single_predictions(oracle, tokenizer, sequences):
    """Test predict() method on individual sequences"""
    print("\n" + "="*70)
    print("SINGLE SEQUENCE PREDICTIONS")
    print("="*70)
    
    for header, seq in sequences.items():
        print(f"\n{header}")
        print(f"  Sequence: {seq[:60]}{'...' if len(seq) > 60 else ''}")
        print(f"  Length: {len(seq)} residues")
        
        try:
            # Tokenize the sequence
            tokens = tokenizer.tokenize(seq)
            # predict() now returns a float32 directly
            dg_value = oracle.predict(tokens)
            print(f"  Predicted ΔG: {float(dg_value):.4f}")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()


def test_batch_predictions(oracle, tokenizer, sequences):
    """Test batch_predict() method on multiple sequences"""
    print("\n" + "="*70)
    print("BATCH PREDICTIONS")
    print("="*70)
    
    # Prepare batch
    headers = list(sequences.keys())
    seqs = list(sequences.values())
    
    try:
        # Tokenize all sequences
        batch_tokens = [tokenizer.tokenize(seq) for seq in seqs]
        batch_tokens = tokenizer.pad_tokens(batch_tokens)
        # batch_predict() now returns an array of float32 values
        dg_values = oracle.batch_predict(batch_tokens)
        
        print(f"\nProcessed {len(seqs)} sequences")
        print("\nResults Summary:")
        print("-" * 70)
        print(f"{'Sequence Name':<25} {'Length':>8} {'Predicted ΔG':>15}")
        print("-" * 70)
        
        for i, (header, dg) in enumerate(zip(headers, dg_values)):
            seq_len = len(seqs[i])
            print(f"{header:<25} {seq_len:>8} {float(dg):>15.4f}")
        
        # Statistics
        print("-" * 70)
        print(f"\nStatistics:")
        print(f"  Mean ΔG: {np.mean(dg_values):.4f}")
        print(f"  Std Dev: {np.std(dg_values):.4f}")
        print(f"  Min ΔG:  {np.min(dg_values):.4f} ({headers[np.argmin(dg_values)]})")
        print(f"  Max ΔG:  {np.max(dg_values):.4f} ({headers[np.argmax(dg_values)]})")
            
    except Exception as e:
        print(f"Error in batch prediction: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main test function"""
    print("="*70)
    print("IDPOracle Test Suite")
    print("="*70)
    
    # Load configuration
    args = get_default_args()
    print("\nConfiguration:")
    print(f"  DG Model: {args.dg_file}")
    print(f"  NU Model: {args.nu_file}")
    print(f"  Residues File: {args.residues_file}")
    print(f"  Temperature: {args.temperature} K")
    print(f"  Ionic Strength: {args.ionic_strength} M")
    print(f"  Charge Termini: {args.charge_termini}")
    
    # Initialize Tokenizer
    print("\nInitializing Tokenizer...")
    tokenizer = get_tokenizer(args)
    print("✓ Tokenizer initialized")
    
    # Initialize Oracle
    print("\nInitializing IDPOracle...")
    try:
        oracle = IDPOracle(args, tokenizer)
        print("✓ Oracle initialized successfully")
        print(f"  Loaded features: {', '.join(oracle.feature)}")
        print(f"  Target predictions: {', '.join(oracle.targets)}")
    except Exception as e:
        print(f"✗ Failed to initialize Oracle: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Read FASTA file
    fasta_file = "example.fasta"
    print(f"\nReading sequences from {fasta_file}...")
    sequences = read_fasta(fasta_file)
    print(f"✓ Loaded {len(sequences)} sequences")
    
    if not sequences:
        print("No sequences found in FASTA file!")
        sys.exit(1)
    
    # Display sequence info
    print("\nSequences loaded:")
    for i, (header, seq) in enumerate(sequences.items(), 1):
        print(f"  {i}. {header} ({len(seq)} aa)")
    
    # Run tests
    test_single_predictions(oracle, tokenizer, sequences)
    test_batch_predictions(oracle, tokenizer, sequences)
    
    print("\n" + "="*70)
    print("Test Complete!")
    print("="*70)


if __name__ == "__main__":
    main()