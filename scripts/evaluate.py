import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import re

from gfnxidp import get_default_args
from gfnxidp import get_oracle
from gfnxidp import get_tokenizer
from gfnxidp.utils import Model, AttrSetter

# Motif library
MOTIF_LIBRARY = {
    'RGG_short': 'RGGRGG',
    'FG_repeat': 'FGFG',
    'Aromatic': 'FYFY',
    'Prion': 'SYGQ',
    'RG_dipep': 'RGRG',
}


import tempfile
import subprocess
import os


def analyze_motifs_in_sequence(seq):
    """
    Check which motifs are present in a sequence.
    Returns dict with boolean flags for each motif.
    """
    motif_presence = {}
    for name, motif_seq in MOTIF_LIBRARY.items():
        motif_presence[f'has_{name}'] = motif_seq in seq
    
    # Count total motifs present
    motif_presence['total_motifs'] = sum(motif_presence.values())
    
    return motif_presence


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


def compute_sequence_constraints(seq, args=None, iupred_path=None):
    """
    Compute IDP constraint metrics for a single sequence.
    Returns dict with fractions and violation flags.
    """
    if not seq or len(seq) == 0:
        return {
            'length': 0,
            'cys_fraction': np.nan,
            'hydrophobic_fraction': np.nan,
            'aromatic_fraction': np.nan,
            'disorder_score': np.nan,
            'cys_violation': np.nan,
            'hydrophobic_violation': np.nan,
            'aromatic_violation': np.nan,
            'disorder_violation': np.nan,
            'any_violation': np.nan
        }
    
    # Thresholds (use args if provided, else defaults)
    cys_max = getattr(args, 'cys_max_fraction', 0.03) if args else 0.03
    hydro_max = getattr(args, 'hydrophobic_max_fraction', 0.35) if args else 0.35
    arom_min = getattr(args, 'aromatic_min_fraction', 0.08) if args else 0.08
    arom_max = getattr(args, 'aromatic_max_fraction', 0.12) if args else 0.12
    min_disorder = getattr(args, 'min_disorder_score', 0.65) if args else 0.65
    
    # Amino acid groups
    hydrophobic_aa = set('AILMFWVY')
    aromatic_aa = set('FWY')
    
    length = len(seq)
    
    # Compute fractions
    cys_frac = seq.count('C') / length
    hydro_frac = sum(1 for aa in seq if aa in hydrophobic_aa) / length
    arom_frac = sum(1 for aa in seq if aa in aromatic_aa) / length
    
    # Compute disorder score if iupred_path provided
    if iupred_path:
        disorder_score = predict_disorder_iupred(seq, iupred_path)
    else:
        disorder_score = np.nan
    
    # Check violations
    cys_violation = cys_frac > cys_max
    hydro_violation = hydro_frac > hydro_max
    arom_violation = (arom_frac < arom_min) or (arom_frac > arom_max)
    disorder_violation = disorder_score < min_disorder if not np.isnan(disorder_score) else False
    any_violation = cys_violation or hydro_violation or arom_violation or disorder_violation
    
    return {
        'length': length,
        'cys_fraction': cys_frac,
        'hydrophobic_fraction': hydro_frac,
        'aromatic_fraction': arom_frac,
        'disorder_score': disorder_score,
        'cys_violation': cys_violation,
        'hydrophobic_violation': hydro_violation,
        'aromatic_violation': arom_violation,
        'disorder_violation': disorder_violation,
        'any_violation': any_violation
    }


def compute_constraint_penalty(seq, args=None):
    """
    Compute multiplicative penalty factor for constraint violations.
    Returns value in (0, 1] where 1 = no violations.
    """
    if not seq or len(seq) == 0:
        return 0.0
    
    # Thresholds
    cys_max = getattr(args, 'cys_max_fraction', 0.03) if args else 0.03
    hydro_max = getattr(args, 'hydrophobic_max_fraction', 0.35) if args else 0.35
    arom_min = getattr(args, 'aromatic_min_fraction', 0.08) if args else 0.08
    arom_max = getattr(args, 'aromatic_max_fraction', 0.12) if args else 0.12
    
    # Penalty strengths
    cys_strength = getattr(args, 'cys_penalty_strength', 10.0) if args else 10.0
    hydro_strength = getattr(args, 'hydrophobic_penalty_strength', 10.0) if args else 10.0
    arom_strength = getattr(args, 'aromatic_penalty_strength', 10.0) if args else 10.0
    
    hydrophobic_aa = set('AILMFWVY')
    aromatic_aa = set('FWY')
    length = len(seq)
    
    # Cysteine penalty
    cys_frac = seq.count('C') / length
    if cys_frac <= cys_max:
        cys_penalty = 1.0
    else:
        cys_penalty = np.exp(-cys_strength * (cys_frac - cys_max))
    
    # Hydrophobic penalty
    hydro_frac = sum(1 for aa in seq if aa in hydrophobic_aa) / length
    if hydro_frac <= hydro_max:
        hydro_penalty = 1.0
    else:
        hydro_penalty = np.exp(-hydro_strength * (hydro_frac - hydro_max))
    
    # Aromatic penalty
    arom_frac = sum(1 for aa in seq if aa in aromatic_aa) / length
    if arom_min <= arom_frac <= arom_max:
        arom_penalty = 1.0
    elif arom_frac < arom_min:
        arom_penalty = np.exp(-arom_strength * (arom_min - arom_frac))
    else:
        arom_penalty = np.exp(-arom_strength * (arom_frac - arom_max))
    
    return cys_penalty * hydro_penalty * arom_penalty


def run_oracle_on_json(input_json, output_csv, args, tokenizer, 
                       data_key="logged_data", 
                       seq_key="in_range_sequences"):
    """
    Run DG and CSAT predictions on sequences from a JSON file.
    Includes constraint analysis.
    """
    # Load JSON and extract sequences
    with open(input_json, 'r') as f:
        data = json.load(f)
    
    sequences = data[data_key][seq_key]
    print(f"Loaded {len(sequences)} sequences from JSON")
    
    # Generate protein IDs
    protein_ids = [f"{i+1:03d}" for i in range(len(sequences))]
    
    # Convert token lists to sequences if needed
    processed_sequences = []
    for seq in sequences:
        if isinstance(seq, list):
            seq = tokenizer.detokenize(seq)
        processed_sequences.append(seq)
    sequences = processed_sequences
    
    # ========== CONSTRAINT ANALYSIS ==========
    print("\nAnalyzing constraints...")
    iupred_path = getattr(args, 'iupred2a', None)
    if iupred_path:
        print(f"  Using IUPred2A at: {iupred_path}")
    else:
        print("  IUPred2A not configured - disorder scores will be NaN")
    
    constraint_results = []
    motif_results = []
    for seq in tqdm(sequences):
        result = compute_sequence_constraints(seq, args, iupred_path=iupred_path)
        result['penalty_factor'] = compute_constraint_penalty(seq, args)
        constraint_results.append(result)
        
        # Analyze motifs
        motif_result = analyze_motifs_in_sequence(seq)
        motif_results.append(motif_result)
    
    # ========== DG PREDICTIONS ==========
    print("\nRunning DG predictions...")
    args.oracle_mode = 'dg'
    dg_oracle = get_oracle(args, tokenizer)
    
    dg_predictions = []
    for seq in tqdm(sequences):
        try:
            tokens = tokenizer.tokenize(seq)
            pred = dg_oracle.predict(tokens)
            dg_predictions.append(pred)
        except Exception as e:
            print(f"Error: {e}")
            dg_predictions.append(np.nan)
    
    # ========== CSAT PREDICTIONS ==========
    print("\nRunning CSAT predictions...")
    args.oracle_mode = 'csat'
    csat_oracle = get_oracle(args, tokenizer)
    
    log_cdil_predictions = []
    csat_predictions = []
    for seq in tqdm(sequences):
        try:
            tokens = tokenizer.tokenize(seq)
            pred = csat_oracle.predict(tokens)
            log_cdil_predictions.append(pred)
            csat_predictions.append(np.exp(pred))
        except Exception as e:
            print(f"Error: {e}")
            log_cdil_predictions.append(np.nan)
            csat_predictions.append(np.nan)
    
    # ========== CREATE DATAFRAME ==========
    df = pd.DataFrame({
        'protein_id': protein_ids,
        'sequence': sequences,
        'length': [r['length'] for r in constraint_results],
        'dg': dg_predictions,
        'log_cdil': log_cdil_predictions,
        'csat': csat_predictions,
        # Constraint fractions
        'cys_fraction': [r['cys_fraction'] for r in constraint_results],
        'hydrophobic_fraction': [r['hydrophobic_fraction'] for r in constraint_results],
        'aromatic_fraction': [r['aromatic_fraction'] for r in constraint_results],
        'disorder_score': [r['disorder_score'] for r in constraint_results],
        # Violation flags
        'cys_violation': [r['cys_violation'] for r in constraint_results],
        'hydrophobic_violation': [r['hydrophobic_violation'] for r in constraint_results],
        'aromatic_violation': [r['aromatic_violation'] for r in constraint_results],
        'disorder_violation': [r['disorder_violation'] for r in constraint_results],
        'any_violation': [r['any_violation'] for r in constraint_results],
        # Penalty factor
        'penalty_factor': [r['penalty_factor'] for r in constraint_results],
        # Motif presence
        'has_RGG_short': [m['has_RGG_short'] for m in motif_results],
        'has_FG_repeat': [m['has_FG_repeat'] for m in motif_results],
        'has_Aromatic': [m['has_Aromatic'] for m in motif_results],
        'has_Prion': [m['has_Prion'] for m in motif_results],
        'has_RG_dipep': [m['has_RG_dipep'] for m in motif_results],
        'total_motifs': [m['total_motifs'] for m in motif_results],
    })
    
    # Sort by csat ascending
    df = df.sort_values(by='csat', ascending=True).reset_index(drop=True)
    
    # Save results
    df.to_csv(output_csv, index=False)
    
    # ========== PRINT SUMMARY ==========
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Saved to: {output_csv}")
    print(f"Total sequences: {len(df)}")
    print(f"\nOracle Predictions:")
    print(f"  DG mean:   {np.nanmean(dg_predictions):.4f}")
    print(f"  CSAT mean: {np.nanmean(csat_predictions):.4f}")
    
    print(f"\nConstraint Analysis:")
    print(f"  Cysteine violations (>3%):       {df['cys_violation'].sum()} ({df['cys_violation'].mean()*100:.1f}%)")
    print(f"  Hydrophobic violations (>35%):   {df['hydrophobic_violation'].sum()} ({df['hydrophobic_violation'].mean()*100:.1f}%)")
    print(f"  Aromatic violations (<8% or >12%): {df['aromatic_violation'].sum()} ({df['aromatic_violation'].mean()*100:.1f}%)")
    print(f"  Disorder violations (<0.65):     {df['disorder_violation'].sum()} ({df['disorder_violation'].mean()*100:.1f}%)")
    print(f"  Any violation:                   {df['any_violation'].sum()} ({df['any_violation'].mean()*100:.1f}%)")
    print(f"  Mean penalty factor:             {df['penalty_factor'].mean():.4f}")
    
    print(f"\nFraction Statistics:")
    print(f"  Cysteine:    mean={df['cys_fraction'].mean()*100:.2f}%, max={df['cys_fraction'].max()*100:.2f}%")
    print(f"  Hydrophobic: mean={df['hydrophobic_fraction'].mean()*100:.2f}%, max={df['hydrophobic_fraction'].max()*100:.2f}%")
    print(f"  Aromatic:    mean={df['aromatic_fraction'].mean()*100:.2f}%, range=[{df['aromatic_fraction'].min()*100:.2f}%, {df['aromatic_fraction'].max()*100:.2f}%]")
    print(f"  Disorder:    mean={df['disorder_score'].mean():.3f}, min={df['disorder_score'].min():.3f}, max={df['disorder_score'].max():.3f}")
    
    print(f"\nMotif Presence:")
    print(f"  RGG_short (RGGRGG):  {df['has_RGG_short'].sum()} ({df['has_RGG_short'].mean()*100:.1f}%)")
    print(f"  FG_repeat (FGFG):    {df['has_FG_repeat'].sum()} ({df['has_FG_repeat'].mean()*100:.1f}%)")
    print(f"  Aromatic (FYFY):     {df['has_Aromatic'].sum()} ({df['has_Aromatic'].mean()*100:.1f}%)")
    print(f"  Prion (SYGQ):        {df['has_Prion'].sum()} ({df['has_Prion'].mean()*100:.1f}%)")
    print(f"  RG_dipep (RGRG):     {df['has_RG_dipep'].sum()} ({df['has_RG_dipep'].mean()*100:.1f}%)")
    print(f"  Mean motifs per seq: {df['total_motifs'].mean():.2f}")
    print(f"{'='*60}")
    
    return df


def run_oracle_on_dataset(input_csv, output_csv, args, tokenizer):
    """
    Run both DG and CSAT predictions on a CSV dataset.
    Includes constraint analysis.
    """
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} sequences")
    
    sequences = df['sequence'].tolist()
    
    # Constraint analysis
    print("\nAnalyzing constraints...")
    iupred_path = getattr(args, 'iupred2a', None)
    if iupred_path:
        print(f"  Using IUPred2A at: {iupred_path}")
    else:
        print("  IUPred2A not configured - disorder scores will be NaN")
    
    constraint_results = []
    for seq in tqdm(sequences):
        result = compute_sequence_constraints(seq, args, iupred_path=iupred_path)
        result['penalty_factor'] = compute_constraint_penalty(seq, args)
        constraint_results.append(result)
    
    # DG predictions
    print("\nRunning DG predictions...")
    args.oracle_mode = 'dg'
    dg_oracle = get_oracle(args, tokenizer)
    
    dg_predictions = []
    for seq in tqdm(sequences):
        try:
            tokens = tokenizer.tokenize(seq)
            pred = dg_oracle.predict(tokens)
            dg_predictions.append(pred)
        except Exception as e:
            print(f"Error: {e}")
            dg_predictions.append(np.nan)
    
    # CSAT predictions
    print("\nRunning CSAT predictions...")
    args.oracle_mode = 'csat'
    csat_oracle = get_oracle(args, tokenizer)
    
    csat_predictions = []
    for seq in tqdm(sequences):
        try:
            tokens = tokenizer.tokenize(seq)
            pred = csat_oracle.predict(tokens)
            csat_predictions.append(pred)
        except Exception as e:
            print(f"Error: {e}")
            csat_predictions.append(np.nan)
    
    # Add columns
    df['dg'] = dg_predictions
    df['csat'] = csat_predictions
    df['cys_fraction'] = [r['cys_fraction'] for r in constraint_results]
    df['hydrophobic_fraction'] = [r['hydrophobic_fraction'] for r in constraint_results]
    df['aromatic_fraction'] = [r['aromatic_fraction'] for r in constraint_results]
    df['disorder_score'] = [r['disorder_score'] for r in constraint_results]
    df['cys_violation'] = [r['cys_violation'] for r in constraint_results]
    df['hydrophobic_violation'] = [r['hydrophobic_violation'] for r in constraint_results]
    df['aromatic_violation'] = [r['aromatic_violation'] for r in constraint_results]
    df['disorder_violation'] = [r['disorder_violation'] for r in constraint_results]
    df['any_violation'] = [r['any_violation'] for r in constraint_results]
    df['penalty_factor'] = [r['penalty_factor'] for r in constraint_results]
    
    # Sort by csat ascending
    df = df.sort_values(by='csat', ascending=True).reset_index(drop=True)
    
    df.to_csv(output_csv, index=False)
    print(f"\nSaved to {output_csv}")
    print(f"DG mean: {np.nanmean(dg_predictions):.4f}")
    print(f"CSAT mean: {np.nanmean(csat_predictions):.4f}")
    print(f"Disorder mean: {df['disorder_score'].mean():.3f}")
    print(f"Sequences with violations: {df['any_violation'].sum()} ({df['any_violation'].mean()*100:.1f}%)")
    
    return df


if __name__ == '__main__':
    args = get_default_args()
    tokenizer = get_tokenizer(args)
    
    # Process JSON file
    df_json = run_oracle_on_json(
        input_json='logs/generating_log.json',
        output_csv='datasets/generated_data.csv',
        args=args,
        tokenizer=tokenizer
    )