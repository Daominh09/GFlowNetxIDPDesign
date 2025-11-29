import torch
import os
import json
import numpy as np
from gfnxidp import get_default_args, get_tokenizer, get_oracle
from torch.distributions import Categorical
from gfnxidp.generator import PropertyNormalizer
from gfnxidp.utils import Model, AttrSetter

def generate_sequences_conditional(model_path, args, tokenizer, 
                                   target_csat_low, target_csat_high, 
                                   n_samples=100, n_points=5):
    """
    Generate sequences for a target CSAT range at inference time (ZERO-SHOT).
    
    Args:
        model_path: Path to trained model
        target_csat_low: Lower bound of target range (e.g., -8.0)
        target_csat_high: Upper bound of target range (e.g., -6.0)
        n_samples: Total sequences to generate
        n_points: Number of intermediate points in range to sample from
    
    Returns:
        Dictionary with sequences and their properties
    """
    from gfnxidp import get_generator
    
    # Load model
    generator = get_generator(args, tokenizer)
    checkpoint = torch.load(model_path, map_location=args.device)
    generator.model.load_state_dict(checkpoint['model_state_dict'])
    generator.model.eval()
    
    # Initialize normalizer
    normalizer = PropertyNormalizer(args)
    
    # Create list of target CSAT values to sample from
    target_csats = np.linspace(target_csat_low, target_csat_high, n_points)
    
    sequences = []
    oracle_scores = []
    condition_values = []
    
    print(f"\nGenerating sequences for CSAT range [{target_csat_low}, {target_csat_high}]")
    print(f"Using {n_points} intermediate target points")
    
    with torch.no_grad():
        batch_size = 32
        samples_per_target = n_samples // n_points
        
        for target_csat in target_csats:
            print(f"\n  Generating for target CSAT={target_csat:.2f}...")
            
            # Normalize condition
            condition_normalized = normalizer.normalize(torch.tensor([target_csat]))
            # condition_batch = condition_normalized.repeat(batch_size).to(args.device)
            
            for _ in range((samples_per_target + batch_size - 1) // batch_size):
                current_batch = min(batch_size, samples_per_target - len(sequences) // n_points)
                states = [[] for _ in range(current_batch)]
                
                # Generate sequences
                for t in range(args.gen_max_len):
                    x = tokenizer.pad_tokens(states).to(args.device)
                    
                    # Pass condition to model
                    condition_batch_t = condition_normalized.float().repeat(current_batch, 1).to(args.device)
                    logits = generator(x, None, condition=condition_batch_t, coef=args.gen_output_coef, guidance_scale=10.0)
                    
                    if t == 0:
                        logits[:, 0] = -1000
                    
                    cat = Categorical(logits=logits / args.gen_sampling_temperature)
                    actions = cat.sample()
                    
                    for j, a in enumerate(actions):
                        states[j].append(a.item())
                
                # Score sequences with oracle
                for state in states:
                    seq = tokenizer.detokenize(state)
                    tokens = tokenizer.tokenize(seq)
                    score = oracle.predict(tokens)
                    
                    sequences.append(seq)
                    oracle_scores.append(float(score))
                    condition_values.append(target_csat)
    
    return {
        'sequences': sequences,
        'oracle_scores': oracle_scores,
        'target_csats': condition_values,
        'target_range': [target_csat_low, target_csat_high]
    }


if __name__ == '__main__':
    args = get_default_args()
    args.device = torch.device('mps')
    tokenizer = get_tokenizer(args)
    oracle = get_oracle(args, tokenizer)
    
    # ZERO-SHOT: Specify target range WITHOUT retraining
    target_csat_low = float(input("Enter target CSAT low (e.g., -8.0): "))
    target_csat_high = float(input("Enter target CSAT high (e.g., -6.0): "))
    
    results = generate_sequences_conditional(
        model_path=args.model_path,
        args=args,
        tokenizer=tokenizer,
        target_csat_low=target_csat_low,
        target_csat_high=target_csat_high,
        n_samples=1000,
        n_points=5
    )
    
    # Analyze results
    in_range = [s for s, score in zip(results['sequences'], results['oracle_scores']) 
                if results['target_range'][0] <= score <= results['target_range'][1]]
    
    print(f"\n{'='*60}")
    print(f"ZERO-SHOT GENERATION RESULTS")
    print(f"{'='*60}")
    print(f"Target range: [{target_csat_low}, {target_csat_high}]")
    print(f"Total generated: {len(results['sequences'])}")
    print(f"In target range: {len(in_range)} ({100*len(in_range)/len(results['sequences']):.1f}%)")
    print(f"Mean score: {np.mean(results['oracle_scores']):.4f}")
    print(f"Score range: [{np.min(results['oracle_scores']):.4f}, {np.max(results['oracle_scores']):.4f}]")
    
    # Save results
    output_path = args.generate_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump({
            'logged_data': {
                'target_range': results['target_range'],
                'sequences': results['sequences'],
                'oracle_scores': results['oracle_scores'],
                'condition_targets': results['target_csats'],
                'statistics': {
                    'total_generated': len(results['sequences']),
                    'in_range_count': len(in_range),
                    'in_range_percentage': 100*len(in_range)/len(results['sequences']),
                    'mean_score': float(np.mean(results['oracle_scores'])),
                }
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")