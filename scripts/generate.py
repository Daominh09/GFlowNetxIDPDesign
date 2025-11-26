import torch
import os
import json
from gfnxidp import get_default_args, get_tokenizer, get_oracle
from torch.distributions import Categorical
from gfnxidp.utils import Model, AttrSetter

def load_model(generator, load_path, device):
    print(f"Loading model from: {load_path}")
    
    checkpoint = torch.load(load_path, map_location=device)
    generator.model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Model loaded (Z={checkpoint['Z']:.4f})")
    return generator

def generate_sequences(model_path, args, tokenizer, n_samples=100):
    """
    Generate sequences from a saved model.
    
    Args:
        model_path: Path to saved model (e.g., 'logs/model.pt')
        args: Arguments with device, max_len, etc.
        tokenizer: Tokenizer instance
        n_samples: Number of sequences to generate
    
    Returns:
        sequences: List of generated sequences
    """
    from gfnxidp import get_generator
    
    # Create and load model
    generator = get_generator(args, tokenizer)
    generator = load_model(generator, model_path, args.device)
    generator.model.eval()
    
    sequences = []
    
    print(f"Generating {n_samples} sequences...")
    
    with torch.no_grad():
        batch_size = 32
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        for _ in range(n_batches):
            current_batch = min(batch_size, n_samples - len(sequences))
            states = [[] for _ in range(current_batch)]
            
            for t in range(args.gen_max_len):
                x = tokenizer.pad_tokens(states).to(args.device)
                logits = generator(x, None, coef=args.gen_output_coef)
                
                if t == 0:
                    logits[:, 0] = -1000
                
                cat = Categorical(logits=logits / args.gen_sampling_temperature)
                actions = cat.sample()
                
                for j, a in enumerate(actions):
                    states[j].append(a.item())
            
            for state in states:
                sequences.append(tokenizer.detokenize(state))
    
    print(f"✓ Generated {len(sequences)} sequences")
    return sequences

if __name__ == '__main__':
    
    # Setup
    args = get_default_args()
    args.device = torch.device('mps')
    tokenizer = get_tokenizer(args)
    oracle = get_oracle(args, tokenizer)
    model_path = args.model_path
    # Generate sequences
    sequences = generate_sequences(
        model_path=model_path,
        args=args,
        tokenizer=tokenizer,
        n_samples=1000
    )
    
    # Evaluate with oracle
    print("\nEvaluating sequences with oracle...")
    pair_data = []
    sequence_data = []
    for seq in sequences:
        tokens = tokenizer.tokenize(seq)
        score = oracle.predict(tokens)
        pair_data.append({
            'sequence': seq,
            'score': float(score)
        })
        sequence_data.append(seq)
    
    target_low = getattr(args, 'target_logcdil_low', -3)
    target_high = getattr(args, 'target_logcdil_high', 2.5)

    
    # Filter sequences in range
    in_range = [p['sequence'] for p in pair_data if target_low <= p['score'] <= target_high]
    
    # Create output structure
    output_data = {
        "logged_data": {
            'target_range': {
                'low': target_low,
                'high': target_high,
                'oracle_mode': args.oracle_mode
            },
            'statistics': {
                'total_generated': len(sequences),
                'in_range_count': len(in_range),
                'in_range_percentage': len(in_range) / len(sequences) * 100,
                'mean_score': sum(s['score'] for s in pair_data) / len(pair_data),
                'min_score': min(s['score'] for s in pair_data),
                'max_score': max(s['score'] for s in pair_data)
            },
            'all_sequences': sequence_data,
            'in_range_sequences': in_range
        }
    }
    
    # Save to JSON
    output_path = args.generate_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total sequences generated: {len(sequences)}")
    print(f"Target range: [{target_low}, {target_high}]")
    print(f"Sequences in range: {len(in_range)} ({len(in_range)/len(sequences)*100:.1f}%)")
    print(f"Mean score: {output_data['logged_data']['statistics']['mean_score']:.4f}")
    print(f"Score range: [{output_data['logged_data']['statistics']['min_score']:.4f}, {output_data['logged_data']['statistics']['max_score']:.4f}]")
    print(f"\nResults saved to: {output_path}")
    print(f"{'='*60}")
    
    
