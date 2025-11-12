import sys
import os

from lib.args import get_default_args
from lib.dataset import IDPDataset
from lib.tokenizer import get_tokenizer
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


def main():
    args = get_default_args()
    tokenizer = get_tokenizer(args)
    dataset = IDPDataset(args, tokenizer)

    sizes = dataset.get_dataset_size()
    print(f"Training set: {sizes['train_size']}")
    print(f"Validation set: {sizes['valid_size']}")
    print(f"Total size: {sizes['total_size']}")
    print(f"Total collected during training: {sizes['total_collected']}")
    print()

    x, y = dataset.sample(32)
    print(f"Sample batch size: {len(x)}")
    print(f"First sample tokens: {x[0]}")
    print(f"First sample dG: {y[0]}")
    print()

    val_tokens, val_dg = dataset.validation_set()
    print(f"Validation set size: {len(val_tokens)}")
    print()

    top_10_seqs, top_10_scores = dataset.top_k(10)
    print(f"Top 10 sequences:")
    for i in range(min(3, len(top_10_seqs))):
        print(f"  {i+1}. {top_10_seqs[i][:50]}... (dG: {top_10_scores[i]:.3f})")
    print()

    stats = dataset.get_statistics()
    print(f"Dataset Statistics:")
    print(f"  Mean dG: {stats['dg_mean']:.3f}")
    print(f"  Std dG: {stats['dg_std']:.3f}")
    print(f"  Min dG: {stats['dg_min']:.3f}")
    print(f"  Max dG: {stats['dg_max']:.3f}")
    print(f"  Mean sequence length: {stats['seq_length_mean']:.1f}")
    print(f"  Sequence length range: {stats['seq_length_min']}-{stats['seq_length_max']}")
    print(f"  Vocabulary size: {stats['vocab_size']}")
    print(f"  Total dataset size: {stats['dataset_size']}")

if __name__ == "__main__":
    main()