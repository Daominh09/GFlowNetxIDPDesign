import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

class IDPDataset:
    def __init__(self, args, tokenizer):
        self.args = args
        self.rng = np.random.RandomState(args.seed)
        self.tokenizer = tokenizer
        self.y_mean = None
        self.y_std = None
        self.is_normalized = False
        
        # Check if using direct mode (no normalization needed)
        self.use_direct_mode = getattr(args, "proxy_mode", "gaussian").lower() == "direct"
        
        self._load_dataset(args.train_path, args.test_size)
        self.train_added = len(self.train)
        self.val_added = len(self.valid)
        
        if self.use_direct_mode:
            print("\n" + "="*60)
            print("DIRECT MODE: Scores assumed to be in [0,1] range")
            print("No normalization will be applied")
            print("="*60 + "\n")

    def _load_dataset(self, csv_path, test_size):
        df = pd.read_csv(csv_path)
        seqs = df['sequence'].values
        scores = df['csat'].values
        batch_tokens = []
        batch_scores = []
        for seq, score in zip(seqs, scores):
            if pd.isna(score):
                continue
            tokens = self.tokenizer.tokenize(seq)
            if tokens is not None:
                batch_tokens.append(tokens)
                batch_scores.append(score)
        
        batch_scores = np.array(batch_scores)
        
        # Validate scores for direct mode
        if self.use_direct_mode:
            if np.any(batch_scores < 0) or np.any(batch_scores > 1):
                print(f"WARNING: Direct mode enabled but scores outside [0,1]!")
                print(f"  Score range: [{batch_scores.min():.4f}, {batch_scores.max():.4f}]")
                print(f"  Clipping to [0,1]...")
                batch_scores = np.clip(batch_scores, 0, 1)
        
        self.train, self.valid, self.train_scores, self.valid_scores = train_test_split(
            batch_tokens, batch_scores, test_size=test_size, random_state=self.rng
        )
        
    def map_normalize_y(self):
        """Normalize scores (only if not using direct mode)"""
        if self.use_direct_mode:
            print("Direct mode enabled - skipping normalization")
            return
        
        all_scores = np.concatenate((self.train_scores, self.valid_scores), axis=0)
        self.y_mean = np.mean(all_scores)
        self.y_std = np.std(all_scores)
        
        self.train_scores = (self.train_scores - self.y_mean) / self.y_std
        self.valid_scores = (self.valid_scores - self.y_mean) / self.y_std
        self.is_normalized = True
    
    def normalize_scores(self, scores):
        """Normalize scores (pass-through in direct mode)"""
        if self.use_direct_mode or not self.is_normalized:
            return scores
        if isinstance(scores, (list, np.ndarray)):
            return (np.array(scores) - self.y_mean) / self.y_std
        else:
            return (scores - self.y_mean) / self.y_std
    
    def denormalize_scores(self, scores):
        """Denormalize scores (pass-through in direct mode)"""
        if self.use_direct_mode or not self.is_normalized:
            return scores
        if isinstance(scores, (list, np.ndarray)):
            return np.array(scores) * self.y_std + self.y_mean
        else:
            return scores * self.y_std + self.y_mean
        
    def sample(self, n, max_len=None):
        if max_len is None:
            indices = np.random.randint(0, len(self.train), n)
            return ([self.train[i] for i in indices],
                    [self.train_scores[i] for i in indices])
        
        valid_indices = [i for i in range(len(self.train)) if len(self.train[i]) <= max_len]
        
        if len(valid_indices) == 0:
            raise ValueError(f"No training sequences with length <= {max_len}")
        
        if len(valid_indices) < n:
            print(f"Warning: Only {len(valid_indices)} sequences with length <= {max_len}, " 
                  f"but requested {n} samples. Sampling with replacement.")
        
        sampled_indices = self.rng.choice(valid_indices, size=n, replace=True)
        
        return ([self.train[i] for i in sampled_indices],
                [self.train_scores[i] for i in sampled_indices])

    def validation_set(self):
        return self.valid, self.valid_scores

    def add(self, batch):
        samples, scores = batch
        
        # In direct mode, scores are already in [0,1], no normalization needed
        if self.use_direct_mode:
            normalized_scores = np.clip(scores, 0, 1)  # Just ensure valid range
        else:
            normalized_scores = self.normalize_scores(scores)
        
        train, val = [], []
        train_seq, val_seq = [], []
        
        for x, score in zip(samples, normalized_scores):
            if np.random.uniform() < 0.1:
                val_seq.append(x)
                val.append(score)
            else:
                train_seq.append(x)
                train.append(score)
        
        if train:
            self.train_scores = np.concatenate((self.train_scores, train), axis=0).reshape(-1)
            self.train.extend(train_seq)
        
        if val:
            self.valid_scores = np.concatenate((self.valid_scores, val), axis=0).reshape(-1)
            self.valid.extend(val_seq)
            
    def _tostr(self, batch_tokens):
        return [self.tokenizer.detokenize(tokens) for tokens in batch_tokens]

    def _top_k(self, data, k, denormalize=True):
        """Sample k sequences that span the entire target range."""
        scores = data[1]
        tokens = data[0]
        
        target_low = self.args.target_logcdil_low
        target_high = self.args.target_logcdil_high
        denorm_scores = self.denormalize_scores(scores) if self.is_normalized else scores
        
        # Find indices within target range
        in_range_mask = (denorm_scores >= target_low) & (denorm_scores <= target_high)
        in_range_indices = np.where(in_range_mask)[0]
        
        if len(in_range_indices) == 0:
            # Fall back to closest to range
            distances = np.minimum(
                np.abs(denorm_scores - target_low),
                np.abs(denorm_scores - target_high)
            )
            indices = np.argsort(distances)[:k]
        elif len(in_range_indices) <= k:
            indices = in_range_indices
        else:
            # Stratified sampling: divide range into k bins
            in_range_scores = denorm_scores[in_range_indices]
            
            # Create bins across the target range
            bin_edges = np.linspace(target_low, target_high, k + 1)
            bin_assignments = np.digitize(in_range_scores, bin_edges[1:-1])
            
            selected_indices = []
            for bin_idx in range(k):
                bin_mask = bin_assignments == bin_idx
                bin_candidates = in_range_indices[bin_mask]
                
                if len(bin_candidates) > 0:
                    # Pick one from this bin (could randomize or pick closest to bin center)
                    bin_center = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
                    bin_scores = denorm_scores[bin_candidates]
                    best_in_bin = bin_candidates[np.argmin(np.abs(bin_scores - bin_center))]
                    selected_indices.append(best_in_bin)
            
            # If some bins were empty, fill remaining slots
            if len(selected_indices) < k:
                remaining = set(in_range_indices) - set(selected_indices)
                remaining_scores = denorm_scores[list(remaining)]
                # Fill with evenly spaced samples from remaining
                additional_needed = k - len(selected_indices)
                if len(remaining) > 0:
                    remaining_list = list(remaining)
                    step = max(1, len(remaining_list) // additional_needed)
                    for i in range(0, len(remaining_list), step):
                        if len(selected_indices) >= k:
                            break
                        selected_indices.append(remaining_list[i])
            
            indices = np.array(selected_indices[:k])
        
        topk_scores = scores[indices]
        if denormalize and not self.use_direct_mode:
            topk_scores = self.denormalize_scores(topk_scores)
        
        topk_tokens = [tokens[i] for i in indices]
        return self._tostr(topk_tokens), topk_scores

    def top_k(self, k, denormalize=True):
        tokens = self.train + self.valid
        scores = np.concatenate((self.train_scores, self.valid_scores), axis=0)
        data = (tokens, scores)
        return self._top_k(data, k, denormalize=denormalize)

    def top_k_collected(self, k, denormalize=True):
        tokens = self.train[self.train_added:] + self.valid[self.val_added:]
        scores = np.concatenate(
            (self.train_scores[self.train_added:], self.valid_scores[self.val_added:]),
            axis=0
        )
        data = (tokens, scores)
        return self._top_k(data, k, denormalize=denormalize)
    
    def get_dataset_size(self):
        return {
            "train_size": len(self.train),
            "valid_size": len(self.valid),
            "total_size": len(self.train) + len(self.valid),
            "train_collected": len(self.train) - self.train_added,
            "valid_collected": len(self.valid) - self.val_added,
            "total_collected": (len(self.train) - self.train_added) + (len(self.valid) - self.val_added)
        }
    
    def get_statistics(self, denormalize=True):
        all_scores = np.concatenate((self.train_scores, self.valid_scores), axis=0)
        
        if denormalize and not self.use_direct_mode:
            all_scores = self.denormalize_scores(all_scores)
        
        all_seqs = self.train + self.valid
        seq_lengths = [len(seq) for seq in all_seqs]
        
        return {
            "score_mean": np.mean(all_scores),
            "score_std": np.std(all_scores),
            "score_min": np.min(all_scores),
            "score_max": np.max(all_scores),
            "seq_length_mean": np.mean(seq_lengths),
            "seq_length_std": np.std(seq_lengths),
            "seq_length_min": np.min(seq_lengths),
            "seq_length_max": np.max(seq_lengths),
            "vocab_size": len(self.tokenizer.stoi),
            "dataset_size": len(all_seqs)
        }
    

def get_dataset(args, tokenizer):
    dataset = IDPDataset(args=args, tokenizer=tokenizer)
    dataset.map_normalize_y()
    return dataset