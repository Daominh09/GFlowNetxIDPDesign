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
        self._load_dataset(args.train_path, args.test_size)
        self.train_added = len(self.train)
        self.val_added = len(self.valid)

    def _load_dataset(self, csv_path, test_size):
        df = pd.read_csv(csv_path)
        seqs = df['fasta'].values
        dg = df['dG'].values
        batch_tokens = []
        batch_dg = []
        for seq, dg_val in zip(seqs, dg):
            if pd.isna(dg_val):
                continue
            tokens = self.tokenizer.tokenize(seq)
            if tokens is not None:
                batch_tokens.append(tokens)
                batch_dg.append(dg_val)
        
        batch_dg = np.array(batch_dg)
        self.train, self.valid, self.train_scores, self.valid_scores = train_test_split(
            batch_tokens, batch_dg, test_size=test_size, random_state=self.rng
        )
        
    def map_normalize_y(self):
        all_scores = np.concatenate((self.train_scores, self.valid_scores), axis=0)
        self.y_mean = np.mean(all_scores)
        self.y_std = np.std(all_scores)
        
        self.train_scores = (self.train_scores - self.y_mean) / self.y_std
        self.valid_scores = (self.valid_scores - self.y_mean) / self.y_std
        self.is_normalized = True
    
    def normalize_scores(self, scores):
        if not self.is_normalized:
            return scores
        if isinstance(scores, (list, np.ndarray)):
            return (np.array(scores) - self.y_mean) / self.y_std
        else:
            return (scores - self.y_mean) / self.y_std
    
    def denormalize_scores(self, scores):
        if not self.is_normalized:
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
    
    def sample_replay(self, n, shaper, device, max_len=None, alpha=0.5, beta=0.1):
        if max_len is None:
            indices = np.random.randint(0, len(self.train), n)
            return ([self.train[i] for i in indices],
                    [self.train_scores[i] for i in indices])
        
        valid_indices = np.array([i for i in range(len(self.train)) if len(self.train[i]) <= max_len])

        if len(valid_indices) == 0:
            raise ValueError(f"No training sequences with length <= {max_len}")
        
        if len(valid_indices) < n:
            print(f"Warning: Only {len(valid_indices)} sequences with length <= {max_len}, " 
                f"but requested {n} samples. Sampling with replacement.")
        
        # Get valid scores
        valid_scores = self.train_scores[valid_indices]
        
        # Convert to tensor (handle device appropriately)
        valid_tensor = torch.tensor(valid_scores, device=device, dtype=torch.float32)
        valid_reward = shaper.from_label(valid_tensor, already_normalized=True)
        
        # Ensure reward is numpy array for percentile calculation
        if isinstance(valid_reward, torch.Tensor):
            valid_reward = valid_reward.cpu().numpy()
        
        # Calculate threshold and masks
        threshold = np.percentile(valid_reward, (1-beta)*100)
        high_reward_mask = valid_reward >= threshold
        low_reward_mask = valid_reward < threshold
        
        # Map back to original indices
        high_reward_idx = valid_indices[high_reward_mask]
        low_reward_idx = valid_indices[low_reward_mask]
        
        # Handle edge cases
        if len(high_reward_idx) == 0:
            print(f"Warning: No high-reward sequences found (all rewards < {threshold:.4f}). Sampling uniformly.")
            sampled_indices = self.rng.choice(valid_indices, n, replace=True)
            return ([self.train[i] for i in sampled_indices],
                    [self.train_scores[i] for i in sampled_indices])
        
        if len(low_reward_idx) == 0:
            print(f"Warning: No low-reward sequences found (all rewards >= {threshold:.4f}). Sampling only high-reward.")
            sampled_indices = self.rng.choice(high_reward_idx, n, replace=True)
            return ([self.train[i] for i in sampled_indices],
                    [self.train_scores[i] for i in sampled_indices])
        
        # Normal sampling
        n_high = int(n * alpha)
        n_low = n - n_high
        
        high_samples = self.rng.choice(high_reward_idx, n_high, replace=True)
        low_samples = self.rng.choice(low_reward_idx, n_low, replace=True)
        sampled_indices = np.concatenate([high_samples, low_samples])
        return ([self.train[i] for i in sampled_indices],
                [self.train_scores[i] for i in sampled_indices])

    def validation_set(self):
        return self.valid, self.valid_scores

    def add(self, batch):
        samples, scores = batch
        
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
        scores = data[1]                  
        tokens = data[0]
        
        if self.args.proxy_mode == "range":
            target_low = self.args.target_dg_low
            target_high = self.args.target_dg_high
            denorm_scores = self.denormalize_scores(scores) if self.is_normalized else scores
            lower_margin = denorm_scores - target_low  # positive if above lower bound
            upper_margin = target_high - denorm_scores  # positive if below upper bound
            
            # Calculate fitness score based on preference direction
            if self.args.preference_direction == 1:
                # Prefer higher values within range
                fitness = np.where(
                    (denorm_scores >= target_low) & (denorm_scores <= target_high),
                    denorm_scores,  # In range: use actual value (higher is better)
                    -np.abs(np.minimum(lower_margin, upper_margin))  # Out of range: penalize by distance
                )
            elif self.args.preference_direction == -1:
                # Prefer lower values within range
                fitness = np.where(
                    (denorm_scores >= target_low) & (denorm_scores <= target_high),
                    -denorm_scores,  # In range: use negative value (lower is better)
                    -np.abs(np.minimum(lower_margin, upper_margin))  # Out of range: penalize by distance
                )
            else:  # preference_direction == 0
                fitness = np.where(
                    (denorm_scores >= target_low) & (denorm_scores <= target_high),
                    np.minimum(denorm_scores - target_low, target_high - denorm_scores),  # In range: distance to nearest boundary
                    -np.abs(np.minimum(lower_margin, upper_margin))  # Out of range: penalize by distance
                )
            
            # Get top k based on fitness
            indices = np.argsort(fitness)[::-1][:k]
            topk_scores = scores[indices]
            
        else:
            # Original behavior: find sequences closest to target_y
            target = self.normalize_scores(self.args.target_y)
            distance = abs(scores - target)  
            indices = np.argsort(distance)[::-1][-k:]
            topk_scores = scores[indices]
        
        if denormalize:
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
        
        if denormalize:
            all_scores = self.denormalize_scores(all_scores)
        
        all_seqs = self.train + self.valid
        seq_lengths = [len(seq) for seq in all_seqs]
        
        return {
            "dg_mean": np.mean(all_scores),
            "dg_std": np.std(all_scores),
            "dg_min": np.min(all_scores),
            "dg_max": np.max(all_scores),
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