import torch
import numpy as np
from torch.distributions import Categorical
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional

from gfnxidp import get_dataset
from gfnxidp import get_generator
from gfnxidp import get_oracle
from gfnxidp import get_proxy, IDPConstraintPenalty
from gfnxidp import get_tokenizer
from gfnxidp import get_default_args
from gfnxidp.utils import Model, AttrSetter
from gfnxidp.generator import PropertyNormalizer

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


class MbStack:
    def __init__(self, f):
        self.stack = []
        self.f = f

    def push(self, x, i):
        self.stack.append((x, i))

    def pop_all(self):
        if not len(self.stack):
            return []
        with torch.no_grad():
            ys = self.f.batch_predict([i[0] for i in self.stack])
        idxs = [i[1] for i in self.stack]
        self.stack = []
        return zip(ys, idxs)


class RolloutWorker:
    def __init__(self, args, oracle, tokenizer, reward_shaper=None):
        self.oracle = oracle
        self.max_len = args.max_len
        self.max_len = args.gen_max_len
        self.episodes_per_step = args.gen_episodes_per_step
        self.random_action_prob = args.gen_random_action_prob
        self.sampling_temperature = args.gen_sampling_temperature
        self.out_coef = args.gen_output_coef

        self.balanced_loss = args.gen_balanced_loss == 1
        self.loss_eps = torch.tensor(float(args.gen_loss_eps)).to(args.device)
        self.leaf_coef = args.gen_leaf_coef
        
        self.tokenizer = tokenizer
        self.device = args.device
        self.args = args
        self.workers = MbStack(oracle)
        
        # Conditioning
        self.condition_normalizer = PropertyNormalizer(args)
        self.target_csat_values = None
        
        # Use reward shaper instead of l2r
        self.reward_shaper = reward_shaper
        self.constraint_penalty = IDPConstraintPenalty(args, tokenizer)

    def rollout(self, model, episodes, use_rand_policy=True, condition=None):
        """
        Execute rollout trajectories
        
        Args:
            model: Generator model
            episodes: Number of episodes to generate
            use_rand_policy: Whether to use random exploration
            condition: torch.Tensor of normalized CSAT targets [episodes, 1]
        
        Returns:
            Tuple of (visited, states, traj_states, traj_actions, traj_rewards, traj_dones)
        """
        visited = []
        lists = lambda n: [list() for i in range(n)]
        states = [[] for i in range(episodes)]
        traj_states = [[[]] for i in range(episodes)]
        traj_actions = lists(episodes)
        traj_rewards = lists(episodes)
        traj_dones = lists(episodes)

        traj_logprob = np.zeros(episodes)

        for t in (range(self.max_len) if episodes > 0 else []):
            
            x = self.tokenizer.pad_tokens(states).to(self.device)
            lens = torch.tensor([len(i) for i in states]).long().to(self.device)
            
            with torch.no_grad():
                logits = model(x, None, condition=condition, coef=self.out_coef)
            
            if t == 0:
                logits[:, 0] = -1000  # Prevent model from stopping without output
                                     
            
            cat = Categorical(logits=logits / self.sampling_temperature)
            actions = cat.sample()
            
            if use_rand_policy and self.random_action_prob > 0:
                for i in range(actions.shape[0]):
                    if np.random.uniform(0, 1) < self.random_action_prob:
                        actions[i] = torch.tensor(np.random.randint(1 if t == 0 else 0, logits.shape[1])).to(self.device)
            
            for i, a in enumerate(actions):
                if t == self.max_len - 1:
                    self.workers.push(states[i] + [a.item()], i)
                    r = 0
                    d = 1
                else:
                    r = 0
                    d = 0
                traj_states[i].append(states[i] + [a.item()])
                traj_actions[i].append(a)
                traj_rewards[i].append(r)
                traj_dones[i].append(d)
                states[i] += [a.item()]
        
        return visited, states, traj_states, traj_actions, traj_rewards, traj_dones


    def execute_train_episode_batch(self, model, it=0, dataset=None, use_rand_policy=True, target_csat=None):
        """
        Execute training episode batch with conditioning
        
        Args:
            model: Generator model
            it: Current iteration
            dataset: Dataset for sampling
            use_rand_policy: Whether to use random exploration
            target_csat: torch.Tensor of target CSAT values [episodes_per_step]
        
        Returns:
            Dictionary with trajectories and condition values
        """
        if it % 200 == 0 and it != 0 and it != self.args.gen_num_iterations:
            self.reward_shaper.lambda_decay *= 0.5
        
        lists = lambda n: [list() for i in range(n)]
        
        # Prepare conditioning
        if target_csat is not None:
            if not torch.is_tensor(target_csat):
                target_csat = torch.tensor(target_csat, dtype=torch.float32)
            if target_csat.device != self.device:
                target_csat = target_csat.to(self.device)
            if target_csat.dim() == 1:
                target_csat = target_csat.unsqueeze(1)  # [batch, 1]
        else:
            # If not specified, sample random targets from distribution
            target_csat = torch.randn(self.episodes_per_step, 1, device=self.device) * 0.5
        
        visited, states, traj_states, \
            traj_actions, traj_rewards, traj_dones = self.rollout(
                model, self.episodes_per_step, use_rand_policy=use_rand_policy,
                condition=target_csat
            )
        
        lens = np.mean([len(i) for i in traj_rewards])
        bulk_trajs = []
        rq = []
        
        # Process generated trajectories
        for (r, mbidx) in self.workers.pop_all():
            # r is the oracle score (normalized), convert to reward using shaper
            oracle_score = torch.tensor(r, device=self.device, dtype=torch.float32)
            base_reward = self.reward_shaper.from_label(oracle_score, target_csat[mbidx], already_normalized=False)
            seq = self.tokenizer.detokenize(states[mbidx])
            print(base_reward)
            if self.constraint_penalty is not None:
                penalty, constraint_info = self.constraint_penalty.compute_total_penalty(seq)
                reward = base_reward * penalty
            else:
                reward = base_reward

            if it != 0 and it % 100 == 0:
                if constraint_info:
                    print(f"  Constraint penalty: {penalty:.3f} | {constraint_info}")
            
            traj_rewards[mbidx][-1] = reward
            rq.append(r.item())
            s = states[mbidx]
            visited.append((s, reward, r.item()))
            bulk_trajs.append((s, reward))

        dataset_target_csat = torch.zeros(0, 1, device=self.device)
        # Add dataset samples
        if self.args.gen_data_sample_per_step > 0 and dataset is not None:
            n = self.args.gen_data_sample_per_step
            m = len(traj_states)
            x, y = dataset.sample(n, self.args.max_len)  # y is already normalized
            n = len(x)
            traj_states += lists(n)
            traj_actions += lists(n)
            traj_rewards += lists(n)
            traj_dones += lists(n)
    
            sequences = [self.tokenizer.detokenize(tokens) for tokens in x]
            # Convert normalized labels to rewards using reward shaper
            y_tensor = torch.tensor(y, device=self.device, dtype=torch.float32)
            dataset_target_csat = target_csat
            base_rewards = self.reward_shaper.from_label(y_tensor, dataset_target_csat, already_normalized=True)
            print(base_rewards)
            if self.constraint_penalty is not None:
                penalties = self.constraint_penalty.batch_penalties(sequences)
                rewards = base_rewards * penalties
            else:
                rewards = base_rewards

            bulk_trajs += list(zip(x, rewards.cpu().tolist()))
            # Sample corresponding target CSAT values for dataset samples
            for i in range(len(x)):
                traj_states[i+m].append([])
                for c, a in zip(x[i], self.tokenizer.pad_tokens([x[i]]).reshape(-1)):
                    traj_states[i+m].append(traj_states[i+m][-1] + [c])
                    traj_actions[i+m].append(a)
                    # Only assign reward at the terminal state
                    if len(traj_actions[i+m]) == len(x[i]):
                        traj_rewards[i+m].append(rewards[i].item())
                    else:
                        traj_rewards[i+m].append(0)
                    traj_dones[i+m].append(float(len(traj_rewards[i+m]) == len(x[i])))
        
        # Combine condition values: generated + dataset
        all_condition_values = torch.cat([target_csat, dataset_target_csat], dim=0)
        
        return {
            "visited": visited,
            "trajectories": {
                "traj_states": traj_states,
                "traj_actions": traj_actions,
                "traj_rewards": traj_rewards,
                "traj_dones": traj_dones,
                "states": states,
                "bulk_trajs": bulk_trajs
            },
            "condition_values": all_condition_values
        }


def filter_len(x, y, max_len):
    res = ([], [])
    for i in range(len(x)):
        if len(x[i]) < max_len:
            res[0].append(x[i])
            res[1].append(y[i])
    return res


def sample_csat_distribution(dataset, batch_size, normalizer, device):
    """
    Sample target CSAT values from a normal distribution based on dataset statistics
    
    Args:
        dataset: IDPDataset object
        batch_size: Number of samples to generate
        normalizer: PropertyNormalizer (for reference/verification only)
        device: Torch device
    
    Returns:
        torch.Tensor of normalized CSAT values [batch_size, 1]
    """
    # Get mean and std from dataset
    mean_score = dataset.y_mean
    score_std = dataset.y_std
    
    # Sample from normal distribution N(mean, std)
    sampled_scores = np.random.normal(0, 1, size=batch_size)
    
    # Convert to tensor [batch_size, 1]
    condition_tensor = torch.tensor(sampled_scores, dtype=torch.float32, device=device).unsqueeze(1)
    
    return condition_tensor


def train_generator(args, generator, oracle, tokenizer, dataset, logged_data, reward_shaper):
    """
    Train the conditional generator
    
    Args:
        args: Configuration arguments
        generator: Generator model
        oracle: Oracle for evaluation
        tokenizer: Tokenizer for sequences
        dataset: Dataset for sampling
        logged_data: Dictionary to log metrics
        reward_shaper: Reward shaping module
    
    Returns:
        Tuple of (rollout_worker, losses)
    """
    print("Training conditional generator on diverse CSAT targets")
    visited = []
    rollout_worker = RolloutWorker(args, oracle, tokenizer, reward_shaper=reward_shaper)
    normalizer = PropertyNormalizer(args)
    
    for it in tqdm(range(args.gen_num_iterations + 1)):
        # Sample diverse target CSAT values from dataset distribution
        target_csat_batch = sample_csat_distribution(
            dataset, args.gen_episodes_per_step, normalizer, args.device
        )
        rollout_artifacts = rollout_worker.execute_train_episode_batch(
            generator, it, dataset, target_csat=target_csat_batch
        )
        visited.extend(rollout_artifacts["visited"])
        condition_values = rollout_artifacts["condition_values"]
        
        # Train step with condition
        loss, loss_info = generator.train_step(
            rollout_artifacts["trajectories"],
            condition_values=condition_values
        )
        print(loss_info)
        # Log generator loss
        if 'generator_total_loss' not in logged_data:
            logged_data['generator_total_loss'] = []
        logged_data['generator_total_loss'].append(loss.item())
        
        for key, val in loss_info.items():
            full_key = f"generator_{key}"
            if full_key not in logged_data:
                logged_data[full_key] = []
            logged_data[full_key].append(val.item())
        
        if it % 10 == 0:
            rs = torch.tensor([i[-1] for i in rollout_artifacts["trajectories"]["traj_rewards"][:args.gen_episodes_per_step]]).mean().item()
            if 'gen_reward' not in logged_data:
                logged_data['gen_reward'] = []
            logged_data['gen_reward'].append(rs)
            print(f"Iteration {it}, Generator Loss: {loss.item():.6f}, Reward: {rs:.6f}")
        
        if it % 100 == 0:
            save_data(args.save_path, logged_data, args)

        # Call during training
        if it % 50 == 0:
            analyze_generation_quality(rollout_worker, generator, tokenizer, normalizer=normalizer)

    
    return rollout_worker, None


def sample_batch(args, rollout_worker, generator, dataset, oracle, target_csat=None):
    """
    Sample high-reward sequences
    
    Args:
        args: Configuration arguments
        rollout_worker: Rollout worker
        generator: Generator model
        dataset: Dataset for normalization
        oracle: Oracle for evaluation
        target_csat: Optional target CSAT value for zero-shot generation (normalized)
    
    Returns:
        Tuple of (sequences, scores)
    """
    normalizer = PropertyNormalizer(args)
    
    if target_csat is not None:
        print(f"Generating samples (target CSAT normalized: {target_csat})")
        condition_val = torch.full(
            (args.gen_episodes_per_step, 1),
            target_csat,
            dtype=torch.float32,
            device=args.device
        )
    else:
        print("Generating samples")
        condition_val = None
    
    samples = ([], [])
    scores = []
    
    while len(samples[0]) < args.num_sampled_per_round * 1:
        rollout_artifacts = rollout_worker.execute_train_episode_batch(
            generator, it=0, use_rand_policy=False, target_csat=condition_val
        )
        states = rollout_artifacts["trajectories"]["states"]
        vals = oracle.batch_predict(states).reshape(-1)
        samples[0].extend(states)
        samples[1].extend(vals)
        scores.extend(torch.tensor(rollout_artifacts["trajectories"]["traj_rewards"])[:, -1].numpy().tolist())
    
    idx_pick = np.argsort(scores)[::-1][:args.num_sampled_per_round]
    return (np.array(samples[0])[idx_pick].tolist(), np.array(samples[1])[idx_pick].tolist())

def log_overall_metrics(dataset, logged_data, round_num, collected=False):
    """Log overall metrics for the round"""
    top100 = dataset.top_k(128)
    logged_data[f'round_{round_num}_top_128_scores'] = np.mean(top100[1])
    logged_data[f'round_{round_num}_top_128_seqs'] = top100[0]
    print(f"Round {round_num} - Top 128 Scores: {np.mean(top100[1]):.6f}")
    
    if collected:
        top100 = dataset.top_k_collected(128)
        logged_data[f'round_{round_num}_top_128_collected_scores'] = np.mean(top100[1])
        logged_data[f'round_{round_num}_max_128_collected_scores'] = np.max(top100[1])
        logged_data[f'round_{round_num}_top_128_collected_seqs'] = top100[0]
        print(f"Round {round_num} - Collected Scores: Mean={np.mean(top100[1]):.6f}, Max={np.max(top100[1]):.6f}, Min={np.min(top100[1]):.6f}, Median={np.percentile(top100[1], 50):.6f}")


def save_data(save_path, logged_data, args):
    """Save logged data to JSON"""
    import json
    import os

    if not save_path.endswith(".json"):
        save_path = os.path.splitext(save_path)[0] + ".json"

    data_to_save = {
        "logged_data": logged_data,
        "args": vars(args)
    }

    def to_serializable(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.item() if obj.ndim == 0 else obj.cpu().tolist()
        return str(obj)

    # Save as readable JSON
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=2, ensure_ascii=False, default=to_serializable)

    print(f"Data saved to {save_path}")


def save_model(generator, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'model_state_dict': generator.model.state_dict(),
        'Z': generator.model.Z.item(),
    }
    
    torch.save(checkpoint, save_path)
    print(f"✓ Model saved to: {save_path}")


def train(args, oracle, dataset, tokenizer):
    """
    Main training loop with conditional generation
    
    Args:
        args: Configuration arguments
        oracle: Oracle for evaluation
        dataset: Initial dataset
        tokenizer: Tokenizer for sequences
    """
    logged_data = {}
    
    proxy = get_proxy(args, tokenizer, dataset=dataset)
    
    for round in range(args.num_rounds):
        print(f"\n=== Starting Round {round+1}/{args.num_rounds} ===")
        generator = get_generator(args, tokenizer)
        
        # Train generator on diverse CSAT targets
        rollout_worker, losses = train_generator(
            args, generator, oracle, tokenizer, dataset, logged_data, 
            reward_shaper=proxy.shaper
        )
        
        # Sample batch (can specify target_csat for zero-shot)
        target_csat_raw = -1.0  # Original scale
        target_csat_normalized = dataset.normalize_scores(target_csat_raw)
        batch = sample_batch(args, rollout_worker, generator, dataset, oracle, target_csat=target_csat_normalized)
        
        logged_data[f'round_{round+1}_collected_seqs'] = batch[0]
        logged_data[f'round_{round+1}_collected_scores'] = batch[1]
        
        dataset.add(batch)
        log_overall_metrics(dataset, logged_data, round+1, collected=True)
        
        save_data(args.save_path, logged_data, args)
    
    model_path = args.model_path
    save_model(generator, model_path)
    print("\n✓ Training complete!")


def analyze_generation_quality(rollout_worker, generator, tokenizer, n_episodes=16, normalizer=None):
    """
    Comprehensive analysis including constraints and motifs
    
    Args:
        rollout_worker: Rollout worker
        generator: Generator model
        tokenizer: Tokenizer
        n_episodes: Number of episodes to analyze
        normalizer: PropertyNormalizer for conditioning
    """
    print("\n=== Generation Quality Analysis ===")
    rollout_worker.workers.stack = []
    # Generate with random condition or no condition
    if normalizer is not None:
        condition = torch.randn(n_episodes, 1, device=rollout_worker.device) * 0.5
    else:
        condition = None
    
    _, states, _, _, _, _ = rollout_worker.rollout(generator, n_episodes, use_rand_policy=False, condition=condition)
    sequences = [tokenizer.detokenize(s) for s in states]
    rollout_worker.workers.pop_all()
    # Constraint analysis
    if rollout_worker.constraint_penalty is not None:
        print("\n--- Constraint Analysis ---")
        violations = {'cysteine': 0, 'hydrophobic': 0, 'aromatic': 0, 'any': 0}
        penalty_stats = []
        
        for seq in sequences:
            penalty, breakdown = rollout_worker.constraint_penalty.compute_total_penalty(seq)
            penalty_stats.append(penalty)
            
            cys_frac = seq.count('C') / len(seq) if len(seq) > 0 else 0
            hydro_frac = sum(1 for aa in seq if aa in 'AILMFWVY') / len(seq) if len(seq) > 0 else 0
            arom_frac = sum(1 for aa in seq if aa in 'FWY') / len(seq) if len(seq) > 0 else 0
            
            if cys_frac > 0.03:
                violations['cysteine'] += 1
            if hydro_frac > 0.35:
                violations['hydrophobic'] += 1
            if arom_frac < 0.08 or arom_frac > 0.12:
                violations['aromatic'] += 1
            if penalty < 0.99:
                violations['any'] += 1
        
        n = len(sequences)
        print(f"Sequences: {n}")
        print(f"Mean constraint penalty: {np.mean(penalty_stats):.3f}")
        print(f"Violations: Cys={violations['cysteine']}, Hydro={violations['hydrophobic']}, "
              f"Arom={violations['aromatic']}, Any={violations['any']}")
    
    # Action distribution
    from collections import Counter
    all_actions = []
    for state in states:
        all_actions.extend(state)
    
    action_counts = Counter(all_actions)
    total = len(all_actions)
    print(f"\n--- Amino Acid Distribution ---")
    print(f"Total actions: {total}, Unique AAs: {len(action_counts)}")
    print("Top 10:")
    for aa_idx, count in action_counts.most_common(10):
        aa = tokenizer.itos[aa_idx] if aa_idx < len(tokenizer.itos) else f"IDX{aa_idx}"
        print(f"  {aa}: {count} ({100*count/total:.1f}%)")


def main(args):
    """Main entry point"""
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    np.random.seed(args.seed)
    args.device = torch.device('mps')
    tokenizer = get_tokenizer(args)
    oracle = get_oracle(args, tokenizer)
    dataset = get_dataset(args, tokenizer)

    train(args, oracle, dataset, tokenizer)


if __name__ == "__main__":
    args = get_default_args()
    main(args)