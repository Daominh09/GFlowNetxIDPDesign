import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm
import pickle
import gzip

from lib.dataset import get_dataset
from lib.generator import get_generator
from lib.oracle import get_oracle, ProtVec
from lib.proxy import get_proxy
from lib.tokenizer import get_tokenizer
from lib.args import get_default_args
from lib.utils import Model, AttrSetter, edit_dist


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
            ys = self.f([i[0] for i in self.stack]) # eos_tok == 2
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
        
        # Use reward shaper instead of l2r
        self.reward_shaper = reward_shaper

    def rollout(self, model, episodes, use_rand_policy=True):
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
                logits = model(x, None, coef=self.out_coef)
            if t == 0:
                logits[:, 0] = -1000 # Prevent model from stopping
                                     # without having output anything
            
            cat = Categorical(logits=logits / self.sampling_temperature)
            actions = cat.sample()
            if use_rand_policy and self.random_action_prob > 0:
                for i in range(actions.shape[0]):
                    if np.random.uniform(0,1) < self.random_action_prob:
                        actions[i] = torch.tensor(np.random.randint(t == 0, logits.shape[1])).to(self.device)
            
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


    def execute_train_episode_batch(self, model, it=0, dataset=None, use_rand_policy=True):
        # run an episode
        lists = lambda n: [list() for i in range(n)]
        visited, states, traj_states, \
            traj_actions, traj_rewards, traj_dones = self.rollout(model, self.episodes_per_step, use_rand_policy=use_rand_policy) 
        lens = np.mean([len(i) for i in traj_rewards])
        bulk_trajs = []
        rq = []
        
        # Process generated trajectories
        for (r, mbidx) in self.workers.pop_all():
            # r is the oracle score (normalized), convert to reward using shaper
            reward = r
            traj_rewards[mbidx][-1] = reward
            rq.append(r.item())
            s = states[mbidx]
            visited.append((s, reward, r.item()))
            bulk_trajs.append((s, reward))
        
        # Add dataset samples
        if args.gen_data_sample_per_step > 0 and dataset is not None:
            n = args.gen_data_sample_per_step
            m = len(traj_states)
            if it > 300:
                x, y = dataset.sample_replay(n, self.reward_shaper, self.device, self.max_len, alpha=0.5, beta=0.1)
            x, y = dataset.sample(n, self.max_len)  # y is already normalized
            n = len(x)
            traj_states += lists(n)
            traj_actions += lists(n)
            traj_rewards += lists(n)
            traj_dones += lists(n)
            
            # Convert normalized labels to rewards using reward shaper
            y_tensor = torch.tensor(y, device=self.device, dtype=torch.float32)
            rewards = self.reward_shaper.from_label(y_tensor, already_normalized=True)
            bulk_trajs += list(zip(x, rewards.cpu().tolist()))
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
        
        
        return {
            "visited": visited,
            "trajectories": {
                "traj_states": traj_states,
                "traj_actions": traj_actions,
                "traj_rewards": traj_rewards,
                "traj_dones": traj_dones,
                "states": states,
                "bulk_trajs": bulk_trajs
            }
        }

def filter_len(x, y, max_len):
    res = ([], [])
    for i in range(len(x)):
        if len(x[i]) < max_len:
            res[0].append(x[i])
            res[1].append(y[i])
    return res
  
def train_generator(args, generator, oracle, tokenizer, dataset, logged_data, reward_shaper):
    print("Training generator")
    visited = []
    rollout_worker = RolloutWorker(args, oracle, tokenizer, reward_shaper=reward_shaper)
    for it in tqdm(range(args.gen_num_iterations + 1)):
        rollout_artifacts = rollout_worker.execute_train_episode_batch(generator, it, dataset)
        visited.extend(rollout_artifacts["visited"])
        loss, loss_info = generator.train_step(rollout_artifacts["trajectories"])
        
        # Log generator loss
        if 'generator_total_loss' not in logged_data:
            logged_data['generator_total_loss'] = []
        logged_data['generator_total_loss'].append(loss.item())
        
        for key, val in loss_info.items():
            full_key = f"generator_{key}"
            if full_key not in logged_data:
                logged_data[full_key] = []
            logged_data[full_key].append(val.item())
        
        if it % 100 == 0:
            print([i[-1] for i in rollout_artifacts["trajectories"]["traj_rewards"]])
            rs = torch.tensor([i[-1] for i in rollout_artifacts["trajectories"]["traj_rewards"]]).mean().item()
            if 'gen_reward' not in logged_data:
                logged_data['gen_reward'] = []
            logged_data['gen_reward'].append(rs)
            print(f"Iteration {it}, Generator Loss: {loss.item():.6f}, Reward: {rs:.6f}")
        
        if it % 100 == 0:
            save_data(args.save_path, logged_data, args)

        # Call during training
        if it % 50 == 0:
            analyze_action_distribution(rollout_worker, generator, tokenizer)
    
    return rollout_worker, None

def sample_batch(args, rollout_worker, generator, current_dataset, oracle):
    print("Generating samples")
    samples = ([], [])
    scores = []
    while len(samples[0]) < args.num_sampled_per_round * 5:
        rollout_artifacts = rollout_worker.execute_train_episode_batch(generator, it=0, use_rand_policy=False)
        states = rollout_artifacts["trajectories"]["states"]
        vals = oracle.batch_predict(states).reshape(-1)
        samples[0].extend(states)
        samples[1].extend(vals)
        scores.extend(torch.tensor(rollout_artifacts["trajectories"]["traj_rewards"])[:, -1].numpy().tolist())
    idx_pick = np.argsort(scores)[::-1][:args.num_sampled_per_round]
    return (np.array(samples[0])[idx_pick].tolist(), np.array(samples[1])[idx_pick].tolist())

def mean_pairwise_distances(seqs):
    dists = []
    for pair in itertools.combinations(seqs, 2):
        dists.append(edit_dist(*pair))
    return np.mean(dists)

def log_overall_metrics(dataset, logged_data, round_num, collected=False):
    top100 = dataset.top_k(128)
    logged_data[f'round_{round_num}_top_128_scores'] = np.mean(top100[1])
    dist100 = mean_pairwise_distances(top100[0])
    logged_data[f'round_{round_num}_top_128_dists'] = dist100
    logged_data[f'round_{round_num}_top_128_seqs'] = top100[0]
    print(f"Round {round_num} - Top 128 Scores: {np.mean(top100[1]):.6f}, Dists: {dist100:.6f}")
    
    if collected:
        top100 = dataset.top_k_collected(128)
        logged_data[f'round_{round_num}_top_128_collected_scores'] = np.mean(top100[1])
        logged_data[f'round_{round_num}_max_128_collected_scores'] = np.max(top100[1])
        dist100 = mean_pairwise_distances(top100[0])
        logged_data[f'round_{round_num}_top_128_collected_dists'] = dist100
        logged_data[f'round_{round_num}_top_128_collected_seqs'] = top100[0]
        print(f"Round {round_num} - Collected Scores: Mean={np.mean(top100[1]):.6f}, Max={np.max(top100[1]):.6f}, Median={np.percentile(top100[1], 50):.6f}")
        print(f"Round {round_num} - Collected Dists: {dist100:.6f}")

def save_data(save_path, logged_data, args):
    import json
    import numpy as np
    import torch
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


def train(args, oracle, dataset, tokenizer):
    logged_data = {}
    
    proxy = get_proxy(args, tokenizer, dataset=dataset)
    proxy.update(dataset)
    
    for round in range(args.num_rounds):
        print(f"\n=== Starting Round {round+1}/{args.num_rounds} ===")
        generator = get_generator(args, tokenizer)
        
        # Pass the reward shaper from proxy
        rollout_worker, losses = train_generator(
            args, generator, proxy, tokenizer, dataset, logged_data, 
            reward_shaper=proxy.shaper
        )
        
        batch = sample_batch(args, rollout_worker, generator, dataset, oracle)
        
        logged_data[f'round_{round+1}_collected_seqs'] = batch[0]
        logged_data[f'round_{round+1}_collected_scores'] = batch[1]
        
        dataset.add(batch)
        log_overall_metrics(dataset, logged_data, round+1, collected=True)
        
        if round != args.num_rounds - 1: 
            proxy.update(dataset)
        
        save_data(args.save_path, logged_data, args)

# After some training iterations, check action distribution:
def analyze_action_distribution(rollout_worker, generator, tokenizer, n_episodes=16):
    print("\n=== Action Distribution Analysis ===")
    
    _, states, _, _, _, _ = rollout_worker.rollout(generator, n_episodes, use_rand_policy=False)
    
    # Count amino acid frequencies
    from collections import Counter
    all_actions = []
    for state in states:
        all_actions.extend(state)
    
    action_counts = Counter(all_actions)
    total = len(all_actions)
    
    print(f"Total actions: {total}")
    print(f"Unique amino acids used: {len(action_counts)}/20")
    print("\nTop 10 amino acids:")
    for aa_idx, count in action_counts.most_common(10):
        aa = tokenizer.itos[aa_idx] if aa_idx < len(tokenizer.itos) else f"IDX{aa_idx}"
        pct = 100 * count / total
        print(f"  {aa}: {count} ({pct:.1f}%)")
    
    # Check if dominated by single AA
    top_aa_pct = action_counts.most_common(1)[0][1] / total
    if top_aa_pct > 0.5:
        print(f"\n⚠️  WARNING: One amino acid dominates {top_aa_pct*100:.1f}% of sequences!")



def main(args):
    torch.manual_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    np.random.seed(args.seed)
    args.device = torch.device('cpu')
    tokenizer = get_tokenizer(args)
    oracle = get_oracle(args, tokenizer)
    dataset = get_dataset(args, tokenizer)

    train(args, oracle, dataset, tokenizer)


if __name__ == "__main__":
    args = get_default_args()
    main(args)