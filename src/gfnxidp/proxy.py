import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from gfnxidp.generator import MLP
from typing import List, Optional, Tuple

class DropoutRegressor(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.num_tokens = args.vocab_size
        self.max_len = args.max_len
        self.tokenizer = tokenizer
        self.init_model()
        self.sigmoid = nn.Sigmoid()
        self.proxy_num_iterations = args.proxy_num_iterations
        
        self.device = args.device

    def init_model(self):
        self.model = MLP(num_tokens=self.num_tokens,
                            num_outputs=1,
                            num_hid=self.args.proxy_num_hid,
                            num_layers=self.args.proxy_num_layers,
                            dropout=self.args.proxy_dropout,
                            max_len=self.max_len)
        self.model.to(self.args.device)
        self.opt = torch.optim.Adam(self.model.parameters(), self.args.proxy_learning_rate,
                            weight_decay=self.args.proxy_L2)

    def fit(self, data, reset=False):
        losses = []
        test_losses = []
        best_params = None
        best_loss = 1e6
        early_stop_tol = self.args.proxy_early_stop_tol
        early_stop_count = 0
        epoch_length = 100
        if reset:
            self.init_model()
        

        for it in range(self.proxy_num_iterations):
            x, y = data.sample(self.args.proxy_num_per_minibatch)

            x = self.tokenizer.pad_tokens(x).to(self.device)
            inp_x = F.one_hot(x, num_classes=self.num_tokens+1)[:, :, :-1].to(torch.float32)
            inp = torch.zeros(x.shape[0], self.max_len, self.num_tokens)
            inp[:, :inp_x.shape[1], :] = inp_x
            x = inp.reshape(x.shape[0], -1).to(self.device).detach()
            y = torch.tensor(y, device=self.device, dtype=torch.float).reshape(-1)
            
            output = self.model(x, None).squeeze(1)
            loss = (output - y).pow(2).mean()
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
            
            losses.append(loss.item())
            if it % 10 == 0:
                print(f"Iteration {it}, Train Loss: {loss.item():.6f}")
            
            if not it % epoch_length:
                vx, vy = data.validation_set()
                vlosses = []
                n_samples = 0
                batch_size = 256
                
                # Process all validation samples, including the last incomplete batch
                for j in range(0, len(vx), batch_size):
                    # Get batch (will be smaller than batch_size for last batch)
                    batch_x = vx[j:j+batch_size]
                    batch_y = vy[j:j+batch_size]
                    batch_len = len(batch_x)
                    
                    # Tokenize and preprocess
                    x = self.tokenizer.pad_tokens(batch_x).to(self.device)
                    inp_x = F.one_hot(x, num_classes=self.num_tokens+1)[:, :, :-1].to(torch.float32)
                    inp = torch.zeros(x.shape[0], self.max_len, self.num_tokens)
                    inp[:, :inp_x.shape[1], :] = inp_x
                    x = inp.reshape(x.shape[0], -1).to(self.device).detach()
                    y = torch.tensor(batch_y, device=self.device, dtype=torch.float).reshape(-1)
                    
                    # Compute loss
                    with torch.no_grad():
                        output = self.model(x, None).squeeze(1)
                        loss = (output - y).pow(2)
                    
                    # Accumulate sum of squared errors and count
                    vlosses.append(loss.sum().item())
                    n_samples += batch_len

                # Compute mean validation loss
                test_loss = np.sum(vlosses) / n_samples
                test_losses.append(test_loss)
                print(f"Epoch {it//epoch_length}, Validation Loss: {test_loss:.6f}, Samples: {n_samples}/{len(vx)}")
                
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_params = [i.data.cpu().numpy() for i in self.model.parameters()]
                    early_stop_count = 0
                else:
                    early_stop_count += 1

                if early_stop_count >= early_stop_tol:
                    print(f'Early stopping at iteration {it}, best validation loss: {best_loss:.6f}')
                    break

        if self.args.proxy_early_stop_to_best_params:
            # Put best parameters back in
            for i, besti in zip(self.model.parameters(), best_params):
                i.data = torch.tensor(besti).to(self.device)
        return {}

    def forward(self, curr_x, uncertainty_call=False):
        x = self.tokenizer.pad_tokens(curr_x).to(self.device)
        inp_x = F.one_hot(x, num_classes=self.num_tokens+1)[:, :, :-1].to(torch.float32)
        inp = torch.zeros(x.shape[0], self.max_len, self.num_tokens)
        inp[:, :inp_x.shape[1], :] = inp_x
        x = inp.reshape(x.shape[0], -1).to(self.device).detach()
        if uncertainty_call:
            ys = self.model(x, None).unsqueeze(0)
        else:
            self.model.eval()
            ys = self.model(x, None)
            self.model.train()

        return ys
    
    def forward_with_uncertainty(self, x):
        self.model.train()
        with torch.no_grad():
            outputs = torch.cat([self.forward(x, True) for _ in range(self.args.proxy_num_dropout_samples)])
        return outputs.mean(dim=0), outputs.std(dim=0)

    def save(self, path):
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.load_state_dict(path)

class RangeTargetRewardShaper:
    """
    Reward shaper based on A-GFN's property-conditioned reward (Equation 2)
    Supports target ranges [c_low, c_high] with preference direction
    Works in NORMALIZED score space
    """
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        self.device = args.device
        
        # Target range parameters (in ORIGINAL scale)
        target_logcdil_low_original = getattr(args, 'target_logcdil_low', -8.0)
        target_logcdil_high_original = getattr(args, 'target_logcdil_high', -6.0)
        
        # Normalize the target ranges to match the normalized score space
        self.c_low = dataset.normalize_scores(target_logcdil_low_original)
        self.c_high = dataset.normalize_scores(target_logcdil_high_original)
        
        self.preference_direction = getattr(args, 'preference_direction', 0)
        
        # Decay parameter (like λ in the paper)
        # NOTE: This should also be adjusted for normalized scale
        self.lambda_decay = getattr(args, 'reward_lambda', 1.0)
        
        # Power transformation
        self.reward_exp = getattr(args, 'gen_reward_exp', 2.0)
        self.reward_exp_ramping = getattr(args, 'gen_reward_exp_ramping', 0.0)
        
        print(f"\nRangeTargetRewardShaper initialized (NORMALIZED version):")
        print(f"  Target range (original): [{target_logcdil_low_original}, {target_logcdil_high_original}]")
        print(f"  Target range (normalized): [{self.c_low:.4f}, {self.c_high:.4f}]")
        print(f"  Preference direction: {self.preference_direction}")
        print(f"  Lambda decay: {self.lambda_decay}")
        print(f"  Reward exponent: {self.reward_exp}")

    def from_score(self, p_x, iteration=0, this=False):
        """
        Compute reward based on A-GFN's Equation 2/6/7
        
        Args:
            p_x: Predicted property values (ΔG scores in NORMALIZED space)
            iteration: Current training iteration (for reward ramping)
            
        Returns:
            r: Reward in (0, 1]
        """
        p_x = _safe_tensor(p_x, self.device).view(-1)
        c_low = _safe_tensor(self.c_low, self.device)
        c_high = _safe_tensor(self.c_high, self.device)
        lambda_decay = _safe_tensor(self.lambda_decay, self.device)
        
        # Three cases based on where p_x falls
        below_range = p_x < c_low
        above_range = p_x > c_high
        in_range = ~(below_range | above_range)
        r_base = torch.zeros_like(p_x)
        
        if self.preference_direction > 0:
            # Prefer higher values (Equation 2)
            r_base[below_range] = 0.5 * torch.exp(-(c_low - p_x[below_range]) / lambda_decay)
            r_base[above_range] = torch.exp(-(p_x[above_range] - c_high) / lambda_decay)
            r_base[in_range] = 0.5 * (p_x[in_range] - c_low) / (c_high - c_low) + 0.5
            
        elif self.preference_direction < 0:
            # Prefer lower values (Equation 6)
            r_base[below_range] = torch.exp(-(c_low - p_x[below_range]) / lambda_decay)
            r_base[above_range] = 0.5 * torch.exp(-(p_x[above_range] - c_high) / lambda_decay)
            r_base[in_range] = -0.5 * (p_x[in_range] - c_low) / (c_high - c_low) + 1.0
            
        else:
            # No preference, just want to be in range (Equation 7)
            r_base[below_range] = torch.exp(-(c_low - p_x[below_range]) / lambda_decay)
            r_base[above_range] = torch.exp(-(p_x[above_range] - c_high) / lambda_decay)
            r_base[in_range] = 1.0
        return r_base
    
    def from_label(self, y, already_normalized=False, iteration=0):
        """
        Convert oracle labels (ΔG values) to rewards
        
        Args:
            y: Oracle predictions (ΔG values)
            already_normalized: Whether y is already normalized
            iteration: Current training iteration
            
        Returns:
            r: Reward tensor
        """
        y = _safe_tensor(y, self.device).view(-1)
        if not already_normalized:
            # y is in original scale, normalize it
            y_np = y.cpu().numpy() if torch.is_tensor(y) else y
            y_norm = self.dataset.normalize_scores(y_np)
            y = _safe_tensor(y_norm, self.device).view(-1)
        return self.from_score(y, iteration=iteration)

class RangeTargetProxy:
    """Proxy with range-based reward shaping (A-GFN style)"""
    def __init__(self, args, model, dataset):
        self.args = args
        self.model = model
        self.dataset = dataset
        self.device = args.device
        self.shaper = RangeTargetRewardShaper(args, dataset)
        
        print(f"\nRangeTargetProxy initialized with target range reward shaping")

    def fit(self, data):
        self.model.fit(data, reset=True)

    def update(self, data):
        self.fit(data)

    def __call__(self, x, iteration=0):
        """
        Compute rewards for sequences x
        
        Args:
            x: List of token sequences
            iteration: Current training iteration
            
        Returns:
            r: Range-based reward
        """
        mean, std = self.model.forward_with_uncertainty(x)
        mean = mean.view(-1)
        # Use mean prediction for reward (can incorporate uncertainty if desired)
        r = self.shaper.from_score(mean, iteration=iteration, this=True)
        
        return r



    
class IDPConstraintPenalty:
    """
    Applies biophysical constraint penalties for IDP design.
    All penalties are multiplicative factors in (0, 1].
    """
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.device = args.device
        
        # Constraint thresholds (configurable via args)
        self.cys_max_fraction = getattr(args, 'cys_max_fraction', 0.03)
        self.hydrophobic_max_fraction = getattr(args, 'hydrophobic_max_fraction', 0.35)
        self.aromatic_min_fraction = getattr(args, 'aromatic_min_fraction', 0.08)
        self.aromatic_max_fraction = getattr(args, 'aromatic_max_fraction', 0.12)
        self.min_disorder_score = getattr(args, 'min_disorder_score', 0.65)
        
        # Penalty strengths (how harsh the penalty is)
        self.cys_penalty_strength = getattr(args, 'cys_penalty_strength', 10.0)
        self.hydrophobic_penalty_strength = getattr(args, 'hydrophobic_penalty_strength', 10.0)
        self.aromatic_penalty_strength = getattr(args, 'aromatic_penalty_strength', 10.0)
        self.disorder_penalty_strength = getattr(args, 'disorder_penalty_strength', 5.0)
        
        # Amino acid groups
        self.hydrophobic_aa = set('AILMFWVY')
        self.aromatic_aa = set('FWY')
        self.cysteine = 'C'
        
        # Optional: IUPred2A path for disorder prediction
        self.iupred_path = getattr(args, 'iupred2a', None)
        self.use_disorder_filter = getattr(args, 'use_disorder_filter', False)
        
        self._print_config()
    
    def _print_config(self):
        print(f"\n{'='*60}")
        print("IDP Constraint Penalties Initialized:")
        print(f"  Cysteine: max {self.cys_max_fraction*100:.1f}% (penalty strength: {self.cys_penalty_strength})")
        print(f"  Hydrophobic: max {self.hydrophobic_max_fraction*100:.1f}% (penalty strength: {self.hydrophobic_penalty_strength})")
        print(f"  Aromatic: {self.aromatic_min_fraction*100:.1f}%-{self.aromatic_max_fraction*100:.1f}% (penalty strength: {self.aromatic_penalty_strength})")
        if self.use_disorder_filter:
            print(f"  Disorder: min score {self.min_disorder_score}, (penelty strength: {self.disorder_penalty_strength})")
        print(f"{'='*60}\n")
    
    def compute_cysteine_penalty(self, seq: str) -> float:
        """Penalty if cysteine fraction > threshold"""
        if len(seq) == 0:
            return 1.0
        cys_frac = seq.count(self.cysteine) / len(seq)
        if cys_frac <= self.cys_max_fraction:
            return 1.0
        excess = cys_frac - self.cys_max_fraction
        return np.exp(-self.cys_penalty_strength * excess)
    
    def compute_hydrophobic_penalty(self, seq: str) -> float:
        """Penalty if hydrophobic fraction > threshold"""
        if len(seq) == 0:
            return 1.0
        hydro_count = sum(1 for aa in seq if aa in self.hydrophobic_aa)
        hydro_frac = hydro_count / len(seq)
        if hydro_frac <= self.hydrophobic_max_fraction:
            return 1.0
        excess = hydro_frac - self.hydrophobic_max_fraction
        return np.exp(-self.hydrophobic_penalty_strength * excess)
    
    def compute_aromatic_penalty(self, seq: str) -> float:
        """Penalty if aromatic fraction outside [min, max] range"""
        if len(seq) == 0:
            return 1.0
        arom_count = sum(1 for aa in seq if aa in self.aromatic_aa)
        arom_frac = arom_count / len(seq)
        
        if self.aromatic_min_fraction <= arom_frac <= self.aromatic_max_fraction:
            return 1.0
        
        if arom_frac < self.aromatic_min_fraction:
            deficit = self.aromatic_min_fraction - arom_frac
        else:
            deficit = arom_frac - self.aromatic_max_fraction
        return np.exp(-self.aromatic_penalty_strength * deficit)
    
    def compute_disorder_penalty(self, seq: str) -> float:
        """
        Penalty based on predicted disorder score.
        Requires IUPred2A to be available.
        """
        if not self.use_disorder_filter or self.iupred_path is None:
            return 1.0
        
        try:
            disorder_scores = self._predict_disorder(seq)
            if disorder_scores is None:
                return 1.0
            
            # Penalty for low average disorder
            avg_disorder = np.mean(disorder_scores)
            if avg_disorder < self.min_disorder_score:
                deficit = self.min_disorder_score - avg_disorder
                avg_penalty = np.exp(-self.disorder_penalty_strength * deficit)
            else:
                avg_penalty = 1.0
            
            return avg_penalty
            
        except Exception as e:
            print(f"Disorder prediction failed: {e}")
            return 1.0
    
    def _predict_disorder(self, seq: str) -> Optional[np.ndarray]:
        """Run IUPred2A to get per-residue disorder scores"""
        import tempfile
        import subprocess
        import os
        
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.fasta')
        try:
            with open(tmp.name, 'w') as f:
                f.write(f'>seq\n{seq}')
            
            result = subprocess.run(
                ['python', self.iupred_path, tmp.name, 'long'],
                capture_output=True, text=True, timeout=30
            )
            
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
            return np.array(scores) if scores else None
            
        finally:
            os.unlink(tmp.name)
    
    
    def compute_total_penalty(self, seq: str) -> Tuple[float, dict]:
        """
        Compute combined penalty from all constraints.
        Returns (total_penalty, breakdown_dict)
        """
        cys_p = self.compute_cysteine_penalty(seq)
        hydro_p = self.compute_hydrophobic_penalty(seq)
        arom_p = self.compute_aromatic_penalty(seq)
        disorder_p = self.compute_disorder_penalty(seq)

        total = cys_p * hydro_p * arom_p * disorder_p
        
        breakdown = {
            'cysteine': cys_p,
            'hydrophobic': hydro_p,
            'aromatic': arom_p,
            'disorder': disorder_p,
            'total': total
        }
        return total, breakdown
    
    def batch_penalties(self, sequences: List[str]) -> torch.Tensor:
        """Compute penalties for a batch of sequences"""
        penalties = [self.compute_total_penalty(seq)[0] for seq in sequences]
        return torch.tensor(penalties, device=self.device, dtype=torch.float32)




def _safe_tensor(x, device, dtype=torch.float32):
    """Convert various types to torch tensor safely"""
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def get_proxy(args, tokenizer, dataset):
    """Factory function to create appropriate proxy based on mode"""
    model = DropoutRegressor(args, tokenizer)

    mode = getattr(args, "proxy_mode", "gaussian").lower()
    proxy = RangeTargetProxy(args, model, dataset)

    # Expose common adapter methods for compatibility
    proxy.score_to_reward = lambda s, std=None, it=0: proxy.shaper.from_score(s, iteration=it)
    proxy.label_to_reward = lambda y, norm=None, it=0: proxy.shaper.from_label(y, already_normalized=norm, iteration=it)
    
    print(f"\n{'='*60}")
    print(f"Proxy Configuration:")
    print(f"  Mode: {mode}")
    if mode == "range":
        print(f"  Target ΔG range: [{args.target_logcdil_low}, {args.target_logcdil_high}]")
        print(f"  Preference direction: {args.preference_direction}")
    elif mode == "direct":
        print(f"  Direct reward mode (scores in [0,1])")
    else:
        print(f"  Target ΔG: {args.target_y}")
    print(f"  Reward exponent: {proxy.shaper.reward_exp}")
    print(f"  Reward ramping factor: {proxy.shaper.reward_exp_ramping}")
    print(f"{'='*60}\n")
    
    return proxy