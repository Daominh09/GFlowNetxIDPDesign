import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from lib.generator import MLP

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
        target_dg_low_original = getattr(args, 'target_dg_low', -8.0)
        target_dg_high_original = getattr(args, 'target_dg_high', -6.0)
        
        # Normalize the target ranges to match the normalized score space
        self.c_low = dataset.normalize_scores(target_dg_low_original)
        self.c_high = dataset.normalize_scores(target_dg_high_original)
        
        self.preference_direction = getattr(args, 'preference_direction', 0)
        
        # Decay parameter (like λ in the paper)
        # NOTE: This should also be adjusted for normalized scale
        self.lambda_decay = getattr(args, 'reward_lambda', 1.0)
        
        # Power transformation
        self.reward_exp = getattr(args, 'gen_reward_exp', 2.0)
        self.reward_exp_ramping = getattr(args, 'gen_reward_exp_ramping', 0.0)
        
        print(f"\nRangeTargetRewardShaper initialized (NORMALIZED version):")
        print(f"  Target range (original): [{target_dg_low_original}, {target_dg_high_original}]")
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
    
class TargetRewardShaper:
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        self.device = args.device
        self.y_star_norm = self.dataset.normalize_scores(args.target_y)
        
        # Reward transformation parameters (like l2r in original paper)
        self.reward_exp = getattr(args, 'gen_reward_exp', 2.0)
        self.reward_exp_ramping = getattr(args, 'gen_reward_exp_ramping', 3.0)

    def from_score(self, s, std=None, iteration=0):
        args = self.args
        s = _safe_tensor(s, self.device).view(-1)
        mode = getattr(args, "proxy_mode", "gaussian").lower()
        # Compute base reward in [0, 1] based on distance from target
        if mode == "gaussian":
            y_star = _safe_tensor(self.y_star_norm, self.device)
            if args.use_uncertainty_in_width and std is not None:
                std = _safe_tensor(std, self.device).view(-1)
                var = std.pow(2) + (args.target_tau ** 2)
            else:
                var = _safe_tensor(args.target_tau ** 2, self.device).expand_as(s)
            var = torch.clamp(var, min=1e-12)
            
            # Base Gaussian reward [0, 1]
            r_base = torch.exp(- (s - y_star).pow(2) / (2.0 * var))

        elif mode == "interval":
            y_star = _safe_tensor(self.y_star_norm, self.device)
            eps = _safe_tensor(args.target_eps, self.device)
            if args.use_uncertainty_in_width and std is not None:
                std = _safe_tensor(std, self.device).view(-1)
                sig = torch.sqrt(std.pow(2) + (args.target_tau ** 2))
            else:
                sig = _safe_tensor(args.target_tau, self.device).expand_as(s)
            sig = torch.clamp(sig, min=1e-6)
            
            # CDF of standard normal
            Phi = lambda z: 0.5 * (1.0 + torch.erf(z / torch.sqrt(torch.tensor(2.0, device=self.device))))
            z_hi = (y_star + eps - s) / sig
            z_lo = (y_star - eps - s) / sig
            r_base = torch.clamp(Phi(z_hi) - Phi(z_lo), 0.0, 1.0)

        elif mode == "laplace":
            y_star = _safe_tensor(self.y_star_norm, self.device)
            b = _safe_tensor(args.target_b, self.device)
            b = torch.clamp(b, min=1e-6)
            r_base = torch.exp(- torch.abs(s - y_star) / b)

        else:
            # Fallback: just clamp scores
            r_base = torch.clamp(s, min=0.0, max=1.0)
        
        if self.reward_exp_ramping > 0:
            t = iteration
            current_exp = 1 + (self.reward_exp - 1) * (1 - 1/(1 + t / self.reward_exp_ramping))
        else:
            # No ramping: constant exponent
            current_exp = self.reward_exp
        r_transformed = r_base.pow(current_exp)
        return r_transformed
    
    def from_label(self, y_list_or_array, already_normalized=None, iteration=0):
        if already_normalized is None:
            already_normalized = getattr(self.dataset, "is_normalized", False)

        if already_normalized:
            y_norm = y_list_or_array
        else:
            y_norm = self.dataset.normalize_scores(y_list_or_array)

        y_norm = _safe_tensor(y_norm, self.device).view(-1)
        return self.from_score(y_norm, std=None, iteration=iteration)


    
class GaussianTargetProxy:
    """Proxy with Gaussian reward shaping around target"""
    def __init__(self, args, model, dataset):
        self.args = args
        self.model = model
        self.dataset = dataset
        self.device = args.device
        self.shaper = TargetRewardShaper(args, dataset)

    def fit(self, data):
        self.model.fit(data, reset=True)

    def update(self, data):
        self.fit(data)

    def __call__(self, x, iteration=0):
        """
        Compute rewards for sequences x at training iteration.
        
        Args:
            x: List of token sequences
            iteration: Current training iteration (for reward ramping)
            
        Returns:
            r: Reward tensor in (0,1]
        """
        mean, std = self.model.forward_with_uncertainty(x)  # [N,1] each
        mean = mean.view(-1)
        std = std.view(-1)
        r = self.shaper.from_score(mean, std, iteration=iteration)
        return r




def _safe_tensor(x, device, dtype=torch.float32):
    """Convert various types to torch tensor safely"""
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)

class DirectRewardShaper:
    """
    Direct reward shaper for scores already in [0,1] range
    No normalization, denormalization, or transformation needed
    Use case: Deep phase scores, probabilities, or any pre-normalized metrics
    """
    def __init__(self, args, dataset=None):
        self.args = args
        self.dataset = dataset
        self.device = args.device
        
        # Optional: Apply power transformation even for direct rewards
        self.reward_exp = getattr(args, 'gen_reward_exp', 1.0)  # Default 1.0 = no transform
        self.reward_exp_ramping = getattr(args, 'gen_reward_exp_ramping', 0.0)
        
        # Clipping bounds (to ensure valid reward range)
        self.min_reward = getattr(args, 'reward_min_clip', 1e-6)
        self.max_reward = getattr(args, 'reward_max_clip', 1.0)
        
        print(f"\nDirectRewardShaper initialized:")
        print(f"  Score range: [0, 1] (no transformation needed)")
        print(f"  Reward exponent: {self.reward_exp}")
        print(f"  Reward ramping: {self.reward_exp_ramping}")
        print(f"  Clip range: [{self.min_reward}, {self.max_reward}]")

    def from_score(self, s, std=None, iteration=0):
        """
        Use predicted scores directly as rewards
        
        Args:
            s: Predicted scores in [0,1] range (e.g., deep phase scores)
            std: Standard deviation (optional, for uncertainty-based adjustments)
            iteration: Current training iteration (for reward ramping)
            
        Returns:
            r: Reward tensor in [min_reward, max_reward]
        """
        s = _safe_tensor(s, self.device).view(-1)
        
        # Clip to valid range
        r_base = torch.clamp(s, min=0.0, max=1.0)
        
        # Optional: Apply power transformation with ramping
        if self.reward_exp_ramping > 0:
            t = iteration
            current_exp = 1 + (self.reward_exp - 1) * (1 - 1/(1 + t / self.reward_exp_ramping))
        else:
            current_exp = self.reward_exp
        
        if current_exp != 1.0:
            r_transformed = r_base.pow(current_exp)
        else:
            r_transformed = r_base
        
        # Final clipping to ensure minimum reward for numerical stability
        r_final = torch.clamp(r_transformed, min=self.min_reward, max=self.max_reward)
        
        return r_final
    
    def from_label(self, y, already_normalized=None, iteration=0):
        """
        Convert oracle labels to rewards (no normalization needed)
        
        Args:
            y: Oracle predictions already in [0,1] range
            already_normalized: Ignored for direct mode
            iteration: Current training iteration
            
        Returns:
            r: Reward tensor
        """
        y = _safe_tensor(y, self.device).view(-1)
        return self.from_score(y, std=None, iteration=iteration)


class DirectRewardProxy:
    """
    Proxy for direct reward usage (no normalization/denormalization)
    Use when predicted scores are already in [0,1] range
    """
    def __init__(self, args, model, dataset):
        self.args = args
        self.model = model
        self.dataset = dataset
        self.device = args.device
        self.shaper = DirectRewardShaper(args, dataset)
        
        print(f"\nDirectRewardProxy initialized")
        print(f"  Model will predict scores in [0,1] range directly")
        print(f"  No normalization/denormalization applied")

    def fit(self, data):
        """Train proxy model on raw [0,1] scores"""
        self.model.fit(data, reset=True)

    def update(self, data):
        """Update proxy with new data"""
        self.fit(data)

    def __call__(self, x, iteration=0):
        """
        Compute rewards for sequences x
        
        Args:
            x: List of token sequences
            iteration: Current training iteration
            
        Returns:
            r: Direct reward from predicted scores
        """
        mean, std = self.model.forward_with_uncertainty(x)
        mean = mean.view(-1)
        
        # Use mean prediction directly as reward
        r = self.shaper.from_score(mean, std=std, iteration=iteration)
        
        return r

def _safe_tensor(x, device, dtype=torch.float32):
    """Convert various types to torch tensor safely"""
    if torch.is_tensor(x):
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, device=device, dtype=dtype)


def get_proxy(args, tokenizer, dataset):
    """Factory function to create appropriate proxy based on mode"""
    model = DropoutRegressor(args, tokenizer)

    mode = getattr(args, "proxy_mode", "gaussian").lower()
    
    if mode == "gaussian":
        proxy = GaussianTargetProxy(args, model, dataset)
    elif mode == "range":
        proxy = RangeTargetProxy(args, model, dataset)
    elif mode == "direct":  # NEW MODE
        proxy = DirectRewardProxy(args, model, dataset)
    else:
        raise ValueError(f"Unknown proxy_mode: {mode}. Choose from: 'gaussian', 'range', 'direct'")

    # Expose common adapter methods for compatibility
    proxy.score_to_reward = lambda s, std=None, it=0: proxy.shaper.from_score(s, iteration=it)
    proxy.label_to_reward = lambda y, norm=None, it=0: proxy.shaper.from_label(y, already_normalized=norm, iteration=it)
    
    print(f"\n{'='*60}")
    print(f"Proxy Configuration:")
    print(f"  Mode: {mode}")
    if mode == "range":
        print(f"  Target ΔG range: [{args.target_dg_low}, {args.target_dg_high}]")
        print(f"  Preference direction: {args.preference_direction}")
    elif mode == "direct":
        print(f"  Direct reward mode (scores in [0,1])")
    else:
        print(f"  Target ΔG: {args.target_y}")
    print(f"  Reward exponent: {proxy.shaper.reward_exp}")
    print(f"  Reward ramping factor: {proxy.shaper.reward_exp_ramping}")
    print(f"{'='*60}\n")
    
    return proxy