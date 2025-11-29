import torch
import torch.nn as nn
import torch.nn.functional as F

LOGINF = 1000


class PropertyNormalizer:
    """Normalize CSAT values to [-1, 1] range for stable conditioning"""
    def __init__(self, args):
        self.csat_mean = getattr(args, 'csat_mean', -7.0)
        self.csat_std = getattr(args, 'csat_std', 1.5)
        
    def normalize(self, csat_values):
        if torch.is_tensor(csat_values):
            normalized = (csat_values - self.csat_mean) / (self.csat_std + 1e-8)
            return normalized.clamp(-3, 3)
        return (csat_values - self.csat_mean) / (self.csat_std + 1e-8)
    
    def denormalize(self, normalized_values):
        if torch.is_tensor(normalized_values):
            return normalized_values * self.csat_std + self.csat_mean
        return normalized_values * self.csat_std + self.csat_mean


class ConditionEmbedding(nn.Module):
    """FiLM-style conditioning: produce γ and β."""
    def __init__(self, hidden_dim, condition_dim=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )

    def forward(self, condition):
        params = self.mlp(condition)
        gamma, beta = params.chunk(2, dim=-1)
        return gamma, beta


class MLP(nn.Module):
    """Improved MLP with better condition integration"""
    def __init__(self, num_tokens, num_outputs, num_hid,
                 num_layers, max_len=60, dropout=0.1,
                 partition_init=150.0, condition_dim=1,
                 **kwargs):
        super().__init__()
        
        self.sequence_embedding = nn.Linear(num_tokens * max_len, num_hid)
        self.condition_embedding = ConditionEmbedding(num_hid, condition_dim)
        
        hidden_layers = []
        for i in range(num_layers):
            hidden_layers.append(nn.Dropout(dropout))
            hidden_layers.append(nn.Linear(num_hid, num_hid))
            hidden_layers.append(nn.LayerNorm(num_hid))
            hidden_layers.append(nn.ReLU())
        
        self.hidden = nn.Sequential(*hidden_layers)
        self.output = nn.Linear(num_hid, num_outputs)
        self.property_head = nn.Linear(num_hid, 1)
        
        self.max_len = max_len
        self.condition_dim = condition_dim
        self.num_tokens = num_tokens
        self._Z = nn.Parameter(torch.ones(64) * partition_init / 64)

    @property
    def Z(self):
        return self._Z.sum()

    def model_params(self):
        return (list(self.sequence_embedding.parameters()) + 
                list(self.condition_embedding.parameters()) +
                list(self.hidden.parameters()) + 
                list(self.output.parameters()) +
                list(self.property_head.parameters()))

    def Z_param(self):
        return [self._Z]

    def encode(self, x, condition=None):
        x_embed = self.sequence_embedding(x)
        if condition is not None:
            condition = torch.clamp(condition, -3, 3)
            gamma, beta = self.condition_embedding(condition)
            x_embed = x_embed * (1 + gamma) + beta
        h = self.hidden(x_embed)
        return h

    def forward(self, x, mask, condition=None, return_all=False, lens=None):
        batch_size = x.shape[0]

        gamma = beta = None
        if condition is not None:
            condition = torch.clamp(condition, -3, 3)
            gamma, beta = self.condition_embedding(condition)

        if return_all:
            outputs = []
            for i in range(self.max_len):
                causal_mask = torch.cat(
                    (torch.ones(batch_size, self.num_tokens * i),
                     torch.zeros(batch_size, self.num_tokens * (self.max_len - i))),
                    dim=1
                ).to(x.device)

                masked_input = causal_mask * x
                x_masked_embed = self.sequence_embedding(masked_input)
                
                if gamma is not None:
                    x_masked_embed = x_masked_embed * (1 + gamma) + beta

                h = self.hidden(x_masked_embed)
                outputs.append(self.output(h).unsqueeze(0))
            return torch.cat(outputs, dim=0)

        h = self.encode(x, condition=condition)
        return self.output(h)


class TBGFlowNetGenerator(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.leaf_coef = args.gen_leaf_coef
        self.out_coef = args.gen_output_coef
        self.reward_exp_min = args.reward_exp_min
        self.loss_eps = torch.tensor(float(args.gen_loss_eps)).to(args.device)
        self.pad_tok = 1
        self.num_tokens = args.vocab_size
        self.max_len = args.max_len
        self.tokenizer = tokenizer
        
        self.condition_dim = 1
        self.condition_normalizer = PropertyNormalizer(args)
        
        self.model = MLP(
            num_tokens=self.num_tokens, 
            num_outputs=self.num_tokens, 
            num_hid=1024,
            num_layers=2,
            max_len=self.max_len,
            dropout=0.1,
            partition_init=args.gen_partition_init,
            condition_dim=self.condition_dim
        )
        self.model.to(args.device)
        
        self.opt = torch.optim.Adam(
            self.model.model_params(),
            args.gen_learning_rate, 
            weight_decay=args.gen_L2,
            betas=(0.9, 0.999)
        )
        self.opt_Z = torch.optim.Adam(
            self.model.Z_param(),
            args.gen_Z_learning_rate, 
            weight_decay=args.gen_L2,
            betas=(0.9, 0.999)
        )
        
        self.device = args.device
        self.logsoftmax = torch.nn.LogSoftmax(1)
        self.logsoftmax2 = torch.nn.LogSoftmax(2)
        
        self.condition_effect_log = []
        self.property_loss_coef = getattr(args, "property_loss_coef", 1.0)

    def train_step(self, input_batch, condition_values=None):
        loss, info = self.get_loss(input_batch, condition_values)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gen_clip)
        self.opt.step()
        self.opt_Z.step()
        self.opt.zero_grad()
        self.opt_Z.zero_grad()
        return loss, info

    @property
    def Z(self):
        return self.model.Z
    
    def get_loss(self, batch, condition_values=None):
        strs, r = zip(*batch["bulk_trajs"])
        s = self.tokenizer.pad_tokens(strs).to(self.device)
        r = torch.tensor(r).to(self.device).clamp(min=0)
        r = torch.nan_to_num(r, nan=0, posinf=0, neginf=0)
        
        inp_x = F.one_hot(s, num_classes=self.num_tokens+1)[:, :, :-1].to(torch.float32)
        inp = torch.zeros(s.shape[0], self.max_len, self.num_tokens)
        inp[:, :inp_x.shape[1], :] = inp_x
        x = inp.reshape(s.shape[0], -1).to(self.device).detach()
        
        if condition_values is not None:
            if not torch.is_tensor(condition_values):
                condition_values = torch.tensor(condition_values, dtype=torch.float32, device=self.device)
            elif condition_values.device != self.device:
                condition_values = condition_values.to(self.device)
            
            if condition_values.dim() == 1:
                condition_values = condition_values.unsqueeze(1)
            
            condition_normalized = condition_values
            if torch.rand(1) < 0.1:
                with torch.no_grad():
                    logits_with = self.model(x, None, condition=condition_normalized, return_all=True, lens=None)
                    logits_without = self.model(x, None, condition=None, return_all=True, lens=None)
                    cond_effect = (logits_with - logits_without).abs().mean().item()
                    self.condition_effect_log.append(cond_effect)
                    
                    if len(self.condition_effect_log) % 50 == 0:
                        avg_effect = sum(self.condition_effect_log[-50:]) / 50
                        if avg_effect < 0.01:
                            print(f"  WARNING: Low condition effect: {avg_effect:.6f}")
        else:
            condition_normalized = None
        
        lens = [len(i) for i in strs]
        pol_logits = self.logsoftmax2(
            self.model(x, None, condition=condition_normalized, return_all=True, lens=lens)
        )[:-1]
        mask = s.eq(self.num_tokens)
        s = s.swapaxes(0, 1)
        n = (s.shape[0] - 1) * s.shape[1]
        seq_logits = (
            pol_logits
            .reshape((n, self.num_tokens))[torch.arange(n, device=self.device),(s[1:,].reshape((-1,))).clamp(0, self.num_tokens-1)]
            .reshape(s[1:].shape)
            * mask[:,1:].swapaxes(0,1).logical_not().float()
        ).sum(0)
        
        lens = torch.tensor([len(i) for i in strs]).float().to(self.device)
        seq_logits_norm = seq_logits / lens
        
        tb_loss = (
            self.model.Z.log()
            + seq_logits_norm
            - r.clamp(min=self.reward_exp_min).log()
        ).pow(2).mean()

        prop_loss = torch.tensor(0.0, device=self.device)
        if condition_normalized is not None and self.property_loss_coef > 0:
            h = self.model.encode(x, condition=condition_normalized)
            property_pred = self.model.property_head(h).view(-1)
            target = condition_normalized.view(-1).detach()
            prop_loss = F.mse_loss(property_pred, target)

        loss = tb_loss + self.property_loss_coef * prop_loss

        info = {
            "tb_loss": tb_loss.detach(),
            "prop_loss": prop_loss.detach()
        }

        return loss, info

    def forward(self, x, lens, condition=None, return_all=False, coef=1, pad=2, guidance_scale=10.0):
        inp_x = F.one_hot(x, num_classes=self.num_tokens+1)[:, :, :-1].to(torch.float32)
        inp = torch.zeros(x.shape[0], self.max_len, self.num_tokens)
        inp[:, :inp_x.shape[1], :] = inp_x
        inp = inp.reshape(x.shape[0], -1).to(self.device)
        
        if condition is not None:
            if not torch.is_tensor(condition):
                condition = torch.tensor(condition, dtype=torch.float32, device=self.device)
            elif condition.device != self.device:
                condition = condition.to(self.device)
            
            if condition.dim() == 1:
                condition = condition.unsqueeze(1)
            
            if guidance_scale > 1.0:
                out_with = self.model(inp, None, condition=condition, lens=lens, return_all=return_all)
                out_without = self.model(inp, None, condition=None, lens=lens, return_all=return_all)
                out = out_without + guidance_scale * (out_with - out_without)
            else:
                out = self.model(inp, None, condition=condition, lens=lens, return_all=return_all)
        else:
            out = self.model(inp, None, condition=None, lens=lens, return_all=return_all)
        
        return out * coef    


def get_generator(args, tokenizer):
    return TBGFlowNetGenerator(args=args, tokenizer=tokenizer)
