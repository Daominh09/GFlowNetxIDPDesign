
from itertools import chain
import numpy as np
import torch
from torch.distributions import Categorical
import wandb
from pathlib import Path
from tqdm import tqdm

from ..utils import tensor_to_np, batch, pack, unpack
from ..data import Experience

import copy, pickle
import numpy as np
from polyleven import levenshtein

import gflownet.trainers as trainers
from gflownet.GFNs import models
from gflownet.MDPs import seqpamdp, seqinsertmdp, seqarmdp
from gflownet.monitor import TargetRewardDistribution, Monitor

def dynamic_inherit_mdp(base, args):

  class TFBind8MDP(base):
    def __init__(self, args):
      super().__init__(args,
                       alphabet=list('0123'),
                       forced_stop_len=8)
      self.args = args

      # Read from file
      print(f'Loading data ...')
      with open('datasets/tfbind8/tfbind8-exact-v0-all.pkl', 'rb') as f:
        oracle_d = pickle.load(f)
      
      munge = lambda x: ''.join([str(c) for c in list(x)])
      self.oracle = {self.state(munge(x), is_leaf=True): float(y)
          for x, y in zip(oracle_d['x'], oracle_d['y'])}
      
      # Scale rewards
      self.scaled_oracle = copy.copy(self.oracle)
      py = np.array(list(self.scaled_oracle.values()))

      REWARD_EXP = 3
      py = py ** REWARD_EXP

      py = py * (args.scale_reward_max / max(py))
      py = np.maximum(py, 0.001)
      self.scaled_oracle = {x: y for x, y in zip(self.scaled_oracle.keys(), py)}

      # Rewards
      self.rs_all = [y for x, y in self.scaled_oracle.items()]

      # Modes
      with open('datasets/tfbind8/modes_tfbind8.pkl', 'rb') as f:
        modes = pickle.load(f)
      self.modes = set([self.state(munge(x), is_leaf=True) for x in modes])

    # Core
    def reward(self, x):
      assert x.is_leaf, 'Error: Tried to compute reward on non-leaf node.'
      return self.scaled_oracle[x]

    def is_mode(self, x, r):
      return x in self.modes

    '''
      Interpretation & visualization
    '''
    def dist_func(self, state1, state2):
      """ States are SeqPAState or SeqInsertState objects. """
      return levenshtein(state1.content, state2.content)

    def make_monitor(self):
      target = TargetRewardDistribution()
      target.init_from_base_rewards(self.rs_all)
      return Monitor(self.args, target, dist_func=self.dist_func,
                     is_mode_f=self.is_mode, callback=self.add_monitor)

    def add_monitor(self, xs, rs, allXtoR):
      """ Reimplement scoring with oracle, not unscaled oracle (used as R). """
      tolog = dict()
      return tolog

  return TFBind8MDP(args)


def main(args):
  print('Running experiment TFBind8 ...')

  if args.mdp_style == 'pa':
    base = seqpamdp.SeqPrependAppendMDP
    actorclass = seqpamdp.SeqPAActor
  elif args.mdp_style == 'insert':
    base = seqinsertmdp.SeqInsertMDP
    actorclass = seqinsertmdp.SeqInsertActor
  elif args.mdp_style == 'autoregressive':
    base = seqarmdp.SeqAutoregressiveMDP
    actorclass = seqarmdp.SeqARActor
  mdp = dynamic_inherit_mdp(base, args)

  actor = actorclass(args, mdp)
  model = models.make_model(args, mdp, actor)
  monitor = mdp.make_monitor()

  # Save memory, after constructing monitor with target rewards
  del mdp.rs_all

  trainer = trainers.Trainer(args, model, mdp, actor, monitor)
  trainer.learn()
  return


class BaseTBGFlowNet():
  """ Trajectory balance parameterization:
      logZ, forward policy, backward policy.
      Default behavior:
      - No parameter sharing between forward/backward policy
      - Separate optimizers for forward/backward policy; this is needed for
        guided TB. Logically unnecessary for regular TB, but equivalent to
        using a single optimizer for both policies.

      Forward and backward policy classes are specified by mdp.
  """
  def __init__(self, args, mdp, actor):
    self.args = args
    self.mdp = mdp
    self.actor = actor

    self.policy_fwd = actor.policy_fwd
    self.policy_back = actor.policy_back

    self.logZ = torch.nn.Parameter(torch.tensor([5.], device=self.args.device))
    self.logZ_lower = 0

    self.nets = [self.policy_fwd, self.policy_back]
    for net in self.nets:
      net.to(args.device)

    self.clip_grad_norm_params = [self.policy_fwd.parameters(),
                                  self.policy_back.parameters()]

    self.optimizer_back = torch.optim.Adam([
        {
          'params': self.policy_back.parameters(),
          'lr': args.lr_policy
        }])
    self.optimizer_fwdZ = torch.optim.Adam([
        {
          'params': self.policy_fwd.parameters(),     
          'lr': args.lr_policy
        }, {
          'params': self.logZ, 
          'lr': args.lr_z
        }])
    self.optimizers = [self.optimizer_fwdZ, self.optimizer_back]
    pass
  
  """
    logZ
  """
  def init_logz(self, val):
    print(f'Initializing Z to {val}. Using this as floor for clamping ...')
    self.logZ.data = torch.tensor([val],
        device=self.args.device, requires_grad=True)
    assert self.logZ.is_leaf
    self.logZ_lower = val
    return

  def clamp_logZ(self):
    """ Clamp logZ to min value. Default assumes logZ > 0 (Z > 1). """
    self.logZ.data = torch.clamp(self.logZ, min=self.logZ_lower)
    return

  """
    Forward and backward policy
  """
  def fwd_logps_unique(self, batch):
    """ Differentiable; output logps of unique children/parents.
    
        See policy.py : logps_unique for more info.

        Input: List of [State], n items
        Returns
        -------
        state_to_logp: List of dicts mapping state to torch.tensor
    """
    return self.policy_fwd.logps_unique(batch)

  def fwd_sample(self, batch, epsilon=0.0):
    """ Non-differentiable; sample a child or parent.
    
        See policy.py : sample for more info.

        Input: batch: List of [State], or State
        Output: List of [State], or State
    """
    return self.policy_fwd.sample(batch, epsilon=epsilon)

  def back_logps_unique(self, batch):
    """ Differentiable; output logps of unique children/parents. """
    return self.policy_back.logps_unique(batch)

  def back_sample(self, batch):
    """ Non-differentiable; sample a child or parent.

        Input: batch: List of [State], or State
        Output: List of [State], or State
    """
    return self.policy_back.sample(batch)

  """
    Exploration & modified policies
  """
  def batch_fwd_sample(self, n, epsilon=0.0, uniform=False):
    """ Batch samples dataset with n items.

        Parameters
        ----------
        n: int, size of dataset.
        epsilon: Chance in [0, 1] of uniformly sampling a unique child.
        uniform: If true, overrides epsilon to 1.0
        unique: bool, whether all samples should be unique

        Returns
        -------
        dataset: List of [Experience]
    """
    print('Sampling dataset ...')
    if uniform:
      print('Using uniform forward policy on unique children ...')
      epsilon = 1.0
    incomplete_trajs = [[self.mdp.root()] for _ in range(n)]
    complete_trajs = []
    while len(incomplete_trajs) > 0:
      inp = [t[-1] for t in incomplete_trajs]
      samples = self.fwd_sample(inp, epsilon=epsilon)
      for i, sample in enumerate(samples):
        incomplete_trajs[i].append(sample)
      
      # Remove complete trajs that hit leaf
      temp_incomplete = []
      for t in incomplete_trajs:
        if not t[-1].is_leaf:
          temp_incomplete.append(t)
        else:
          complete_trajs.append(t)
      incomplete_trajs = temp_incomplete

    # convert trajs to exps
    list_exps = []
    for traj in complete_trajs:
      x = traj[-1]
      r = self.mdp.reward(x)
      exp = Experience(traj=traj, x=x, r=r,
        logr=torch.log(torch.tensor(r, device=self.args.device)))
      list_exps.append(exp)
    return list_exps

  def batch_back_sample(self, xs):
    """ Batch samples trajectories backwards from xs.
        Batches over xs, iteratively sampling parents for each x in parallel.
        Effective batch size decreases when some trajectories hit root early.

        Input xs: List of [State], or State
        Return trajs: List of List[State], or List[State]
    """
    batched = bool(type(xs) is list)
    if not batched:
      xs = [xs]

    complete_trajs = []
    incomplete_trajs = [[x] for x in xs]
    while len(incomplete_trajs) > 0:
      inp = [t[0] for t in incomplete_trajs]
      samples = self.back_sample(inp)
      for i, sample in enumerate(samples):
        incomplete_trajs[i].insert(0, sample)
      
      # Remove complete trajectories that hit root
      temp_incomplete = []
      for t in incomplete_trajs:
        if t[0] != self.mdp.root():
          temp_incomplete.append(t)
        else:
          complete_trajs.append(t)
      incomplete_trajs = temp_incomplete

    return complete_trajs if batched else complete_trajs[0]

  """
    Trajectories
  """
  def traj_fwd_logp(self, exp):
    """ Computes logp(trajectory) under current model.
        Batches over states in trajectory. 
    """
    states_to_logps = self.fwd_logps_unique(exp.traj[:-1])
    total = 0
    for state_to_logp, child in zip(states_to_logps, exp.traj[1:]):
      try:
        total += state_to_logp[child]
      except ValueError:
        print(f'Hit ValueError. {child=}, {state_to_logp=}')
        import code; code.interact(local=dict(globals(), **locals()))
    return total

  def traj_back_logp(self, exp):
    """ Computes logp(trajectory) under current model.
        Batches over states in trajectory. 
    """
    states_to_logps = self.back_logps_unique(exp.traj[1:])
    total = 0
    for state_to_logp, parent in zip(states_to_logps, exp.traj[:-1]):
      total += state_to_logp[parent]
    return total

  def batch_traj_fwd_logp(self, batch):
    """ Computes logp(trajectory) under current model.
        Batches over all states in all trajectories in a batch.

        Batch: List of [trajectory]

        Returns: Tensor of batch_size, logp
    """
    trajs = [exp.traj for exp in batch]
    fwd_states, back_states, unroll_idxs = unroll_trajs(trajs)

    states_to_logps = self.fwd_logps_unique(fwd_states)
    fwd_logp_chosen = [s2lp[c] for s2lp, c in zip(states_to_logps, back_states)]
    
    fwd_chain = self.logZ.repeat(len(batch))
    fwd_chain = accumulate_by_traj(fwd_chain, fwd_logp_chosen, unroll_idxs)
    # fwd chain is [bsize]
    return fwd_chain

  def batch_traj_back_logp(self, batch):
    """ Computes logp(trajectory) under current model.
        Batches over all states in all trajectories in a batch.

        Batch: List of [trajectory]

        Returns: Tensor of batch_size, logp
    """
    trajs = [exp.traj for exp in batch]
    fwd_states, back_states, unroll_idxs = unroll_trajs(trajs)

    states_to_logps = self.back_logps_unique(back_states)
    back_logp_chosen = [s2lp[p] for s2lp, p in zip(states_to_logps, fwd_states)]
    
    back_chain = torch.stack([exp.logr for exp in batch])
    back_chain = accumulate_by_traj(back_chain, back_logp_chosen, unroll_idxs)
    # back_chain is [bsize]
    return back_chain

  """
    Learning
  """
  def batch_loss_trajectory_balance(self, batch):
    """ batch: List of [Experience].

        Calls fwd_logps_unique and back_logps_unique (gpu) in parallel on
        all states in all trajs in batch, then collates.
    """
    fwd_chain = self.batch_traj_fwd_logp(batch)
    back_chain = self.batch_traj_back_logp(batch)

    # obtain target: mix back_chain with logp_guide
    targets = []
    for i, exp in enumerate(batch):
      if exp.logp_guide is not None:
        w = self.args.target_mix_backpolicy_weight
        log_rx = exp.logr.clone().detach().requires_grad_(True)
        target = w * back_chain[i] + (1 - w) * (exp.logp_guide + log_rx)
      else:
        target = back_chain[i]
      targets.append(target)
    targets = torch.stack(targets)

    losses = torch.square(fwd_chain - targets)
    losses = torch.clamp(losses, max=10**2)
    mean_loss = torch.mean(losses)
    return mean_loss

  def train_tb(self, batch, log = True):
    """ Step on trajectory balance loss.

      Parameters
      ----------
      batch: List of [Experience]

      Batching is handled in trainers.py.
    """
    batch_loss = self.batch_loss_trajectory_balance(batch)

    for opt in self.optimizers:
      opt.zero_grad()
  
    batch_loss.backward()
    
    for param_set in self.clip_grad_norm_params:
      torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm, error_if_nonfinite=True)
    for opt in self.optimizers:
      opt.step()
    self.clamp_logZ()

    if log:
      batch_loss = tensor_to_np(batch_loss)
      print(f'TB training:', batch_loss)
      wandb.log({'Regular TB loss': batch_loss})
    return

  """ 
    IO & misc
  """
  def save_params(self, file):
    print('Saving checkpoint model ...')
    Path('/'.join(file.split('/')[:-1])).mkdir(parents=True, exist_ok=True)
    torch.save({
      'policy_fwd':   self.policy_fwd.state_dict(),
      'policy_back':  self.policy_back.state_dict(),
      'logZ':         self.logZ,
    }, file)
    return

  def load_for_eval_from_checkpoint(self, file):
    print(f'Loading checkpoint model ...')
    checkpoint = torch.load(file)
    self.policy_fwd.load_state_dict(checkpoint['policy_fwd'])
    self.policy_back.load_state_dict(checkpoint['policy_back'])
    self.logZ = checkpoint['logZ']
    for net in self.nets:
      net.eval()
    return

  def clip_policy_logits(self, scores):
    return torch.clip(scores, min=self.args.clip_policy_logit_min,
                              max=self.args.clip_policy_logit_max)


"""
  Trajectory/state rolling and accumulating
"""
def unroll_trajs(trajs):
  # Unroll trajectory into states: (num. trajs) -> (num. states)
  s1s, s2s = [], []
  traj_idx_to_batch_idxs = {}
  for traj_idx, traj in enumerate(trajs):
    start_idx = len(s1s)
    s1s += traj[:-1]
    s2s += traj[1:]
    end_idx = len(s1s)
    traj_idx_to_batch_idxs[traj_idx] = (start_idx, end_idx)
  return s1s, s2s, traj_idx_to_batch_idxs


def accumulate_by_traj(chain, batch_logp, traj_idx_to_batch_idxs):
  # Sum states by trajectory: (num. states) -> (num. trajs)
  for traj_idx, (start, end) in traj_idx_to_batch_idxs.items():
    chain[traj_idx] = chain[traj_idx] + sum(batch_logp[start:end])
  return chain


class SubstructureGFN(BaseTBGFlowNet):
  """ Substructure GFN. Learns with guided trajectory balance. """
  def __init__(self, args, mdp, actor):
    super().__init__(args, mdp, actor)
    print('Model: Substructure GFN')

  def train(self, batch):
    return self.train_substructure(batch)

  def train_substructure(self, batch, log = True):
    """ Guided trajectory balance for substructure GFN.
        1. Update back policy to approximate guide,
        2. Update forward policy to match back policy with TB.
        
        Batch: List of [Experience]

        Uses 1 pass for fwd and back net.
    """
    fwd_chain = self.batch_traj_fwd_logp(batch)
    back_chain = self.batch_traj_back_logp(batch)

    # 1. Obtain back policy loss
    logp_guide = torch.stack([exp.logp_guide for exp in batch])
    back_losses = torch.square(back_chain - logp_guide)
    back_losses = torch.clamp(back_losses, max=10**2)
    mean_back_loss = torch.mean(back_losses)

    # 2. Obtain TB loss with target: mix back_chain with logp_guide
    targets = []
    for i, exp in enumerate(batch):
      if exp.logp_guide is not None:
        w = self.args.target_mix_backpolicy_weight
        target = w * back_chain[i].detach() + (1 - w) * (exp.logp_guide + exp.logr)
      else:
        target = back_chain[i].detach()
      targets.append(target)
    targets = torch.stack(targets)

    tb_losses = torch.square(fwd_chain - targets)
    tb_losses = torch.clamp(tb_losses, max=10**2)
    loss_tb = torch.mean(tb_losses)

    # 1. Update back policy on back loss
    self.optimizer_back.zero_grad()
    loss_step1 = mean_back_loss
    loss_step1.backward()
    for param_set in self.clip_grad_norm_params:
      torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm, error_if_nonfinite=True)
    self.optimizer_back.step()
    if log:
      loss_step1 = tensor_to_np(loss_step1)
      print(f'Back training:', loss_step1)

    # 2. Update fwd policy on TB loss
    self.optimizer_fwdZ.zero_grad()
    loss_tb.backward()
    for param_set in self.clip_grad_norm_params:
      torch.nn.utils.clip_grad_norm_(param_set, self.args.clip_grad_norm, error_if_nonfinite=True)
    self.optimizer_fwdZ.step()
    self.clamp_logZ()
    if log:
      loss_tb = tensor_to_np(loss_tb)
      print(f'Fwd training:', loss_tb)

    if log:
      logZ = tensor_to_np(self.logZ)
      print(f'{logZ=}')
      wandb.log({
        'Sub back loss': loss_step1,
        'Sub fwdZ loss': loss_tb,
        'Sub logZ': logZ,
      })
    return
  
import numpy as np
from tqdm import tqdm
import torch
from collections import namedtuple, defaultdict

from ..data import Experience
from .. import utils


def uniform_policy(parent, children):
  return np.random.choice(children)


def unique_keep_order_remove_none(items):
  """ Remove duplicates, keeping order. Uses hashing. """
  return [x for x in list(dict.fromkeys(items)) if x is not None]


class BaseState():
  def __init__(self, content, is_leaf=False, is_root=False):
    self.content = self.canonicalize(content)
    self.is_leaf = is_leaf
    self.is_root = is_root

  def __repr__(self):
    """ Human-readable string representation of state. """
    raise NotImplementedError()

  def __eq__(self, other):
    return self.content_equals(other) and \
           self.is_leaf == other.is_leaf and \
           self.is_root == other.is_root

  def __hash__(self):
    """ Hash. Used for fast equality checking.
    
        Important for data structures that require unique states, e.g.,
        sampling unique datasets of x, or allXtoR dict that stores map
        {x: r} of all unique training data seen so far. 
        Importantly, policy needs to return logp over unique children
        or parents, which requires summing over equivalent states that
        can arise from different actions.
    """
    return hash(repr(self))

  def canonicalize(self, content):
    """ Standardize content. Ex: default sort for unordered items. """
    raise NotImplementedError()

  def content_equals(self, other):
    raise NotImplementedError()

  def is_member(self, other):
    """ Returns bool, whether there exists a path in the MDP from
        self state to other state. Critical for substructure guided
        GFlowNet.
    """
    raise NotImplementedError()


class BaseMDP:
  """ Markov Decision Process.
      
      Specifies relations between States, which States are leafs and their
      rewards. Also specifies forward and backward policy nets with
      high-dimensional outputs corresponding to actions, network
      architecture and logic for forward passing (translating actions into
      unique child/parent states).

      Inherited by object-specific MDPs (e.g., GraphMDP, BagMDP) in MDPs
      folder, and further inherited by task-specific scripts (e.g.,) in 
      experiments (exp) folder (e.g., TFBind8MDP) which specify reward
      functions, etc. 
  """
  def __init__(self):
    pass

  # Fundamentals
  def root(self):
    """ Return the root state. """
    raise NotImplementedError

  # Membership
  def is_member(self, query, target):
    """ Return bool, whether there exists an MDP path from query to target. """
    raise NotImplementedError

  """
    Children and parents
  """
  def get_children(self, state):
    """ Return list of children in deterministic order.
        Calls self.transition_fwd on actions from self.get_fwd_actions.
    """
    return [self.transition_fwd(state, act)
            for act in self.get_fwd_actions(state)]

  def get_parents(self, state):
    """ Return list of children in deterministic order.
        Calls self.transition_back on actions from self.get_back_actions.
    """
    return [self.transition_back(state, act)
            for act in self.get_back_actions(state)]

  def get_unique_children(self, state):
    """ Return unique states, keeping order. Used for substructure guide.
        Removes None
    """
    return unique_keep_order_remove_none(self.get_children(state))

  def get_unique_parents(self, state):
    """ Return unique states, keeping order.
        Used for maximum entropy gflownet with uniform backward policy.
    """
    return unique_keep_order_remove_none(self.get_parents(state))

  # Transitions
  def transition_fwd(self, state, action):
    """ Applies Action to state. Returns State or None (invalid transition). """
    raise NotImplementedError

  def transition_back(self, state, action):
    """ Applies Action to state. Returns State or None (invalid transition). """
    raise NotImplementedError

  # Actions
  def get_fwd_actions(self, state):
    """ Gets forward actions from state. Returns List of Actions.

        For many MDPs, this is independent of state. The num actions
        returned must match the policy's output dim. List of actions
        is used to associate policy output scores with states, so it
        must be in a consistent, deterministic order given state.
    """
    raise NotImplementedError

  def get_back_actions(self, state):
    """ See get_fwd_actions. """
    raise NotImplementedError

  # Specs
  def reward(self, x):
    """ Leaf State -> float """
    raise NotImplementedError

  def has_stop(self, state):
    """ State -> bool """
    raise NotImplementedError

  def has_forced_stop(self, state):
    raise NotImplementedError

  # Featurization
  def featurize(self, state):
    raise NotImplementedError

  def get_net_io_shapes(self):
    """ Specify input, output shapes for fwd/back policies.

        Note that MDPs that allow stopping at arbitrary points
        need an output dimension for the stop action.

        Returns
        -------
        dict with fields ['forward/backward']['in/out'],
            specifying the input and output shape of networks.
    """
    raise NotImplementedError

class SeqInsertMDP(BaseMDP):
  """ MDP for building a string by inserting chars.

      Action set is a deterministic function of state.

      Forward actions: [stop, insert A at 0, insert B at 0, ...]
      Reverse actions: [Unstop, del 0, del 1, ..., del N]

      This implementation uses a fixed-size action set, with string
      pushed as leftward as possible.
      Featurization: Denote max sequence length as N. Then we one-hot encode
      for |alphabet|*N features. 
      Inserting or deleting positions beyond current sequence length are 
      invalid actions.

      Cannot contain any CUDA elements: instance is passed
      to ray remote workers for substructure guidance, which need
      access to get_children & is_member.
  """
  def __init__(self, args, alphabet=list('0123'), forced_stop_len=8):
    self.args = args
    self.alphabet = alphabet
    self.alphabet_set = set(self.alphabet)
    self.forced_stop_len = forced_stop_len
    
    self.positions = list(range(self.forced_stop_len))
    ins_act = lambda char, position: SeqInsertAction(
        SeqInsertActionType.InsertChar, char=char, position=position
    )
    self.fwd_actions = [SeqInsertAction(SeqInsertActionType.Stop)] + \
                       [ins_act(char=c, position=p)
                        for p in self.positions
                        for c in self.alphabet]
    self.back_actions = [SeqInsertAction(SeqInsertActionType.UnStop)] + \
                        [SeqInsertAction(SeqInsertActionType.DelPos, position=p)
                         for p in self.positions]
    self.state = SeqInsertState
    self.parallelize_policy = True

  def root(self):
    return self.state('')

  @functools.cache
  def is_member(self, query, target):
    # Returns true if there is a path from query to target in the MDP
    return query.is_member(target)
  
  """
    Children, parents, and transition.
    Calls BaseMDP functions.
    Uses transition_fwd/back and get_fwd/back_actions.
  """
  @functools.cache
  def get_children(self, state):
    return BaseMDP.get_children(self, state)

  @functools.cache
  def get_parents(self, state):
    return BaseMDP.get_parents(self, state)

  @functools.cache
  def get_unique_children(self, state):
    return BaseMDP.get_unique_children(self, state)

  @functools.cache
  def get_unique_parents(self, state):
    return BaseMDP.get_unique_parents(self, state)

  def has_stop(self, state):
    return len(state) == self.forced_stop_len

  def has_forced_stop(self, state):
    return len(state) == self.forced_stop_len

  def transition_fwd(self, state, action):
    """ Applies SeqInsertAction to state.
        Returns State or None (invalid transition). 
    """
    if state.is_leaf:
      return None
    if self.has_forced_stop(state) and action.action != SeqInsertActionType.Stop:
        return None

    if action.action == SeqInsertActionType.Stop:
      if self.has_stop(state):
        return state._terminate()
      else:
        return None

    if action.action == SeqInsertActionType.InsertChar:
      return state._insert(action)

  def transition_back(self, state, action):
    """ Applies SeqInsertAction to state. Returns State or None (invalid transition). 
    """
    if state == self.root():
      return None
    if state.is_leaf and action.action != SeqInsertActionType.UnStop:
      return None

    if action.action == SeqInsertActionType.UnStop:
      if state.is_leaf:
        return state._unterminate()
      else:
        return None

    if action.action == SeqInsertActionType.DelPos:
      return state._del(action)

  """
    Actions
  """
  def get_fwd_actions(self, state):
    """ Gets forward actions from state. Returns List of Actions.

        For many MDPs, this is independent of state. The num actions
        returned must match the policy's output dim. List of actions
        is used to associate policy output scores with states, so it
        must be in a consistent, deterministic order given state.
    """
    return self.fwd_actions

  def get_back_actions(self, state):
    """ Gets backward actions from state. Returns List of Actions.

        For many MDPs, this is independent of state. The num actions
        returned must match the policy's output dim. List of actions
        is used to associate policy output scores with states, so it
        must be in a consistent, deterministic order given state.
    """
    return self.back_actions


"""
  Actor
"""
class SeqInsertActor(Actor):
  """ Holds SeqInsertMDP and GPU elements: featurize & policies. """
  def __init__(self, args, mdp):
    self.args = args
    self.mdp = mdp

    self.alphabet = mdp.alphabet
    self.forced_stop_len = mdp.forced_stop_len

    self.char_to_idx = {a: i for (i, a) in enumerate(self.alphabet)}
    self.onehotencoder = OneHotEncoder(sparse = False)
    self.onehotencoder.fit([[c] for c in self.alphabet])

    self.ft_dim = self.get_feature_dim()

    self.policy_fwd = super().make_policy(self.args.sa_or_ssr, 'forward')
    self.policy_back = super().make_policy(self.args.sa_or_ssr, 'backward')

  # Featurization
  @functools.cache
  def featurize(self, state):
    """ fixed dim repr of sequence
        [one hot encoding of variable-length string] + [0 padding]
    """
    if len(state.content) > 0:
      embed = np.concatenate(self.onehotencoder.transform(
          [[c] for c in state.content]
      ))
      num_rem = self.forced_stop_len - len(state.content)
      padding = np.zeros((1, num_rem*len(self.alphabet))).flatten()
      embed = np.concatenate([embed, padding])
    else:
      embed = np.zeros((1, self.forced_stop_len*len(self.alphabet))).flatten()
    return torch.tensor(embed, dtype=torch.float, device = self.args.device)

  def get_feature_dim(self):
    # return self.featurize(state).shape[-1]
    return len(self.alphabet) * self.forced_stop_len

  """
    Networks
  """
  def net_forward_sa(self):
    hid_dim = self.args.sa_hid_dim
    n_layers = self.args.sa_n_layers
    net = network.make_mlp(
        [self.ft_dim] + \
        [hid_dim] * n_layers + \
        [len(self.mdp.fwd_actions)]
    )
    return network.StateFeaturizeWrap(net, self.featurize)

  def net_backward_sa(self):
    hid_dim = self.args.sa_hid_dim
    n_layers = self.args.sa_n_layers
    net = network.make_mlp(
        [self.ft_dim] + \
        [hid_dim] * n_layers + \
        [len(self.mdp.back_actions)]
    )
    return network.StateFeaturizeWrap(net, self.featurize)
  
  def net_encoder_ssr(self):
    hid_dim = self.args.ssr_encoder_hid_dim
    n_layers = self.args.ssr_encoder_n_layers
    ssr_embed_dim = self.args.ssr_embed_dim
    net =  network.make_mlp(
      [self.ft_dim] + \
      [hid_dim] * n_layers + \
      [ssr_embed_dim]
    )
    return network.StateFeaturizeWrap(net, self.featurize)

  def net_scorer_ssr(self):
    """ [encoding1, encoding2] -> scalar """
    hid_dim = self.args.ssr_scorer_hid_dim
    n_layers = self.args.ssr_scorer_n_layers
    ssr_embed_dim = self.args.ssr_embed_dim
    return network.make_mlp(
        [2*ssr_embed_dim] + \
        [hid_dim] * n_layers + \
        [1]
    )

class Trainer:
  def __init__(self, args, model, mdp, actor, monitor):
    self.args = args
    self.model = model
    self.mdp = mdp
    self.actor = actor
    self.monitor = monitor

  def learn(self, *args, **kwargs):
    if self.args.model == 'sub':
      print(f'Learning with ray guide workers ...')
      self.learn_with_ray_workers(*args, **kwargs)
    else:
      print(f'Learning without guide workers ...')
      self.learn_default(*args, **kwargs)

  def handle_init_dataset(self, initial_XtoR):
    if initial_XtoR:
      print(f'Using initial dataset of size {len(initial_XtoR)}. \
              Skipping first online round ...')
      if self.args.init_logz:
        self.model.init_logz(np.log(sum(initial_XtoR.values())))
    else:
      print(f'No initial dataset used')
    return

  """
    Training
  """
  def learn_default(self, initial_XtoR=None):
    """ Main learning training loop.
        Each learning round:
          Each online batch:
            sample a new dataset using exploration policy.
          Each offline batch:
            resample batch from full historical dataset
        Monitor exploration - judge modes with monitor_explore callable.

        To learn on fixed dataset only: Set 0 online batches per round,
        and provide initial dataset.

        dataset = List of [Experience]
    """
    allXtoR = initial_XtoR if initial_XtoR else dict()
    self.handle_init_dataset(initial_XtoR)

    num_online = self.args.num_online_batches_per_round
    num_offline = self.args.num_offline_batches_per_round
    online_bsize = self.args.num_samples_per_online_batch
    offline_bsize = self.args.num_samples_per_offline_batch
    monitor_fast_every = self.args.monitor_fast_every
    monitor_num_samples = self.args.monitor_num_samples
    print(f'Starting active learning. \
            Each round: {num_online=}, {num_offline=}')

    for round_num in tqdm(range(self.args.num_active_learning_rounds)):
      print(f'Starting learning round {round_num+1} / {self.args.num_active_learning_rounds} ...')
      
      # Online training - skip first if initial dataset was provided
      if not initial_XtoR or round_num > 0:
        for _ in range(num_online):
          # Sample new dataset
          with torch.no_grad():
            explore_data = self.model.batch_fwd_sample(online_bsize,
                epsilon=self.args.explore_epsilon)

          # Save to full dataset
          for exp in explore_data:
            if exp.x not in allXtoR:
              allXtoR[exp.x] = exp.r          

          # Train on online dataset
          for step_num in range(self.args.num_steps_per_batch):
            self.model.train(explore_data)

      # Offline training
      for _ in range(num_offline):
        offline_xs = self.select_offline_xs(allXtoR, offline_bsize)
        offline_dataset = self.offline_PB_traj_sample(offline_xs, allXtoR)

        # Train
        for step_num in range(self.args.num_steps_per_batch):
          self.model.train(offline_dataset)

      if round_num % monitor_fast_every == 0 and round_num > 0:
        truepolicy_data = self.model.batch_fwd_sample(monitor_num_samples,
              epsilon=0)
        self.monitor.log_samples(round_num, truepolicy_data)

      self.monitor.maybe_eval_samplelog(self.model, round_num, allXtoR)

      if round_num % self.args.save_every_x_active_rounds == 0:
        if round_num > 0:
          self.model.save_params(self.args.saved_models_dir + \
                                 self.args.run_name + \
                                 f'_round_{round_num}.pth')

    print('Finished training.')
    self.model.save_params(self.args.saved_models_dir + \
                           self.args.run_name + '_final.pth')
    self.monitor.maybe_eval_samplelog(self.model, round_num, allXtoR)
    return

  """
    Learn with ray workers
  """
  def learn_with_ray_workers(self, initial_XtoR=None):
    """ Guided trajectory balance - ray workers compute guide trajectories.
    """
    # Ray init
    # ray.init(num_cpus = self.args.num_guide_workers)
    guidemanager = guide.RayManager(self.args, self.mdp)

    allXtoR = initial_XtoR if initial_XtoR else dict()
    guidemanager.update_allXtoR(allXtoR)
    self.handle_init_dataset(initial_XtoR)

    num_online = self.args.num_online_batches_per_round
    num_offline = self.args.num_offline_batches_per_round
    online_bsize = self.args.num_samples_per_online_batch
    offline_bsize = self.args.num_samples_per_offline_batch
    monitor_fast_every = self.args.monitor_fast_every
    monitor_num_samples = self.args.monitor_num_samples
    print(f'Starting active learning. \
            Each round: {num_online=}, {num_offline=}')

    for round_num in tqdm(range(self.args.num_active_learning_rounds)):
      print(f'Starting learning round {round_num+1}/{self.args.num_active_learning_rounds} ...')
      
      # 1. Sample online x with explore policy
      for _ in range(num_online):
        print(f'Sampling new x ...')
        with torch.no_grad():
          explore_data = self.model.batch_fwd_sample(online_bsize,
              epsilon=self.args.explore_epsilon)

        online_xs = [exp.x for exp in explore_data]
        for exp in explore_data:
          if exp.x not in allXtoR:
            allXtoR[exp.x] = exp.r
        guidemanager.update_allXtoR(allXtoR)

        # 2. Submit online jobs - get guide traj for x
        guidemanager.submit_online_jobs(online_xs)

      # 2a. Submit offline jobs
      for _ in range(num_offline):
        offline_xs = self.select_offline_xs(allXtoR, offline_bsize)

        # Submit offline jobs
        if self.args.offline_style == 'guide_resamples_traj':
          guidemanager.submit_online_jobs(offline_xs)
        if self.args.offline_style == 'guide_scores_back_policy_traj':
          print(f'Sampling offline trajectories with back policy ...')
          with torch.no_grad():
            offline_trajs = self.model.batch_back_sample(offline_xs)
          guidemanager.submit_offline_jobs(offline_trajs)

      # 4. Train if possible
      for _ in range(num_online + num_offline):
        batch = guidemanager.get_results(batch_size=online_bsize)
        if batch is not None:
          print(f'Training ...')
          for step_num in range(self.args.num_steps_per_batch):
            self.model.train(batch)

      # 5. End of active round - monitor and save
      if round_num % monitor_fast_every == 0 and round_num > 0:
        truepolicy_data = self.model.batch_fwd_sample(monitor_num_samples,
              epsilon=0)
        self.monitor.log_samples(round_num, truepolicy_data)

      # Save to full dataset & log to monitor
      self.monitor.maybe_eval_samplelog(self.model, round_num, allXtoR)

      if round_num and round_num % self.args.save_every_x_active_rounds == 0:
        self.model.save_params(self.args.saved_models_dir + \
                               self.args.run_name + f'_round_{round_num}.pth')

    print('Finished training.')
    self.model.save_params(self.args.saved_models_dir + \
                           self.args.run_name + '_final.pth')
    self.monitor.maybe_eval_samplelog(self.model, round_num, allXtoR)
    return

  """
    Offline training
  """
  def select_offline_xs(self, allXtoR, batch_size):
    select = self.args.get('offline_select', 'biased')
    if select == 'biased':
      return self.__biased_sample_xs(allXtoR, batch_size)
    elif select == 'random':
      return self.__random_sample_xs(allXtoR, batch_size)

  def __biased_sample_xs(self, allXtoR, batch_size):
    """ Select xs for offline training. Returns List of [State].
        Draws 50% from top 10% of rewards, and 50% from bottom 90%. 
    """
    if len(allXtoR) < 10:
      return []
    rewards = np.array(list(allXtoR.values()))
    threshold = np.percentile(rewards, 90)
    top_xs = [x for x, r in allXtoR.items() if r >= threshold]
    bottom_xs = [x for x, r in allXtoR.items() if r <= threshold]
    sampled_xs = random.choices(top_xs, k=batch_size // 2) + \
                 random.choices(bottom_xs, k=batch_size // 2)
    return sampled_xs

  def __random_sample_xs(self, allXtoR, batch_size):
    """ Select xs for offline training. Returns List of [State]. """
    return random.choices(list(allXtoR.keys()), k=batch_size)

  def offline_PB_traj_sample(self, offline_xs, allXtoR):
    """ Sample trajectories for x using P_B, for offline training with TB.
        Returns List of [Experience].
    """
    offline_rs = [allXtoR[x] for x in offline_xs]

    # Not subgfn: sample trajectories from backward policy
    print(f'Sampling trajectories from backward policy ...')
    with torch.no_grad():
      offline_trajs = self.model.batch_back_sample(offline_xs)

    offline_dataset = [
      Experience(traj=traj, x=x, r=r,
                  logr=torch.log(torch.tensor(r, device=self.args.device)))
      for traj, x, r in zip(offline_trajs, offline_xs, offline_rs)
    ]
    return offline_dataset