"""
Deep Q-Network (DQN) Load Balancer Module

This module implements a production-grade Deep Q-Network based load balancer
that uses neural networks to learn optimal process-to-processor assignments.

DQN Advantages over Q-Learning:
- Handles continuous state spaces without discretization
- Generalizes to unseen states through function approximation
- Scales better with increasing state dimensions
- More sample-efficient through experience replay and target networks

Key DQN Features:
1. Neural Network Function Approximation - Replaces Q-table with deep network
2. Experience Replay - Breaks temporal correlations in training data
3. Target Network - Stabilizes training with periodic weight syncs
4. Double DQN - Reduces Q-value overestimation
5. Prioritized Experience Replay - Focuses on important transitions
6. Dueling Architecture - Separates state value and advantage streams
7. Gradient Clipping - Prevents exploding gradients
8. Learning Rate Scheduling - Adaptive learning rate decay

OS Concepts Demonstrated:
- Intelligent scheduling with deep learning
- Adaptive optimization through neural networks
- Learning complex patterns in system behavior

Author: AI Enhancement
Date: December 2024
"""

import os
import math
import random
import pickle
import logging
import threading
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Deque, NamedTuple
from dataclasses import dataclass, field
from collections import deque, namedtuple
from enum import Enum
import json

import numpy as np

# PyTorch imports with fallback
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.optim.lr_scheduler import StepLR, ExponentialLR
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None
    optim = None

from config import (
    LoadBalancingAlgorithm,
    ProcessState,
    SimulationConfig,
    DEFAULT_SIMULATION_CONFIG
)
from process import Process
from processor import Processor, ProcessorManager
from load_balancer import LoadBalancer, MigrationRecord

# Configure module logger
logger = logging.getLogger(__name__)


# =============================================================================
# DQN CONFIGURATION
# =============================================================================

@dataclass
class DQNConfig:
    """
    Configuration for Deep Q-Network algorithm.
    
    Hyperparameters are tuned for the load balancing domain with
    production-grade defaults for stable training.
    """
    # Network architecture
    hidden_layers: Tuple[int, ...] = (256, 256, 128)  # Layer sizes
    activation: str = "relu"                           # relu, leaky_relu, elu
    use_dueling: bool = True                           # Dueling DQN architecture
    use_noisy_nets: bool = False                       # Noisy layers for exploration
    
    # Learning parameters
    learning_rate: float = 1e-4                        # Adam optimizer LR
    discount_factor: float = 0.99                      # γ: Importance of future rewards
    tau: float = 0.005                                 # Soft update coefficient
    
    # Exploration parameters (ε-greedy)
    epsilon_start: float = 1.0                         # Initial exploration rate
    epsilon_end: float = 0.01                          # Minimum exploration rate
    epsilon_decay_steps: int = 10000                   # Steps for linear decay
    
    # Experience replay
    replay_buffer_size: int = 100000                   # Maximum transitions to store
    batch_size: int = 64                               # Training batch size
    min_replay_size: int = 1000                        # Minimum buffer before training
    prioritized_replay: bool = True                    # Use prioritized experience replay
    priority_alpha: float = 0.6                        # Priority exponent
    priority_beta_start: float = 0.4                   # IS weight start
    priority_beta_frames: int = 100000                 # Frames to anneal beta
    
    # Target network
    target_update_frequency: int = 1000                # Hard update frequency (if tau=1)
    use_soft_update: bool = True                       # Use soft updates instead
    
    # Double DQN
    use_double_dqn: bool = True                        # Use Double DQN
    
    # Training control
    gradient_clip: float = 10.0                        # Max gradient norm
    update_frequency: int = 4                          # Train every N steps
    
    # Learning rate scheduling
    lr_scheduler: str = "exponential"                  # none, step, exponential
    lr_decay_rate: float = 0.9999                      # LR decay rate
    lr_step_size: int = 1000                           # Steps between LR decay
    lr_min: float = 1e-6                               # Minimum learning rate
    
    # Reward shaping
    reward_scaling: float = 0.1                        # Scale rewards for stability
    fairness_weight: float = 0.3                       # Weight for load balance
    throughput_weight: float = 0.7                     # Weight for throughput
    migration_penalty: float = -0.5                    # Penalty for migration
    
    # Model persistence
    model_save_path: str = "output/dqn_model.pth"
    auto_save_interval: int = 1000                     # Steps between auto-saves
    
    # Device
    device: str = "auto"                               # auto, cpu, cuda, mps
    
    def get_device(self) -> str:
        """Get the appropriate device for PyTorch."""
        if self.device == "auto":
            if TORCH_AVAILABLE:
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return "mps"
            return "cpu"
        return self.device
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'hidden_layers': self.hidden_layers,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'replay_buffer_size': self.replay_buffer_size,
            'batch_size': self.batch_size,
            'use_double_dqn': self.use_double_dqn,
            'use_dueling': self.use_dueling,
            'prioritized_replay': self.prioritized_replay,
            'fairness_weight': self.fairness_weight,
            'throughput_weight': self.throughput_weight
        }


# Default DQN configuration
DEFAULT_DQN_CONFIG = DQNConfig()


# =============================================================================
# EXPERIENCE REPLAY
# =============================================================================

class Transition(NamedTuple):
    """Single experience transition."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class SumTree:
    """
    Sum tree data structure for efficient prioritized sampling.
    
    This tree-based structure allows O(log n) updates and sampling
    proportional to priorities. Essential for prioritized experience replay.
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.write_idx = 0
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """Retrieve leaf index for a given value s."""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    @property
    def total(self) -> float:
        """Total priority sum."""
        return self.tree[0]
    
    def add(self, priority: float, data: Transition):
        """Add experience with given priority."""
        idx = self.write_idx + self.capacity - 1
        
        self.data[self.write_idx] = data
        self.update(idx, priority)
        
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
    
    def update(self, idx: int, priority: float):
        """Update priority of an existing node."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, Transition]:
        """Get experience for a given cumulative priority."""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer.
    
    Samples experiences proportional to their TD-error, focusing
    learning on surprising or informative transitions.
    
    Uses importance sampling weights to correct the bias introduced
    by non-uniform sampling.
    """
    
    def __init__(self, capacity: int, config: DQNConfig):
        self.capacity = capacity
        self.config = config
        self.tree = SumTree(capacity)
        self.beta = config.priority_beta_start
        self.beta_increment = (1.0 - config.priority_beta_start) / config.priority_beta_frames
        self.max_priority = 1.0
        self.min_priority = 1e-6
        self._lock = threading.Lock()
    
    def add(self, transition: Transition):
        """Add transition with maximum priority."""
        with self._lock:
            priority = self.max_priority ** self.config.priority_alpha
            self.tree.add(priority, transition)
    
    def sample(self, batch_size: int) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions.
        
        Returns:
            transitions: List of sampled transitions
            indices: Tree indices for priority updates
            weights: Importance sampling weights
        """
        with self._lock:
            transitions = []
            indices = np.zeros(batch_size, dtype=np.int32)
            weights = np.zeros(batch_size, dtype=np.float32)
            
            # Segment the priority range
            segment = self.tree.total / batch_size
            
            # Increase beta over time
            self.beta = min(1.0, self.beta + self.beta_increment)
            
            # Calculate max weight for normalization
            min_prob = self.min_priority / self.tree.total if self.tree.total > 0 else 1e-6
            max_weight = (min_prob * self.tree.n_entries) ** (-self.beta)
            
            for i in range(batch_size):
                # Sample uniformly from each segment
                low = segment * i
                high = segment * (i + 1)
                s = random.uniform(low, high)
                
                idx, priority, data = self.tree.get(s)
                
                if data is None or not isinstance(data, Transition):
                    # Fallback: use a random valid sample
                    s = random.uniform(0, self.tree.total)
                    idx, priority, data = self.tree.get(s)
                
                indices[i] = idx
                transitions.append(data)
                
                # Calculate importance sampling weight
                prob = priority / self.tree.total if self.tree.total > 0 else 1e-6
                weight = (prob * self.tree.n_entries) ** (-self.beta)
                weights[i] = weight / max_weight
            
            return transitions, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        with self._lock:
            for idx, td_error in zip(indices, td_errors):
                priority = (abs(td_error) + self.min_priority) ** self.config.priority_alpha
                self.tree.update(idx, priority)
                self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        return self.tree.n_entries


class UniformReplayBuffer:
    """Standard uniform experience replay buffer."""
    
    def __init__(self, capacity: int):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)
        self._lock = threading.Lock()
    
    def add(self, transition: Transition):
        """Add a transition to the buffer."""
        with self._lock:
            self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> Tuple[List[Transition], None, None]:
        """Sample a batch of transitions uniformly."""
        with self._lock:
            transitions = random.sample(list(self.buffer), 
                                        min(batch_size, len(self.buffer)))
            return transitions, None, None
    
    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# NEURAL NETWORK ARCHITECTURES
# =============================================================================

if TORCH_AVAILABLE:
    
    class NoisyLinear(nn.Module):
        """
        Noisy linear layer for exploration.
        
        Replaces epsilon-greedy with learned exploration through
        parameter noise. Enables state-dependent exploration.
        """
        
        def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.std_init = std_init
            
            # Learnable parameters
            self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
            self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_sigma = nn.Parameter(torch.empty(out_features))
            
            # Factorized noise
            self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
            self.register_buffer('bias_epsilon', torch.empty(out_features))
            
            self.reset_parameters()
            self.reset_noise()
        
        def reset_parameters(self):
            """Initialize parameters."""
            mu_range = 1 / math.sqrt(self.in_features)
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))
        
        def reset_noise(self):
            """Sample new noise."""
            epsilon_in = self._scale_noise(self.in_features)
            epsilon_out = self._scale_noise(self.out_features)
            self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
            self.bias_epsilon.copy_(epsilon_out)
        
        def _scale_noise(self, size: int) -> torch.Tensor:
            """Generate factorized Gaussian noise."""
            x = torch.randn(size, device=self.weight_mu.device)
            return x.sign() * x.abs().sqrt()
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.training:
                weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
                bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
            else:
                weight = self.weight_mu
                bias = self.bias_mu
            return F.linear(x, weight, bias)
    
    
    class DQNetwork(nn.Module):
        """
        Deep Q-Network with configurable architecture.
        
        Supports:
        - Standard feedforward architecture
        - Dueling architecture (separate value and advantage streams)
        - Noisy layers for exploration
        """
        
        def __init__(self, state_size: int, action_size: int, config: DQNConfig):
            super().__init__()
            self.state_size = state_size
            self.action_size = action_size
            self.config = config
            self.use_dueling = config.use_dueling
            self.use_noisy = config.use_noisy_nets
            
            # Get activation function
            if config.activation == "relu":
                self.activation = F.relu
            elif config.activation == "leaky_relu":
                self.activation = F.leaky_relu
            elif config.activation == "elu":
                self.activation = F.elu
            else:
                self.activation = F.relu
            
            # Build network layers
            Linear = NoisyLinear if self.use_noisy else nn.Linear
            
            # Feature extraction layers
            layers = []
            prev_size = state_size
            for hidden_size in config.hidden_layers[:-1]:
                layers.append(Linear(prev_size, hidden_size))
                layers.append(nn.LayerNorm(hidden_size))
                prev_size = hidden_size
            self.feature_layers = nn.ModuleList(layers)
            
            # Final feature size
            final_hidden = config.hidden_layers[-1] if config.hidden_layers else 128
            self.feature_output = Linear(prev_size, final_hidden)
            self.feature_norm = nn.LayerNorm(final_hidden)
            
            if self.use_dueling:
                # Value stream
                self.value_hidden = Linear(final_hidden, final_hidden // 2)
                self.value_output = Linear(final_hidden // 2, 1)
                
                # Advantage stream
                self.advantage_hidden = Linear(final_hidden, final_hidden // 2)
                self.advantage_output = Linear(final_hidden // 2, action_size)
            else:
                # Standard Q-output
                self.q_output = Linear(final_hidden, action_size)
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass returning Q-values for all actions."""
            # Feature extraction
            for i in range(0, len(self.feature_layers), 2):
                x = self.feature_layers[i](x)  # Linear
                x = self.feature_layers[i + 1](x)  # LayerNorm
                x = self.activation(x)
            
            x = self.feature_output(x)
            x = self.feature_norm(x)
            x = self.activation(x)
            
            if self.use_dueling:
                # Compute value and advantage separately
                value = self.activation(self.value_hidden(x))
                value = self.value_output(value)
                
                advantage = self.activation(self.advantage_hidden(x))
                advantage = self.advantage_output(advantage)
                
                # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,:)))
                q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
            else:
                q_values = self.q_output(x)
            
            return q_values
        
        def reset_noise(self):
            """Reset noise in noisy layers."""
            if self.use_noisy:
                for module in self.modules():
                    if isinstance(module, NoisyLinear):
                        module.reset_noise()


# =============================================================================
# DQN AGENT
# =============================================================================

class DQNAgent:
    """
    Production-grade Deep Q-Network Agent.
    
    This agent learns optimal load balancing through deep reinforcement
    learning, using neural networks to approximate Q-values.
    
    Key Features:
    - Dueling network architecture for better value estimation
    - Double DQN to reduce overestimation
    - Prioritized experience replay for efficient learning
    - Target network for training stability
    - Gradient clipping for robustness
    - Comprehensive training metrics
    
    Modes:
    1. Training: Active exploration and learning
    2. Evaluation: Pure exploitation of learned policy
    """
    
    def __init__(self, state_size: int, action_size: int, 
                 config: DQNConfig = None):
        """
        Initialize the DQN agent.
        
        Args:
            state_size: Dimension of state vector
            action_size: Number of possible actions
            config: DQN hyperparameters
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for DQN. Install with: pip install torch"
            )
        
        self.state_size = state_size
        self.action_size = action_size
        self.config = config or DEFAULT_DQN_CONFIG
        
        # Set device
        device_name = self.config.get_device()
        self.device = torch.device(device_name)
        logger.info(f"DQN using device: {self.device}")
        
        # Create networks
        self.policy_net = DQNetwork(state_size, action_size, self.config).to(self.device)
        self.target_net = DQNetwork(state_size, action_size, self.config).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer with weight decay for regularization
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        if self.config.lr_scheduler == "step":
            self.scheduler = StepLR(
                self.optimizer,
                step_size=self.config.lr_step_size,
                gamma=self.config.lr_decay_rate
            )
        elif self.config.lr_scheduler == "exponential":
            self.scheduler = ExponentialLR(
                self.optimizer,
                gamma=self.config.lr_decay_rate
            )
        else:
            self.scheduler = None
        
        # Experience replay buffer
        if self.config.prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                self.config.replay_buffer_size, self.config
            )
        else:
            self.replay_buffer = UniformReplayBuffer(self.config.replay_buffer_size)
        
        # Exploration rate
        self.epsilon = self.config.epsilon_start
        self.epsilon_decay = (self.config.epsilon_start - self.config.epsilon_end) / \
                             self.config.epsilon_decay_steps
        
        # Training state
        self.training_mode = True
        self.total_steps = 0
        self.episode_count = 0
        self.update_count = 0
        
        # Metrics tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.loss_history: List[float] = []
        self.q_value_history: List[float] = []
        self.epsilon_history: List[float] = []
        
        # Current episode state
        self._current_episode_reward = 0.0
        self._current_episode_length = 0
        self._last_state: Optional[np.ndarray] = None
        self._last_action: Optional[int] = None
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"DQN agent initialized: state_size={state_size}, "
                   f"action_size={action_size}")
    
    def encode_state(self, processors: List[Processor], process: Process) -> np.ndarray:
        """
        Encode system state into a continuous vector.
        
        Unlike Q-learning, DQN uses continuous state representation,
        preserving full information without discretization.
        
        State vector components:
        - Normalized processor loads (0-1)
        - Normalized queue sizes
        - Process characteristics (burst, priority, memory)
        - System-level statistics (mean, std, max load)
        
        Args:
            processors: List of processors
            process: Incoming process
            
        Returns:
            State vector as numpy array
        """
        num_processors = len(processors)
        
        # Processor-level features
        loads = np.array([p.get_load() for p in processors], dtype=np.float32)
        queues = np.array([p.get_queue_size() for p in processors], dtype=np.float32)
        
        # Normalize loads
        max_load = max(loads.max(), 1.0)
        norm_loads = loads / max_load
        
        # Normalize queue sizes
        max_queue = max(queues.max(), 1.0)
        norm_queues = queues / max_queue
        
        # System-level features
        system_features = np.array([
            norm_loads.mean(),           # Average load
            norm_loads.std(),            # Load variance
            norm_loads.max(),            # Max load
            norm_loads.min(),            # Min load
            norm_queues.mean(),          # Average queue
            norm_queues.max(),           # Max queue
        ], dtype=np.float32)
        
        # Process features
        process_features = np.array([
            process.burst_time / 20.0,   # Normalized burst time
            process.priority.value / 3.0, # Normalized priority
            process.memory_required / 512.0 if hasattr(process, 'memory_required') else 0.5,
        ], dtype=np.float32)
        
        # Combine all features
        state = np.concatenate([
            norm_loads,          # Per-processor loads
            norm_queues,         # Per-processor queues
            system_features,     # System statistics
            process_features,    # Process info
        ])
        
        return state
    
    def get_state_size(self, num_processors: int) -> int:
        """Calculate state vector size for given number of processors."""
        return num_processors * 2 + 6 + 3  # loads + queues + system + process
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state vector
            
        Returns:
            Selected action (processor index)
        """
        with self._lock:
            self.total_steps += 1
            self._current_episode_length += 1
            
            # Epsilon-greedy exploration
            if self.training_mode and random.random() < self.epsilon:
                action = random.randrange(self.action_size)
                logger.debug(f"Exploring: random action {action}")
            else:
                # Exploitation: use network
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    self.policy_net.eval()
                    q_values = self.policy_net(state_tensor)
                    self.policy_net.train()
                    action = q_values.argmax(dim=1).item()
                
                logger.debug(f"Exploiting: action {action}")
            
            # Store for later
            self._last_state = state
            self._last_action = action
            
            # Decay epsilon
            if self.training_mode:
                self.epsilon = max(
                    self.config.epsilon_end,
                    self.epsilon - self.epsilon_decay
                )
            
            return action
    
    def step(self, reward: float, next_state: np.ndarray, done: bool):
        """
        Process transition and update network.
        
        Args:
            reward: Reward received
            next_state: Resulting state
            done: Episode termination flag
        """
        if self._last_state is None or self._last_action is None:
            return
        
        with self._lock:
            # Scale reward for stability
            scaled_reward = reward * self.config.reward_scaling
            
            # Store transition
            transition = Transition(
                state=self._last_state,
                action=self._last_action,
                reward=scaled_reward,
                next_state=next_state,
                done=done
            )
            self.replay_buffer.add(transition)
            
            # Track episode reward
            self._current_episode_reward += reward
            
            # Train network
            if (len(self.replay_buffer) >= self.config.min_replay_size and
                self.total_steps % self.config.update_frequency == 0 and
                self.training_mode):
                loss = self._train_step()
                if loss is not None:
                    self.loss_history.append(loss)
            
            # Update target network
            if self.total_steps % self.config.target_update_frequency == 0:
                self._update_target_network()
            
            # Handle episode end
            if done:
                self._end_episode()
    
    def _train_step(self) -> Optional[float]:
        """
        Perform single training step on batch.
        
        Uses Double DQN for action selection and Huber loss
        for robust gradient computation.
        
        Returns:
            Training loss value
        """
        # Sample batch
        transitions, indices, weights = self.replay_buffer.sample(self.config.batch_size)
        
        if not transitions or len(transitions) < self.config.batch_size:
            return None
        
        # Prepare batch tensors
        states = torch.FloatTensor(np.array([t.state for t in transitions])).to(self.device)
        actions = torch.LongTensor([t.action for t in transitions]).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in transitions]).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in transitions])).to(self.device)
        dones = torch.FloatTensor([t.done for t in transitions]).to(self.device)
        
        # Importance sampling weights for prioritized replay
        if weights is not None:
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            weights = torch.ones(self.config.batch_size).to(self.device)
        
        # Current Q-values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            if self.config.use_double_dqn:
                # Double DQN: use policy net to select actions, target net to evaluate
                next_actions = self.policy_net(next_states).argmax(dim=1)
                next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN
                next_q = self.target_net(next_states).max(dim=1)[0]
            
            target_q = rewards + (1 - dones) * self.config.discount_factor * next_q
        
        # Compute TD error for priority updates
        td_errors = (target_q - current_q).detach().cpu().numpy()
        
        # Weighted Huber loss
        loss = F.smooth_l1_loss(current_q, target_q, reduction='none')
        weighted_loss = (loss * weights).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), 
            self.config.gradient_clip
        )
        
        self.optimizer.step()
        self.update_count += 1
        
        # Update learning rate
        if self.scheduler and self.optimizer.param_groups[0]['lr'] > self.config.lr_min:
            self.scheduler.step()
        
        # Update priorities for prioritized replay
        if indices is not None:
            self.replay_buffer.update_priorities(indices, td_errors)
        
        # Track average Q-value
        avg_q = current_q.mean().item()
        self.q_value_history.append(avg_q)
        
        return weighted_loss.item()
    
    def _update_target_network(self):
        """Update target network (soft or hard)."""
        if self.config.use_soft_update:
            # Soft update: θ_target = τ*θ_policy + (1-τ)*θ_target
            for target_param, policy_param in zip(
                self.target_net.parameters(), 
                self.policy_net.parameters()
            ):
                target_param.data.copy_(
                    self.config.tau * policy_param.data + 
                    (1 - self.config.tau) * target_param.data
                )
        else:
            # Hard update
            self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def _end_episode(self):
        """Handle episode completion."""
        # Record statistics
        self.episode_rewards.append(self._current_episode_reward)
        self.episode_lengths.append(self._current_episode_length)
        self.epsilon_history.append(self.epsilon)
        
        self.episode_count += 1
        
        # Reset noise in noisy nets
        if self.config.use_noisy_nets:
            self.policy_net.reset_noise()
            self.target_net.reset_noise()
        
        # Auto-save
        if self.episode_count % (self.config.auto_save_interval // 10) == 0:
            self.save_model()
        
        logger.info(
            f"Episode {self.episode_count} | "
            f"Reward: {self._current_episode_reward:.2f} | "
            f"Length: {self._current_episode_length} | "
            f"ε: {self.epsilon:.4f} | "
            f"Loss: {self.loss_history[-1]:.4f}" if self.loss_history else ""
        )
        
        # Reset episode tracking
        self._current_episode_reward = 0.0
        self._current_episode_length = 0
    
    def set_training_mode(self, training: bool):
        """Set training or evaluation mode."""
        self.training_mode = training
        if training:
            self.policy_net.train()
        else:
            self.policy_net.eval()
            self.epsilon = self.config.epsilon_end  # Minimal exploration
        logger.info(f"DQN mode: {'Training' if training else 'Evaluation'}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive training statistics."""
        with self._lock:
            return {
                'episode_count': self.episode_count,
                'total_steps': self.total_steps,
                'update_count': self.update_count,
                'epsilon': self.epsilon,
                'training_mode': self.training_mode,
                'replay_buffer_size': len(self.replay_buffer),
                'avg_episode_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
                'avg_episode_length': np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0,
                'avg_loss': np.mean(self.loss_history[-100:]) if self.loss_history else 0,
                'avg_q_value': np.mean(self.q_value_history[-100:]) if self.q_value_history else 0,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'device': str(self.device),
            }
    
    def save_model(self, path: str = None):
        """Save model checkpoint."""
        save_path = path or self.config.model_save_path
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', 
                    exist_ok=True)
        
        with self._lock:
            checkpoint = {
                'policy_net_state': self.policy_net.state_dict(),
                'target_net_state': self.target_net.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'total_steps': self.total_steps,
                'episode_count': self.episode_count,
                'update_count': self.update_count,
                'episode_rewards': self.episode_rewards[-1000:],
                'loss_history': self.loss_history[-1000:],
                'q_value_history': self.q_value_history[-1000:],
                'config': self.config.to_dict(),
                'state_size': self.state_size,
                'action_size': self.action_size,
            }
            
            try:
                torch.save(checkpoint, save_path)
                logger.info(f"DQN model saved to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save DQN model: {e}")
    
    def load_model(self, path: str = None) -> bool:
        """Load model checkpoint."""
        load_path = path or self.config.model_save_path
        
        if not os.path.exists(load_path):
            logger.warning(f"No DQN model found at {load_path}")
            return False
        
        try:
            checkpoint = torch.load(load_path, map_location=self.device, weights_only=False)
            
            with self._lock:
                self.policy_net.load_state_dict(checkpoint['policy_net_state'])
                self.target_net.load_state_dict(checkpoint['target_net_state'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
                self.epsilon = checkpoint.get('epsilon', self.config.epsilon_end)
                self.total_steps = checkpoint.get('total_steps', 0)
                self.episode_count = checkpoint.get('episode_count', 0)
                self.update_count = checkpoint.get('update_count', 0)
                self.episode_rewards = checkpoint.get('episode_rewards', [])
                self.loss_history = checkpoint.get('loss_history', [])
                self.q_value_history = checkpoint.get('q_value_history', [])
            
            logger.info(f"DQN model loaded from {load_path}. "
                       f"Episodes: {self.episode_count}, Steps: {self.total_steps}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load DQN model: {e}")
            return False
    
    def reset(self):
        """Reset agent for new training session."""
        with self._lock:
            # Reinitialize networks
            self.policy_net = DQNetwork(
                self.state_size, self.action_size, self.config
            ).to(self.device)
            self.target_net = DQNetwork(
                self.state_size, self.action_size, self.config
            ).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
            # Reset optimizer
            self.optimizer = optim.Adam(
                self.policy_net.parameters(),
                lr=self.config.learning_rate,
                weight_decay=1e-5
            )
            
            # Reset replay buffer
            if self.config.prioritized_replay:
                self.replay_buffer = PrioritizedReplayBuffer(
                    self.config.replay_buffer_size, self.config
                )
            else:
                self.replay_buffer = UniformReplayBuffer(self.config.replay_buffer_size)
            
            # Reset state
            self.epsilon = self.config.epsilon_start
            self.total_steps = 0
            self.episode_count = 0
            self.update_count = 0
            self.episode_rewards.clear()
            self.episode_lengths.clear()
            self.loss_history.clear()
            self.q_value_history.clear()
            self.epsilon_history.clear()
            self._current_episode_reward = 0.0
            self._current_episode_length = 0
            self._last_state = None
            self._last_action = None
        
        logger.info("DQN agent reset complete")


# =============================================================================
# DQN LOAD BALANCER
# =============================================================================

class DQNBalancer(LoadBalancer):
    """
    Deep Q-Network based Load Balancer.
    
    Uses neural network function approximation to learn optimal
    process-to-processor assignments through experience.
    
    Advantages over Q-Learning:
    - Handles continuous state spaces naturally
    - Better generalization to unseen states
    - Scales with increasing state dimensions
    - More sample-efficient with deep learning techniques
    
    Modes:
    - Training: Active learning with exploration
    - Evaluation: Uses learned policy for optimal performance
    """
    
    def __init__(self, config: SimulationConfig = None,
                 dqn_config: DQNConfig = None,
                 num_processors: int = 4):
        """
        Initialize DQN load balancer.
        
        Args:
            config: Simulation configuration
            dqn_config: DQN hyperparameters
            num_processors: Number of processors
        """
        super().__init__(config)
        
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for DQN. Install with: pip install torch"
            )
        
        self.dqn_config = dqn_config or DEFAULT_DQN_CONFIG
        self.num_processors = num_processors
        
        # Calculate state size
        self.state_size = self._calculate_state_size(num_processors)
        
        # Initialize DQN agent
        self.agent = DQNAgent(
            state_size=self.state_size,
            action_size=num_processors,
            config=self.dqn_config
        )
        
        # Track pending assignments for reward
        self._pending_assignments: Dict[int, Tuple[np.ndarray, int, float]] = {}
        
        # Performance metrics
        self.decisions_made = 0
        
        logger.info(f"DQN Balancer initialized: {num_processors} processors, "
                   f"state_size={self.state_size}")
    
    def _calculate_state_size(self, num_processors: int) -> int:
        """Calculate state vector dimension."""
        return self.agent.get_state_size(num_processors) if hasattr(self, 'agent') else \
               num_processors * 2 + 6 + 3
    
    @property
    def algorithm_type(self) -> LoadBalancingAlgorithm:
        return LoadBalancingAlgorithm.DQN
    
    @property
    def name(self) -> str:
        return "AI (DQN)"
    
    def assign_process(self, process: Process, 
                       processors: List[Processor]) -> Optional[Processor]:
        """
        Assign process using DQN policy.
        
        Args:
            process: Process to assign
            processors: Available processors
            
        Returns:
            Selected processor
        """
        if not processors:
            return None
        
        # Handle processor count change
        if len(processors) != self.num_processors:
            self._resize_agent(len(processors))
        
        # Encode state
        state = self.agent.encode_state(processors, process)
        
        # Get action from agent
        action = self.agent.select_action(state)
        
        # Clamp to valid range
        action = min(action, len(processors) - 1)
        
        # Select processor
        selected = processors[action]
        
        # Store for reward calculation
        self._pending_assignments[process.pid] = (state, action, time.time())
        
        # Add process to processor
        selected.add_process(process)
        process.processor_id = selected.processor_id
        self.assignment_count += 1
        self.decisions_made += 1
        
        logger.debug(f"DQN: Assigned P{process.pid} to Processor {selected.processor_id}")
        
        return selected
    
    def process_completed(self, process: Process, processors: List[Processor]):
        """
        Handle process completion - provide reward feedback.
        
        Args:
            process: Completed process
            processors: Current processor states
        """
        if process.pid not in self._pending_assignments:
            return
        
        state, action, assignment_time = self._pending_assignments.pop(process.pid)
        
        # Calculate reward
        reward = self._calculate_reward(process, processors)
        
        # Encode next state
        dummy_process = Process(pid=-1, burst_time=5)
        next_state = self.agent.encode_state(processors, dummy_process)
        
        # Check if episode is done
        all_done = all(
            p.get_queue_size() == 0 and p.current_process is None 
            for p in processors
        )
        
        # Update agent
        self.agent.step(reward, next_state, done=all_done)
        
        logger.debug(f"DQN reward for P{process.pid}: {reward:.3f}")
    
    def _calculate_reward(self, process: Process, 
                          processors: List[Processor]) -> float:
        """
        Calculate reward for process completion.
        
        Reward components:
        1. Turnaround time (negative - minimize)
        2. Load balance fairness bonus
        3. Migration penalty
        """
        config = self.dqn_config
        
        # Turnaround component
        turnaround = process.get_turnaround_time()
        normalized_turnaround = -turnaround / max(process.burst_time, 1)
        throughput_reward = normalized_turnaround * config.throughput_weight
        
        # Fairness component
        loads = [p.get_load() for p in processors]
        max_load = max(loads) if loads else 1
        if max_load > 0:
            load_variance = np.var([l / max_load for l in loads])
            fairness_reward = (1.0 - min(load_variance, 1.0)) * config.fairness_weight
        else:
            fairness_reward = config.fairness_weight
        
        # Migration penalty
        migration_reward = process.migration_count * config.migration_penalty
        
        return throughput_reward + fairness_reward + migration_reward
    
    def _resize_agent(self, new_size: int):
        """Resize agent for different processor count."""
        logger.info(f"Resizing DQN agent: {self.num_processors} -> {new_size}")
        self.num_processors = new_size
        self.state_size = new_size * 2 + 6 + 3
        
        # Create new agent with updated size
        self.agent = DQNAgent(
            state_size=self.state_size,
            action_size=new_size,
            config=self.dqn_config
        )
        
        # Try to load existing model
        self.agent.load_model()
    
    def check_for_migration(self, processors: List[Processor], 
                            current_time: int) -> List[MigrationRecord]:
        """
        Check for migration opportunities using DQN insights.
        
        Uses the learned Q-values to identify suboptimal placements
        and suggest migrations.
        """
        migrations = []
        
        if len(processors) < 2:
            return migrations
        
        # Get current loads
        loads = [(p, p.get_load()) for p in processors]
        max_load = max(l for _, l in loads) if loads else 0
        
        if max_load <= 0:
            return migrations
        
        # Sort by load (highest first)
        loads.sort(key=lambda x: x[1], reverse=True)
        
        # Check heavily loaded processors
        for source_proc, source_load in loads[:len(loads)//2]:
            candidates = self.get_migration_candidates(source_proc)
            if not candidates:
                continue
            
            for process in candidates[:1]:  # Limit migrations
                # Evaluate current vs. optimal placement
                state = self.agent.encode_state(processors, process)
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                
                with torch.no_grad():
                    q_values = self.agent.policy_net(state_tensor).squeeze(0)
                
                best_action = q_values.argmax().item()
                current_action = source_proc.processor_id
                
                if best_action != current_action:
                    q_improvement = (q_values[best_action] - q_values[current_action]).item()
                    
                    if q_improvement > 0.1:  # Threshold
                        dest_proc = processors[best_action]
                        migration = MigrationRecord(
                            process_id=process.pid,
                            source_processor=source_proc.processor_id,
                            destination_processor=dest_proc.processor_id,
                            time=current_time,
                            reason=f"DQN: ΔQ={q_improvement:.3f}",
                            source_load_before=source_load,
                            destination_load_before=dest_proc.get_load()
                        )
                        migrations.append(migration)
                        break
        
        return migrations
    
    def set_training_mode(self, training: bool):
        """Set training/evaluation mode."""
        self.agent.set_training_mode(training)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get combined statistics."""
        agent_stats = self.agent.get_statistics()
        return {
            **agent_stats,
            'decisions_made': self.decisions_made,
            'assignment_count': self.assignment_count,
            'migration_count': self.migration_count,
            'pending_assignments': len(self._pending_assignments),
        }
    
    def save_model(self, path: str = None):
        """Save trained model."""
        self.agent.save_model(path)
    
    def load_model(self, path: str = None) -> bool:
        """Load trained model."""
        return self.agent.load_model(path)
    
    def reset(self):
        """Reset balancer state."""
        super().reset()
        self._pending_assignments.clear()
        self.decisions_made = 0
    
    def full_reset(self):
        """Full reset including agent."""
        self.reset()
        self.agent.reset()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def check_pytorch_available() -> bool:
    """Check if PyTorch is available."""
    return TORCH_AVAILABLE


def get_device_info() -> Dict[str, Any]:
    """Get information about available compute devices."""
    info = {
        'pytorch_available': TORCH_AVAILABLE,
        'cuda_available': False,
        'mps_available': False,
        'device': 'cpu'
    }
    
    if TORCH_AVAILABLE:
        info['cuda_available'] = torch.cuda.is_available()
        info['mps_available'] = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        
        if info['cuda_available']:
            info['device'] = 'cuda'
            info['cuda_device_name'] = torch.cuda.get_device_name(0)
        elif info['mps_available']:
            info['device'] = 'mps'
        else:
            info['device'] = 'cpu'
    
    return info


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'DQNConfig',
    'DEFAULT_DQN_CONFIG',
    'Transition',
    'SumTree',
    'PrioritizedReplayBuffer',
    'UniformReplayBuffer',
    'DQNAgent',
    'DQNBalancer',
    'check_pytorch_available',
    'get_device_info',
    'TORCH_AVAILABLE',
]


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("DQN Load Balancer Module Test")
    print("=" * 70)
    
    # Check PyTorch
    print(f"\nPyTorch available: {TORCH_AVAILABLE}")
    if not TORCH_AVAILABLE:
        print("Install PyTorch to use DQN: pip install torch")
        exit(1)
    
    device_info = get_device_info()
    print(f"Device: {device_info['device']}")
    if device_info['cuda_available']:
        print(f"CUDA device: {device_info['cuda_device_name']}")
    
    # Create test setup
    from process import ProcessGenerator
    from processor import ProcessorManager
    
    config = SimulationConfig(num_processors=4, num_processes=20)
    manager = ProcessorManager(num_processors=4)
    processors = list(manager)
    
    generator = ProcessGenerator(config=config)
    processes = generator.generate_processes(20)
    
    print("\n" + "-" * 70)
    print("1. Testing DQN Agent")
    print("-" * 70)
    
    state_size = 4 * 2 + 6 + 3  # 4 processors
    agent = DQNAgent(state_size=state_size, action_size=4)
    print(f"Agent initialized: state_size={state_size}, action_size=4")
    print(f"Device: {agent.device}")
    print(f"Initial epsilon: {agent.epsilon:.3f}")
    
    # Test state encoding and action selection
    test_process = processes[0]
    state = agent.encode_state(processors, test_process)
    print(f"\nState shape: {state.shape}")
    print(f"State sample: {state[:5]}...")
    
    action = agent.select_action(state)
    print(f"Selected action: {action}")
    
    # Test step
    next_state = agent.encode_state(processors, processes[1])
    agent.step(reward=-1.5, next_state=next_state, done=False)
    print(f"After step - buffer size: {len(agent.replay_buffer)}")
    
    print("\n" + "-" * 70)
    print("2. Testing DQN Balancer")
    print("-" * 70)
    
    balancer = DQNBalancer(config=config, num_processors=4)
    print(f"Balancer: {balancer.name}")
    print(f"Algorithm type: {balancer.algorithm_type}")
    print(f"State size: {balancer.state_size}")
    
    # Assign processes
    for i, process in enumerate(processes[:5]):
        selected = balancer.assign_process(process, processors)
        print(f"  P{process.pid} -> Processor {selected.processor_id}")
    
    print(f"\nStatistics: {balancer.get_statistics()}")
    
    print("\n" + "-" * 70)
    print("3. Testing Model Persistence")
    print("-" * 70)
    
    test_path = "output/test_dqn_model.pth"
    balancer.save_model(test_path)
    print(f"Model saved to {test_path}")
    
    # Create new balancer and load
    new_balancer = DQNBalancer(config=config, num_processors=4)
    loaded = new_balancer.load_model(test_path)
    print(f"Model loaded: {loaded}")
    
    # Cleanup
    if os.path.exists(test_path):
        os.remove(test_path)
        print("Test file cleaned up")
    
    print("\n" + "=" * 70)
    print("All DQN tests completed successfully!")
    print("=" * 70)
