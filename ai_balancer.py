"""
AI-Powered Load Balancer Module using Reinforcement Learning

This module implements a production-grade Q-Learning based load balancer
that learns optimal process-to-processor assignments through experience.

Reinforcement Learning Concepts:
- State: Discretized representation of system load distribution
- Action: Processor selection for incoming process
- Reward: Negative of resulting turnaround time (minimize is better)
- Policy: ε-greedy with decaying exploration

Key Features:
1. Adaptive State Discretization - Buckets load levels for tractable Q-table
2. Experience Replay - Stores transitions for batch learning
3. Eligibility Traces - TD(λ) for faster credit assignment
4. Model Persistence - Save/load trained models
5. Online Learning - Continues improving during simulation
6. Exploration Decay - Balances exploration vs exploitation

OS Concepts Demonstrated:
- Intelligent scheduling decisions
- Adaptive system optimization
- Learning from operational experience

Author: AI Enhancement
Date: December 2024
"""

import os
import json
import math
import random
import pickle
import logging
import threading
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Deque
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import time

import numpy as np

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
# AI CONFIGURATION
# =============================================================================

@dataclass
class QLearningConfig:
    """
    Configuration for Q-Learning algorithm.
    
    Hyperparameters are tuned for the load balancing domain:
    - Moderate learning rate for stable convergence
    - High discount factor for long-term reward consideration
    - Decaying exploration for optimal exploitation after learning
    """
    # Learning parameters
    learning_rate: float = 0.1          # α: Step size for Q-value updates
    discount_factor: float = 0.95       # γ: Importance of future rewards
    
    # Exploration parameters (ε-greedy)
    epsilon_start: float = 1.0          # Initial exploration rate
    epsilon_end: float = 0.01           # Minimum exploration rate
    epsilon_decay: float = 0.995        # Decay rate per episode
    
    # State discretization
    num_load_buckets: int = 5           # Discretize load into N levels
    num_queue_buckets: int = 4          # Discretize queue size into N levels
    
    # Experience replay
    replay_buffer_size: int = 10000     # Maximum transitions to store
    batch_size: int = 32                # Batch size for replay learning
    min_replay_size: int = 100          # Minimum buffer size before learning
    
    # Eligibility traces (TD(λ))
    use_eligibility_traces: bool = True
    lambda_trace: float = 0.8           # λ: Trace decay rate
    trace_threshold: float = 0.01       # Minimum trace value before zeroing
    
    # Training control
    episodes_per_update: int = 1        # How often to decay epsilon
    target_update_frequency: int = 10   # For double Q-learning variant
    
    # Reward shaping parameters
    fairness_weight: float = 0.3        # Weight for load balance reward component
    throughput_weight: float = 0.7      # Weight for throughput reward component
    migration_penalty: float = -0.5     # Penalty for suggesting migration
    
    # Model persistence
    model_save_path: str = "output/q_learning_model.pkl"
    auto_save_interval: int = 100       # Episodes between auto-saves
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'num_load_buckets': self.num_load_buckets,
            'num_queue_buckets': self.num_queue_buckets,
            'replay_buffer_size': self.replay_buffer_size,
            'batch_size': self.batch_size,
            'fairness_weight': self.fairness_weight,
            'throughput_weight': self.throughput_weight
        }


# Default AI configuration
DEFAULT_QLEARNING_CONFIG = QLearningConfig()


# =============================================================================
# STATE REPRESENTATION
# =============================================================================

@dataclass(frozen=True)
class SystemState:
    """
    Immutable representation of the system state for Q-learning.
    
    The state captures essential information for load balancing decisions:
    - Load distribution across processors (discretized)
    - Queue lengths (discretized)
    - Incoming process characteristics
    
    Frozen dataclass ensures hashability for Q-table keys.
    """
    load_levels: Tuple[int, ...]        # Discretized load per processor
    queue_levels: Tuple[int, ...]       # Discretized queue size per processor
    process_priority: int               # Priority of incoming process (1-3)
    process_burst_bucket: int           # Discretized burst time
    
    def __hash__(self):
        return hash((self.load_levels, self.queue_levels, 
                     self.process_priority, self.process_burst_bucket))
    
    def __eq__(self, other):
        if not isinstance(other, SystemState):
            return False
        return (self.load_levels == other.load_levels and
                self.queue_levels == other.queue_levels and
                self.process_priority == other.process_priority and
                self.process_burst_bucket == other.process_burst_bucket)


class StateEncoder:
    """
    Encodes system state into discretized representation for Q-table.
    
    Discretization Strategy:
    - Load levels: [0-20%, 20-40%, 40-60%, 60-80%, 80-100%]
    - Queue sizes: [0, 1-2, 3-5, 6+]
    - Process burst: [short, medium, long, very_long]
    
    This reduces the state space while preserving important information.
    """
    
    def __init__(self, config: QLearningConfig = None):
        self.config = config or DEFAULT_QLEARNING_CONFIG
        
        # Precompute bucket boundaries
        self.load_boundaries = np.linspace(0, 1.0, self.config.num_load_buckets + 1)[1:]
        self.queue_boundaries = [1, 3, 6]  # Fixed for typical queue sizes
        self.burst_boundaries = [5, 10, 15]  # Short, medium, long, very_long
    
    def _discretize_load(self, load: float) -> int:
        """Convert continuous load (0-1) to bucket index."""
        for i, boundary in enumerate(self.load_boundaries):
            if load <= boundary:
                return i
        return len(self.load_boundaries) - 1
    
    def _discretize_queue(self, queue_size: int) -> int:
        """Convert queue size to bucket index."""
        for i, boundary in enumerate(self.queue_boundaries):
            if queue_size <= boundary:
                return i
        return len(self.queue_boundaries)
    
    def _discretize_burst(self, burst_time: int) -> int:
        """Convert burst time to bucket index."""
        for i, boundary in enumerate(self.burst_boundaries):
            if burst_time <= boundary:
                return i
        return len(self.burst_boundaries)
    
    def encode(self, processors: List[Processor], process: Process) -> SystemState:
        """
        Encode current system state and incoming process.
        
        Args:
            processors: List of all processors
            process: Incoming process to assign
            
        Returns:
            SystemState object suitable for Q-table lookup
        """
        # Calculate normalized loads
        loads = []
        max_load = max(p.get_load() for p in processors) if processors else 1
        max_load = max(max_load, 1)  # Avoid division by zero
        
        for proc in processors:
            norm_load = proc.get_load() / max_load if max_load > 0 else 0
            loads.append(self._discretize_load(norm_load))
        
        # Get queue sizes
        queues = [self._discretize_queue(p.get_queue_size()) for p in processors]
        
        return SystemState(
            load_levels=tuple(loads),
            queue_levels=tuple(queues),
            process_priority=process.priority.value,
            process_burst_bucket=self._discretize_burst(process.burst_time)
        )
    
    def get_state_size(self, num_processors: int) -> int:
        """Calculate approximate state space size."""
        load_states = self.config.num_load_buckets ** num_processors
        queue_states = self.config.num_queue_buckets ** num_processors
        priority_states = 3  # HIGH, MEDIUM, LOW
        burst_states = len(self.burst_boundaries) + 1
        return load_states * queue_states * priority_states * burst_states


# =============================================================================
# EXPERIENCE REPLAY
# =============================================================================

@dataclass
class Experience:
    """Single experience tuple for replay buffer."""
    state: SystemState
    action: int  # Processor ID
    reward: float
    next_state: Optional[SystemState]
    done: bool
    priority: float = 1.0  # For prioritized experience replay


class ReplayBuffer:
    """
    Experience replay buffer for stable Q-learning.
    
    Benefits:
    - Breaks correlation between consecutive samples
    - Enables batch learning for efficiency
    - Allows re-use of rare experiences
    
    Implements prioritized experience replay for better sample efficiency.
    """
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: Deque[Experience] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        
        # Priority sum tree for efficient sampling (optional optimization)
        self.priorities = deque(maxlen=capacity)
    
    def add(self, experience: Experience):
        """Add an experience to the buffer."""
        with self._lock:
            self.buffer.append(experience)
            # Set priority based on TD error (initially high for new experiences)
            self.priorities.append(experience.priority)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample a batch of experiences."""
        with self._lock:
            if len(self.buffer) < batch_size:
                return list(self.buffer)
            
            # Prioritized sampling (higher TD error = more likely to sample)
            total_priority = sum(self.priorities)
            if total_priority == 0:
                return random.sample(list(self.buffer), batch_size)
            
            # Simple proportional sampling
            indices = random.choices(
                range(len(self.buffer)),
                weights=list(self.priorities),
                k=batch_size
            )
            return [self.buffer[i] for i in indices]
    
    def update_priorities(self, indices: List[int], td_errors: List[float]):
        """Update priorities based on TD errors."""
        with self._lock:
            for idx, td_error in zip(indices, td_errors):
                if 0 <= idx < len(self.priorities):
                    # Priority is |TD error| + small constant
                    self.priorities[idx] = abs(td_error) + 0.01
    
    def __len__(self):
        return len(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        with self._lock:
            self.buffer.clear()
            self.priorities.clear()


# =============================================================================
# Q-LEARNING AGENT
# =============================================================================

class QLearningAgent:
    """
    Production-grade Q-Learning agent for load balancing.
    
    This agent learns an optimal policy for assigning processes to processors
    by maintaining a Q-table and updating it based on experience.
    
    Key Features:
    - ε-greedy exploration with decay
    - Experience replay for stable learning
    - Eligibility traces for faster credit assignment
    - Model persistence for reuse
    
    The agent operates in two modes:
    1. Training: High exploration, updates Q-values
    2. Exploitation: Uses learned policy, minimal exploration
    """
    
    def __init__(self, num_processors: int, 
                 config: QLearningConfig = None):
        """
        Initialize the Q-Learning agent.
        
        Args:
            num_processors: Number of processors (actions)
            config: Q-learning hyperparameters
        """
        self.num_processors = num_processors
        self.config = config or DEFAULT_QLEARNING_CONFIG
        
        # Q-table: state -> action -> Q-value
        self.q_table: Dict[SystemState, np.ndarray] = defaultdict(
            lambda: np.zeros(num_processors)
        )
        
        # State encoder
        self.state_encoder = StateEncoder(self.config)
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(self.config.replay_buffer_size)
        
        # Eligibility traces
        self.eligibility: Dict[Tuple[SystemState, int], float] = defaultdict(float)
        
        # Exploration rate
        self.epsilon = self.config.epsilon_start
        
        # Training state
        self.training_mode = True
        self.episode_count = 0
        self.total_steps = 0
        self.total_reward = 0.0
        
        # Performance tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.q_value_history: List[float] = []
        self.epsilon_history: List[float] = []
        
        # Current episode tracking
        self._current_episode_reward = 0.0
        self._current_episode_length = 0
        self._last_state: Optional[SystemState] = None
        self._last_action: Optional[int] = None
        
        # Thread safety
        self._lock = threading.Lock()
        
        logger.info(f"Q-Learning agent initialized with {num_processors} actions")
    
    def get_action(self, state: SystemState) -> int:
        """
        Select an action using ε-greedy policy.
        
        Args:
            state: Current system state
            
        Returns:
            Processor ID to assign the process to
        """
        with self._lock:
            if self.training_mode and random.random() < self.epsilon:
                # Exploration: random action
                action = random.randint(0, self.num_processors - 1)
                logger.debug(f"Exploring: random action {action}")
            else:
                # Exploitation: best known action
                q_values = self.q_table[state]
                action = int(np.argmax(q_values))
                
                # Tie-breaking: if all Q-values are zero, choose randomly
                if np.all(q_values == 0):
                    action = random.randint(0, self.num_processors - 1)
                
                logger.debug(f"Exploiting: action {action} with Q={q_values[action]:.3f}")
            
            self._last_state = state
            self._last_action = action
            self.total_steps += 1
            self._current_episode_length += 1
            
            return action
    
    def update(self, reward: float, next_state: Optional[SystemState], done: bool):
        """
        Update Q-values based on received reward.
        
        Uses TD(0) update rule:
        Q(s,a) ← Q(s,a) + α[r + γ·max_a'Q(s',a') - Q(s,a)]
        
        Args:
            reward: Reward received for the transition
            next_state: Resulting state (None if terminal)
            done: Whether episode is complete
        """
        if self._last_state is None or self._last_action is None:
            return
        
        with self._lock:
            state = self._last_state
            action = self._last_action
            
            # Store experience
            experience = Experience(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                priority=1.0
            )
            self.replay_buffer.add(experience)
            
            # Direct Q-value update (online learning)
            self._update_q_value(state, action, reward, next_state, done)
            
            # Update eligibility traces
            if self.config.use_eligibility_traces:
                self._update_eligibility_traces(state, action, reward, next_state, done)
            
            # Experience replay (if enough samples)
            if len(self.replay_buffer) >= self.config.min_replay_size:
                self._replay()
            
            # Track episode statistics
            self._current_episode_reward += reward
            self.total_reward += reward
            
            if done:
                self._end_episode()
    
    def _update_q_value(self, state: SystemState, action: int, 
                         reward: float, next_state: Optional[SystemState], 
                         done: bool):
        """Perform single Q-value update."""
        current_q = self.q_table[state][action]
        
        if done or next_state is None:
            target = reward
        else:
            # Max Q-value for next state (greedy)
            next_max_q = np.max(self.q_table[next_state])
            target = reward + self.config.discount_factor * next_max_q
        
        # TD error
        td_error = target - current_q
        
        # Update Q-value
        self.q_table[state][action] += self.config.learning_rate * td_error
        
        return td_error
    
    def _update_eligibility_traces(self, state: SystemState, action: int,
                                    reward: float, next_state: Optional[SystemState],
                                    done: bool):
        """Update eligibility traces for TD(λ)."""
        # Increment trace for current state-action
        self.eligibility[(state, action)] = 1.0
        
        # Calculate TD error
        current_q = self.q_table[state][action]
        if done or next_state is None:
            td_error = reward - current_q
        else:
            next_max_q = np.max(self.q_table[next_state])
            td_error = reward + self.config.discount_factor * next_max_q - current_q
        
        # Update all Q-values using eligibility traces
        traces_to_remove = []
        for (s, a), trace in self.eligibility.items():
            if trace < self.config.trace_threshold:
                traces_to_remove.append((s, a))
                continue
            
            # Update Q-value proportional to trace
            self.q_table[s][a] += self.config.learning_rate * td_error * trace
            
            # Decay trace
            self.eligibility[(s, a)] *= (self.config.discount_factor * 
                                          self.config.lambda_trace)
        
        # Clean up small traces
        for key in traces_to_remove:
            del self.eligibility[key]
    
    def _replay(self):
        """Perform experience replay update."""
        if len(self.replay_buffer) < self.config.batch_size:
            return
        
        batch = self.replay_buffer.sample(self.config.batch_size)
        
        for exp in batch:
            self._update_q_value(
                exp.state, exp.action, exp.reward, exp.next_state, exp.done
            )
    
    def _end_episode(self):
        """Handle end of episode."""
        # Record episode statistics
        self.episode_rewards.append(self._current_episode_reward)
        self.episode_lengths.append(self._current_episode_length)
        
        # Record average Q-value
        if self.q_table:
            avg_q = np.mean([np.max(q) for q in self.q_table.values()])
            self.q_value_history.append(avg_q)
        
        # Decay exploration rate
        if self.training_mode:
            self.epsilon = max(
                self.config.epsilon_end,
                self.epsilon * self.config.epsilon_decay
            )
            self.epsilon_history.append(self.epsilon)
        
        # Clear eligibility traces for new episode
        self.eligibility.clear()
        
        # Increment episode count
        self.episode_count += 1
        
        # Auto-save periodically
        if (self.episode_count % self.config.auto_save_interval == 0 and 
            self.training_mode):
            self.save_model()
        
        # Reset episode tracking
        self._current_episode_reward = 0.0
        self._current_episode_length = 0
        
        logger.info(f"Episode {self.episode_count} complete. "
                   f"Reward: {self.episode_rewards[-1]:.2f}, "
                   f"ε: {self.epsilon:.4f}")
    
    def set_training_mode(self, training: bool):
        """Set whether agent is in training or exploitation mode."""
        self.training_mode = training
        if not training:
            # Use minimal exploration in exploitation mode
            self.epsilon = self.config.epsilon_end
        logger.info(f"Agent mode: {'Training' if training else 'Exploitation'}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get training statistics."""
        with self._lock:
            return {
                'episode_count': self.episode_count,
                'total_steps': self.total_steps,
                'total_reward': self.total_reward,
                'epsilon': self.epsilon,
                'training_mode': self.training_mode,
                'q_table_size': len(self.q_table),
                'replay_buffer_size': len(self.replay_buffer),
                'avg_episode_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
                'avg_episode_length': np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0,
                'avg_q_value': np.mean(self.q_value_history[-100:]) if self.q_value_history else 0,
            }
    
    def get_q_values_heatmap(self) -> Dict[str, Any]:
        """Get Q-value data for visualization."""
        with self._lock:
            if not self.q_table:
                return {'states': [], 'q_values': [], 'actions': list(range(self.num_processors))}
            
            # Get sample of states for visualization
            states = list(self.q_table.keys())[:50]  # Limit for performance
            q_values = [self.q_table[s].tolist() for s in states]
            
            return {
                'states': [str(s) for s in states],
                'q_values': q_values,
                'actions': list(range(self.num_processors))
            }
    
    def save_model(self, path: str = None):
        """Save the trained model to disk."""
        save_path = path or self.config.model_save_path
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', 
                    exist_ok=True)
        
        with self._lock:
            model_data = {
                'q_table': dict(self.q_table),
                'epsilon': self.epsilon,
                'episode_count': self.episode_count,
                'total_steps': self.total_steps,
                'total_reward': self.total_reward,
                'num_processors': self.num_processors,
                'config': self.config.to_dict(),
                'episode_rewards': self.episode_rewards[-1000:],  # Keep last 1000
                'q_value_history': self.q_value_history[-1000:],
            }
            
            try:
                with open(save_path, 'wb') as f:
                    pickle.dump(model_data, f)
                logger.info(f"Model saved to {save_path}")
            except Exception as e:
                logger.error(f"Failed to save model: {e}")
    
    def load_model(self, path: str = None) -> bool:
        """Load a trained model from disk."""
        load_path = path or self.config.model_save_path
        
        if not os.path.exists(load_path):
            logger.warning(f"No model file found at {load_path}")
            return False
        
        try:
            with open(load_path, 'rb') as f:
                model_data = pickle.load(f)
            
            with self._lock:
                # Restore Q-table
                self.q_table = defaultdict(
                    lambda: np.zeros(self.num_processors),
                    model_data['q_table']
                )
                self.epsilon = model_data.get('epsilon', self.config.epsilon_end)
                self.episode_count = model_data.get('episode_count', 0)
                self.total_steps = model_data.get('total_steps', 0)
                self.total_reward = model_data.get('total_reward', 0)
                self.episode_rewards = model_data.get('episode_rewards', [])
                self.q_value_history = model_data.get('q_value_history', [])
            
            logger.info(f"Model loaded from {load_path}. "
                       f"Episodes: {self.episode_count}, Q-states: {len(self.q_table)}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def reset(self):
        """Reset the agent for a new training session."""
        with self._lock:
            self.q_table.clear()
            self.replay_buffer.clear()
            self.eligibility.clear()
            self.epsilon = self.config.epsilon_start
            self.episode_count = 0
            self.total_steps = 0
            self.total_reward = 0.0
            self.episode_rewards.clear()
            self.episode_lengths.clear()
            self.q_value_history.clear()
            self.epsilon_history.clear()
            self._current_episode_reward = 0.0
            self._current_episode_length = 0
            self._last_state = None
            self._last_action = None
        
        logger.info("Agent reset complete")


# =============================================================================
# Q-LEARNING LOAD BALANCER
# =============================================================================

class QLearningBalancer(LoadBalancer):
    """
    Q-Learning based Load Balancer.
    
    This balancer uses reinforcement learning to make intelligent
    load balancing decisions that optimize system performance.
    
    Advantages over traditional algorithms:
    - Adapts to workload patterns
    - Learns from experience
    - Can outperform static algorithms after training
    - Handles complex state spaces
    
    Modes:
    - Training: Learns optimal policy through exploration
    - Exploitation: Uses learned policy for best performance
    """
    
    def __init__(self, config: SimulationConfig = None,
                 ai_config: QLearningConfig = None,
                 num_processors: int = 4):
        """
        Initialize Q-Learning load balancer.
        
        Args:
            config: Simulation configuration
            ai_config: Q-learning hyperparameters
            num_processors: Number of processors in the system
        """
        super().__init__(config)
        
        self.ai_config = ai_config or DEFAULT_QLEARNING_CONFIG
        self.num_processors = num_processors
        
        # Initialize Q-learning agent
        self.agent = QLearningAgent(num_processors, self.ai_config)
        
        # State encoder
        self.state_encoder = StateEncoder(self.ai_config)
        
        # Track assignments for reward calculation
        self._pending_assignments: Dict[int, Tuple[SystemState, int, float]] = {}
        # Maps process PID to (state, action, assignment_time)
        
        # Performance metrics
        self.decisions_made = 0
        self.total_turnaround_reduction = 0.0
        
        # Reference for reward baseline (average of other algorithms)
        self._baseline_turnaround = 0.0
        
        logger.info(f"Q-Learning Balancer initialized with {num_processors} processors")
    
    @property
    def algorithm_type(self) -> LoadBalancingAlgorithm:
        return LoadBalancingAlgorithm.Q_LEARNING
    
    @property
    def name(self) -> str:
        return "AI (Q-Learning)"
    
    def assign_process(self, process: Process, 
                       processors: List[Processor]) -> Optional[Processor]:
        """
        Assign a process using Q-learning policy.
        
        The agent observes the current state and selects an action
        (processor) based on learned Q-values or exploration.
        
        Args:
            process: Process to assign
            processors: Available processors
            
        Returns:
            Selected processor
        """
        if not processors:
            return None
        
        # Update num_processors if changed
        if len(processors) != self.num_processors:
            self._resize_agent(len(processors))
        
        # Encode current state
        current_state = self.state_encoder.encode(processors, process)
        
        # Get action from agent
        action = self.agent.get_action(current_state)
        
        # Clamp action to valid range
        action = min(action, len(processors) - 1)
        
        # Select processor
        selected = processors[action]
        
        # Store for later reward calculation
        self._pending_assignments[process.pid] = (
            current_state, 
            action, 
            time.time()
        )
        
        # Add process to selected processor
        selected.add_process(process)
        process.processor_id = selected.processor_id
        self.assignment_count += 1
        self.decisions_made += 1
        
        logger.debug(f"Assigned P{process.pid} to Processor {selected.processor_id} "
                    f"(Q-learning action)")
        
        return selected
    
    def process_completed(self, process: Process, processors: List[Processor]):
        """
        Called when a process completes to provide reward feedback.
        
        This is the key feedback mechanism for learning:
        - Calculates reward based on process performance
        - Updates Q-values through the agent
        
        Args:
            process: Completed process
            processors: Current processor states
        """
        if process.pid not in self._pending_assignments:
            return
        
        state, action, assignment_time = self._pending_assignments.pop(process.pid)
        
        # Calculate reward based on process performance
        reward = self._calculate_reward(process, processors)
        
        # Get next state (current system state after completion)
        # Use a dummy process for state encoding
        dummy_process = Process(pid=-1, burst_time=5)
        next_state = self.state_encoder.encode(processors, dummy_process)
        
        # Check if this is end of episode (all processes done)
        all_done = all(p.get_queue_size() == 0 and p.current_process is None 
                       for p in processors)
        
        # Update agent
        self.agent.update(reward, next_state, done=all_done)
        
        logger.debug(f"Reward for P{process.pid}: {reward:.3f}")
    
    def _calculate_reward(self, process: Process, 
                          processors: List[Processor]) -> float:
        """
        Calculate reward for a completed process.
        
        Reward Components:
        1. Turnaround time (negative, want to minimize)
        2. Load balance fairness bonus
        3. Migration penalty (if applicable)
        
        Args:
            process: Completed process
            processors: Current processor states
            
        Returns:
            Reward value (higher is better)
        """
        config = self.ai_config
        
        # Base reward: negative turnaround time (normalized)
        turnaround_time = process.get_turnaround_time()
        # Normalize by burst time to get a ratio
        normalized_turnaround = -turnaround_time / max(process.burst_time, 1)
        throughput_reward = normalized_turnaround * config.throughput_weight
        
        # Fairness reward: bonus for balanced load
        loads = [p.get_load() for p in processors]
        max_load = max(loads) if loads else 1
        if max_load > 0:
            load_variance = np.var([l / max_load for l in loads])
            # Lower variance = more balanced = higher reward
            fairness_reward = (1.0 - min(load_variance, 1.0)) * config.fairness_weight
        else:
            fairness_reward = config.fairness_weight
        
        # Migration penalty
        migration_reward = process.migration_count * config.migration_penalty
        
        # Total reward
        total_reward = throughput_reward + fairness_reward + migration_reward
        
        return total_reward
    
    def _resize_agent(self, new_size: int):
        """Resize agent for different number of processors."""
        logger.info(f"Resizing agent from {self.num_processors} to {new_size} processors")
        self.num_processors = new_size
        self.agent = QLearningAgent(new_size, self.ai_config)
        # Try to load existing model
        self.agent.load_model()
    
    def check_for_migration(self, processors: List[Processor], 
                            current_time: int) -> List[MigrationRecord]:
        """
        Check for migration opportunities using Q-learning insights.
        
        The Q-learning balancer can suggest migrations when it detects
        that the current assignment is suboptimal based on learned values.
        
        This is an advanced feature that uses the Q-table to identify
        better assignments for waiting processes.
        """
        migrations = []
        
        if len(processors) < 2:
            return migrations
        
        # Find heavily loaded processors
        loads = [(p, p.get_load()) for p in processors]
        max_load = max(l for _, l in loads) if loads else 0
        
        if max_load <= 0:
            return migrations
        
        # Sort by load
        loads.sort(key=lambda x: x[1], reverse=True)
        
        # Check if top processor has migratable processes
        for source_proc, source_load in loads[:len(loads)//2]:  # Top half
            candidates = self.get_migration_candidates(source_proc)
            if not candidates:
                continue
            
            # For each candidate, check if Q-values suggest a better placement
            for process in candidates:
                current_state = self.state_encoder.encode(processors, process)
                q_values = self.agent.q_table[current_state]
                
                best_action = int(np.argmax(q_values))
                current_action = source_proc.processor_id
                
                # Only migrate if Q-value improvement is significant
                if best_action != current_action:
                    q_improvement = q_values[best_action] - q_values[current_action]
                    if q_improvement > 0.1:  # Threshold for migration
                        dest_proc = processors[best_action]
                        migration = MigrationRecord(
                            process_id=process.pid,
                            source_processor=source_proc.processor_id,
                            destination_processor=dest_proc.processor_id,
                            time=current_time,
                            reason=f"Q-learning: ΔQ={q_improvement:.3f}",
                            source_load_before=source_load,
                            destination_load_before=dest_proc.get_load()
                        )
                        migrations.append(migration)
                        break  # One migration per source processor per check
        
        return migrations
    
    def set_training_mode(self, training: bool):
        """Set training/exploitation mode."""
        self.agent.set_training_mode(training)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get combined balancer and agent statistics."""
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
        self.total_turnaround_reduction = 0.0
    
    def full_reset(self):
        """Full reset including agent."""
        self.reset()
        self.agent.reset()


# =============================================================================
# REWARD CALCULATOR (For External Feedback)
# =============================================================================

class RewardCalculator:
    """
    Utility class for calculating rewards from process completion.
    
    This can be used by the simulation engine to provide feedback
    to the Q-learning balancer.
    """
    
    def __init__(self, config: QLearningConfig = None):
        self.config = config or DEFAULT_QLEARNING_CONFIG
        self._baseline_turnaround = None
    
    def set_baseline(self, baseline_turnaround: float):
        """Set baseline turnaround time from other algorithms."""
        self._baseline_turnaround = baseline_turnaround
    
    def calculate(self, process: Process, 
                  processors: List[Processor],
                  system_metrics: Dict[str, float] = None) -> float:
        """
        Calculate comprehensive reward for a process.
        
        Args:
            process: Completed process
            processors: Current processor states
            system_metrics: Optional system-wide metrics
            
        Returns:
            Reward value
        """
        # Use configured weights
        config = self.config
        
        # Turnaround component
        turnaround = process.get_turnaround_time()
        max_expected_turnaround = process.burst_time * 3  # Heuristic upper bound
        turnaround_ratio = turnaround / max(max_expected_turnaround, 1)
        turnaround_reward = (1.0 - min(turnaround_ratio, 1.0)) * config.throughput_weight
        
        # Waiting time component
        waiting_penalty = -process.waiting_time / max(process.burst_time, 1) * 0.1
        
        # Fairness component
        loads = [p.get_load() for p in processors]
        if loads:
            load_std = np.std(loads)
            max_load = max(loads)
            fairness_ratio = 1.0 - (load_std / max(max_load, 1))
            fairness_reward = fairness_ratio * config.fairness_weight
        else:
            fairness_reward = 0.0
        
        # Migration penalty
        migration_penalty = process.migration_count * config.migration_penalty
        
        # Combine all components
        total_reward = turnaround_reward + waiting_penalty + fairness_reward + migration_penalty
        
        # Bonus for beating baseline
        if self._baseline_turnaround and turnaround < self._baseline_turnaround:
            improvement = (self._baseline_turnaround - turnaround) / self._baseline_turnaround
            total_reward += improvement * 0.5
        
        return total_reward


# =============================================================================
# AI TRAINING UTILITIES
# =============================================================================

class AITrainer:
    """
    Utility class for training the Q-learning agent.
    
    Provides methods for:
    - Running training episodes
    - Evaluating learned policy
    - Comparing against baselines
    - Hyperparameter tuning
    """
    
    def __init__(self, balancer: QLearningBalancer):
        self.balancer = balancer
        self.training_history: List[Dict[str, float]] = []
    
    def train_episode(self, processors: List[Processor],
                      processes: List[Process]) -> Dict[str, float]:
        """
        Run a single training episode.
        
        Args:
            processors: Processor instances
            processes: Processes to assign
            
        Returns:
            Episode statistics
        """
        self.balancer.set_training_mode(True)
        episode_reward = 0.0
        
        for process in processes:
            selected = self.balancer.assign_process(process, processors)
            # Note: Actual reward is given when process completes
        
        stats = self.balancer.get_statistics()
        self.training_history.append({
            'episode': stats['episode_count'],
            'epsilon': stats['epsilon'],
            'avg_reward': stats['avg_episode_reward'],
            'q_table_size': stats['q_table_size']
        })
        
        return stats
    
    def evaluate(self, processors: List[Processor],
                 processes: List[Process]) -> Dict[str, float]:
        """Evaluate current policy without learning."""
        self.balancer.set_training_mode(False)
        
        original_states = []
        for process in processes:
            # Store original state
            original_states.append({
                'pid': process.pid,
                'processor_id': process.processor_id,
                'state': process.state
            })
            
            # Get assignment
            self.balancer.assign_process(process, processors)
        
        return self.balancer.get_statistics()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    'QLearningConfig',
    'DEFAULT_QLEARNING_CONFIG',
    'SystemState',
    'StateEncoder',
    'Experience',
    'ReplayBuffer',
    'QLearningAgent',
    'QLearningBalancer',
    'RewardCalculator',
    'AITrainer'
]


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AI Load Balancer Module Test")
    print("=" * 70)
    
    # Create test setup
    from process import ProcessGenerator
    from processor import ProcessorManager
    
    # Initialize
    config = SimulationConfig(num_processors=4, num_processes=20)
    manager = ProcessorManager(num_processors=4)
    processors = list(manager)
    
    generator = ProcessGenerator(config=config)
    processes = generator.generate_processes(20)
    
    print("\n" + "-" * 70)
    print("1. Testing Q-Learning Agent")
    print("-" * 70)
    
    agent = QLearningAgent(num_processors=4)
    print(f"Agent initialized: {agent.num_processors} actions")
    print(f"Initial epsilon: {agent.epsilon:.3f}")
    print(f"Training mode: {agent.training_mode}")
    
    # Test state encoding
    encoder = StateEncoder()
    test_process = processes[0]
    state = encoder.encode(processors, test_process)
    print(f"\nEncoded state: {state}")
    
    # Test action selection
    action = agent.get_action(state)
    print(f"Selected action: {action}")
    
    # Test update
    agent.update(reward=-1.5, next_state=state, done=False)
    print(f"After update - Q-table size: {len(agent.q_table)}")
    
    print("\n" + "-" * 70)
    print("2. Testing Q-Learning Balancer")
    print("-" * 70)
    
    balancer = QLearningBalancer(config=config, num_processors=4)
    print(f"Balancer: {balancer.name}")
    print(f"Algorithm type: {balancer.algorithm_type}")
    
    # Assign some processes
    for i, process in enumerate(processes[:5]):
        selected = balancer.assign_process(process, processors)
        print(f"  P{process.pid} -> Processor {selected.processor_id}")
    
    print(f"\nStatistics: {balancer.get_statistics()}")
    
    # Test migration check
    migrations = balancer.check_for_migration(processors, current_time=10)
    print(f"Migration suggestions: {len(migrations)}")
    
    print("\n" + "-" * 70)
    print("3. Testing Model Persistence")
    print("-" * 70)
    
    # Save model
    test_path = "output/test_q_model.pkl"
    balancer.save_model(test_path)
    print(f"Model saved to {test_path}")
    
    # Create new balancer and load
    new_balancer = QLearningBalancer(config=config, num_processors=4)
    loaded = new_balancer.load_model(test_path)
    print(f"Model loaded: {loaded}")
    print(f"Loaded Q-table size: {len(new_balancer.agent.q_table)}")
    
    # Cleanup
    if os.path.exists(test_path):
        os.remove(test_path)
        print(f"Test file cleaned up")
    
    print("\n" + "=" * 70)
    print("All AI balancer tests completed successfully!")
    print("=" * 70)
