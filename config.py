"""
Configuration Module for Dynamic Load Balancing Simulator

This module contains all configuration constants and parameters used throughout
the simulation. Centralizing configuration makes it easy to adjust system behavior
without modifying core logic.

In Operating Systems, configuration management is crucial for:
- Tuning system performance
- Adapting to different hardware configurations
- Testing various scenarios without code changes

Author: Student
Date: December 2024
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Any


# =============================================================================
# ENUMERATIONS - Define categorical constants
# =============================================================================

class ProcessState(Enum):
    """
    Process states in the Operating System lifecycle.
    
    In a real OS, processes transition through these states:
    - NEW: Process is being created
    - READY: Process is waiting to be assigned to a processor
    - RUNNING: Process is currently being executed by a processor
    - WAITING: Process is waiting for I/O or an event (not used in this simulation)
    - COMPLETED: Process has finished execution
    - MIGRATING: Process is being moved between processors (custom state for load balancing)
    """
    NEW = auto()
    READY = auto()
    RUNNING = auto()
    WAITING = auto()
    COMPLETED = auto()
    MIGRATING = auto()


class LoadBalancingAlgorithm(Enum):
    """
    Available load balancing and scheduling algorithms.
    
    Load Balancing Algorithms:
    - ROUND_ROBIN: Fair distribution, simple, doesn't consider actual load
    - LEAST_LOADED: Assigns to least busy processor, better load distribution
    - THRESHOLD_BASED: Only migrates when load difference exceeds threshold
    - Q_LEARNING: AI-powered adaptive balancing using reinforcement learning (tabular)
    - DQN: Deep Q-Network with neural network function approximation
    
    Classic CPU Scheduling Algorithms:
    - FCFS: First Come First Served - the OG scheduler
    - SJF: Shortest Job First - optimal waiting time (non-preemptive)
    - SRTF: Shortest Remaining Time First - preemptive SJF
    - PRIORITY: Priority-based scheduling with aging
    - PRIORITY_PREEMPTIVE: Preemptive priority scheduling
    - MULTILEVEL_QUEUE: Multiple queues with different priorities
    - MLFQ: Multilevel Feedback Queue - adaptive scheduling
    - EDF: Earliest Deadline First - real-time scheduling
    """
    # Load Balancing Algorithms
    ROUND_ROBIN = "Round Robin"
    LEAST_LOADED = "Least Loaded First"
    THRESHOLD_BASED = "Threshold Based"
    Q_LEARNING = "AI (Q-Learning)"
    DQN = "AI (DQN)"
    
    # Classic CPU Scheduling Algorithms
    FCFS = "FCFS (First Come First Served)"
    SJF = "SJF (Shortest Job First)"
    SRTF = "SRTF (Shortest Remaining Time First)"
    PRIORITY = "Priority Scheduling"
    PRIORITY_PREEMPTIVE = "Priority Scheduling (Preemptive)"
    MULTILEVEL_QUEUE = "Multilevel Queue"
    MLFQ = "MLFQ (Multilevel Feedback Queue)"
    EDF = "EDF (Earliest Deadline First)"


class ProcessPriority(Enum):
    """
    Process priority levels.
    
    In real OS, priority affects scheduling decisions:
    - HIGH: Critical processes, should be executed first
    - MEDIUM: Normal user processes
    - LOW: Background tasks, can wait
    """
    HIGH = 1
    MEDIUM = 2
    LOW = 3


# =============================================================================
# SIMULATION CONFIGURATION
# =============================================================================

@dataclass
class SimulationConfig:
    """
    Main configuration class for the simulation.
    
    Using a dataclass provides:
    - Clean syntax for configuration
    - Type hints for IDE support
    - Easy serialization if needed
    
    Attributes:
        num_processors: Number of virtual processors (4-8 recommended)
        num_processes: Number of processes to generate
        time_quantum: Time slice for Round Robin scheduling (in time units)
        simulation_speed: Delay between simulation steps (seconds)
        load_check_interval: How often to check for load balancing (time units)
    """
    # Processor Configuration
    num_processors: int = 4
    min_processors: int = 2
    max_processors: int = 16
    
    # Process Configuration
    num_processes: int = 20
    min_processes: int = 1
    max_processes: int = 100
    
    # Time Configuration
    time_quantum: int = 4  # Time units per execution slice
    simulation_speed: float = 0.5  # Seconds between simulation updates
    load_check_interval: int = 5  # Check load balance every N time units
    
    # Process Generation Parameters
    min_burst_time: int = 2
    max_burst_time: int = 20
    min_arrival_time: int = 0
    max_arrival_time: int = 30
    
    # Load Balancing Configuration
    default_algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ROUND_ROBIN
    load_threshold: float = 0.3  # 30% load difference triggers migration
    migration_cost: int = 1  # Time units cost for migrating a process
    
    # Threshold-based specific settings
    high_load_threshold: float = 0.8  # 80% utilization considered high
    low_load_threshold: float = 0.2   # 20% utilization considered low
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            bool: True if configuration is valid
            
        Raises:
            ValueError: If any parameter is invalid
        """
        if not (self.min_processors <= self.num_processors <= self.max_processors):
            raise ValueError(f"num_processors must be between {self.min_processors} and {self.max_processors}")
        
        if not (self.min_processes <= self.num_processes <= self.max_processes):
            raise ValueError(f"num_processes must be between {self.min_processes} and {self.max_processes}")
        
        if self.time_quantum < 1:
            raise ValueError("time_quantum must be at least 1")
        
        if self.simulation_speed <= 0:
            raise ValueError("simulation_speed must be positive")
        
        if not (0 < self.load_threshold < 1):
            raise ValueError("load_threshold must be between 0 and 1")
        
        if self.min_burst_time > self.max_burst_time:
            raise ValueError("min_burst_time cannot be greater than max_burst_time")
        
        if self.min_arrival_time > self.max_arrival_time:
            raise ValueError("min_arrival_time cannot be greater than max_arrival_time")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for serialization.
        
        Returns:
            Dict containing all configuration parameters
        """
        return {
            'num_processors': self.num_processors,
            'num_processes': self.num_processes,
            'time_quantum': self.time_quantum,
            'simulation_speed': self.simulation_speed,
            'load_check_interval': self.load_check_interval,
            'min_burst_time': self.min_burst_time,
            'max_burst_time': self.max_burst_time,
            'min_arrival_time': self.min_arrival_time,
            'max_arrival_time': self.max_arrival_time,
            'default_algorithm': self.default_algorithm.value,
            'load_threshold': self.load_threshold,
            'migration_cost': self.migration_cost,
            'high_load_threshold': self.high_load_threshold,
            'low_load_threshold': self.low_load_threshold
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimulationConfig':
        """
        Create configuration from dictionary.
        
        Args:
            data: Dictionary containing configuration parameters
            
        Returns:
            SimulationConfig instance
        """
        config = cls()
        for key, value in data.items():
            if key == 'default_algorithm':
                value = LoadBalancingAlgorithm(value)
            if hasattr(config, key):
                setattr(config, key, value)
        return config


# =============================================================================
# GUI CONFIGURATION
# =============================================================================

@dataclass
class GUIConfig:
    """
    Configuration for the graphical user interface.
    
    Attributes:
        window_width: Main window width in pixels
        window_height: Main window height in pixels
        update_interval: GUI refresh rate in milliseconds
        chart_colors: Colors for different visualization elements
    """
    # Window Settings
    window_title: str = "Dynamic Load Balancing Simulator"
    window_width: int = 1400
    window_height: int = 900
    min_width: int = 1200
    min_height: int = 700
    
    # Update Settings
    update_interval: int = 100  # milliseconds
    chart_update_interval: int = 500  # milliseconds
    animation_speed: int = 100  # milliseconds delay between steps when running with delay
    
    # Color Scheme for Visualization
    # Using a traffic light system for load indication
    color_low_load: str = "#4CAF50"      # Green - Low load (< 30%)
    color_medium_load: str = "#FFC107"   # Yellow/Amber - Medium load (30-70%)
    color_high_load: str = "#F44336"     # Red - High load (> 70%)
    color_idle: str = "#9E9E9E"          # Gray - Idle processor
    
    # Process State Colors
    color_new: str = "#2196F3"           # Blue - New process
    color_ready: str = "#4CAF50"         # Green - Ready to run
    color_running: str = "#FF9800"       # Orange - Currently running
    color_completed: str = "#9C27B0"     # Purple - Completed
    color_migrating: str = "#00BCD4"     # Cyan - Being migrated
    
    # Processor Colors (for Gantt chart and identification)
    processor_colors: tuple = (
        "#E91E63",  # Pink
        "#3F51B5",  # Indigo
        "#009688",  # Teal
        "#FF5722",  # Deep Orange
        "#673AB7",  # Deep Purple
        "#00BCD4",  # Cyan
        "#8BC34A",  # Light Green
        "#FFC107",  # Amber
    )
    
    # Font Settings
    font_family: str = "Helvetica"
    font_size_small: int = 9
    font_size_normal: int = 11
    font_size_large: int = 14
    font_size_title: int = 18
    
    # Chart Settings
    bar_width: float = 0.6
    gantt_row_height: int = 30
    max_gantt_time_display: int = 50  # Maximum time units to show in Gantt
    
    def get_load_color(self, utilization: float) -> str:
        """
        Get color based on processor utilization level.
        
        Args:
            utilization: CPU utilization as a fraction (0.0 to 1.0)
            
        Returns:
            Hex color string representing the load level
        """
        if utilization < 0.01:
            return self.color_idle
        elif utilization < 0.3:
            return self.color_low_load
        elif utilization < 0.7:
            return self.color_medium_load
        else:
            return self.color_high_load
    
    def get_processor_color(self, processor_id: int) -> str:
        """
        Get a unique color for a processor.
        
        Args:
            processor_id: The processor's ID number
            
        Returns:
            Hex color string for the processor
        """
        return self.processor_colors[processor_id % len(self.processor_colors)]
    
    def get_state_color(self, state: ProcessState) -> str:
        """
        Get color for a process state.
        
        Args:
            state: The ProcessState enum value
            
        Returns:
            Hex color string for the state
        """
        state_colors = {
            ProcessState.NEW: self.color_new,
            ProcessState.READY: self.color_ready,
            ProcessState.RUNNING: self.color_running,
            ProcessState.WAITING: self.color_medium_load,
            ProcessState.COMPLETED: self.color_completed,
            ProcessState.MIGRATING: self.color_migrating
        }
        return state_colors.get(state, "#000000")


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

@dataclass
class LoggingConfig:
    """
    Configuration for the logging system.
    
    Logging is essential for:
    - Debugging algorithm behavior
    - Tracking load balancing decisions
    - Analyzing simulation results
    """
    # Log Levels
    log_to_console: bool = True
    log_to_file: bool = True
    log_file_path: str = "simulation.log"
    
    # What to log
    log_process_creation: bool = True
    log_process_completion: bool = True
    log_process_migration: bool = True
    log_load_balancing_decisions: bool = True
    log_processor_state_changes: bool = True
    
    # Log format
    log_format: str = "%(asctime)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # Detail level
    verbose: bool = False  # If True, logs more detailed information


# =============================================================================
# DEFAULT INSTANCES
# =============================================================================

# Create default configuration instances for easy import
DEFAULT_SIMULATION_CONFIG = SimulationConfig()
DEFAULT_GUI_CONFIG = GUIConfig()
DEFAULT_LOGGING_CONFIG = LoggingConfig()


# =============================================================================
# CONSTANTS
# =============================================================================

# Version information
VERSION = "1.0.0"
APP_NAME = "Dynamic Load Balancing Simulator"
AUTHOR = "Student"

# Help text for algorithms
ALGORITHM_DESCRIPTIONS = {
    LoadBalancingAlgorithm.ROUND_ROBIN: """
Round Robin Load Balancing:
- Distributes processes to processors in a cyclic manner
- Simple and fair - each processor gets equal number of processes
- Does not consider actual processor load
- Best for: Homogeneous workloads with similar process sizes
    """.strip(),
    
    LoadBalancingAlgorithm.LEAST_LOADED: """
Least Loaded First Load Balancing:
- Assigns each new process to the processor with the lowest current load
- Considers queue length and current execution
- Better load distribution than Round Robin
- Best for: Variable workloads where processes have different burst times
    """.strip(),
    
    LoadBalancingAlgorithm.THRESHOLD_BASED: """
Threshold-Based Load Balancing:
- Monitors processor loads continuously
- Migrates processes when load difference exceeds threshold
- Includes hysteresis to prevent thrashing (constant migration)
- Best for: Dynamic workloads where load changes over time
    """.strip(),
    
    LoadBalancingAlgorithm.Q_LEARNING: """
AI (Q-Learning) Load Balancing:
- Uses reinforcement learning to learn optimal assignments
- Adapts to workload patterns through experience
- Balances exploration (trying new strategies) and exploitation (using learned knowledge)
- Training Mode: Actively learns and improves policy
- Exploitation Mode: Uses learned policy for best performance
- Best for: Complex, evolving workloads where patterns emerge over time
    """.strip(),
    
    LoadBalancingAlgorithm.DQN: """
AI (DQN - Deep Q-Network) Load Balancing:
- Uses deep neural networks for function approximation
- Handles continuous state spaces without discretization
- Features: Double DQN, Dueling Architecture, Prioritized Experience Replay
- Better generalization to unseen states than Q-Learning
- Training Mode: Learns through neural network optimization
- Evaluation Mode: Uses trained network for optimal decisions
- Best for: Large-scale systems with many processors and complex patterns
    """.strip()
}


if __name__ == "__main__":
    # Test configuration module
    print(f"=== {APP_NAME} v{VERSION} Configuration ===\n")
    
    config = SimulationConfig()
    print("Default Simulation Config:")
    print(f"  Processors: {config.num_processors}")
    print(f"  Processes: {config.num_processes}")
    print(f"  Time Quantum: {config.time_quantum}")
    print(f"  Default Algorithm: {config.default_algorithm.value}")
    
    print("\nValidating configuration...")
    try:
        config.validate()
        print("✓ Configuration is valid")
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
    
    print("\nGUI Config:")
    gui_config = GUIConfig()
    print(f"  Window Size: {gui_config.window_width}x{gui_config.window_height}")
    print(f"  Load Colors: Low={gui_config.color_low_load}, High={gui_config.color_high_load}")
