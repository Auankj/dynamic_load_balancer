"""
Load Balancer Module for Dynamic Load Balancing Simulator

This module implements various load balancing algorithms using the Strategy
design pattern. Each algorithm decides how to distribute processes among
processors to optimize system performance.

Load Balancing Algorithms:
1. Round Robin - Simple cyclic distribution
2. Least Loaded - Assign to processor with minimum load
3. Threshold Based - Dynamic migration when load imbalance exceeds threshold

OS Concepts:
- Load balancing is critical in multiprocessor systems for efficiency
- Trade-offs exist between balance quality and migration overhead
- Different workloads benefit from different algorithms

Design Pattern: Strategy Pattern
- LoadBalancer is the abstract base (Context)
- Concrete balancers implement specific algorithms (Strategies)
- Allows runtime algorithm selection and easy extensibility

Author: Student
Date: December 2024
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from collections import deque
import random

from config import (
    LoadBalancingAlgorithm,
    ProcessState,
    SimulationConfig,
    DEFAULT_SIMULATION_CONFIG
)
from process import Process
from processor import Processor, ProcessorManager


@dataclass
class MigrationRecord:
    """
    Record of a process migration event.
    
    Tracks migrations for analysis and visualization.
    """
    process_id: int
    source_processor: int
    destination_processor: int
    time: int
    reason: str
    source_load_before: float
    destination_load_before: float


class LoadBalancer(ABC):
    """
    Abstract base class for load balancing algorithms.
    
    Defines the interface that all load balancers must implement:
    - assign_process(): Initial assignment of new processes
    - check_for_migration(): Decide if processes should be migrated
    - get_migration_candidates(): Find processes eligible for migration
    
    The Strategy Pattern allows different algorithms to be swapped
    without changing the simulation engine code.
    """
    
    def __init__(self, config: SimulationConfig = None):
        """
        Initialize the load balancer.
        
        Args:
            config: Simulation configuration with algorithm parameters
        """
        self.config = config or DEFAULT_SIMULATION_CONFIG
        self.migration_history: List[MigrationRecord] = []
        self.assignment_count = 0
        self.migration_count = 0
    
    @property
    @abstractmethod
    def algorithm_type(self) -> LoadBalancingAlgorithm:
        """Return the algorithm type enum."""
        pass
    
    @property
    def name(self) -> str:
        """Return the algorithm name."""
        return self.algorithm_type.value
    
    @abstractmethod
    def assign_process(self, process: Process, processors: List[Processor]) -> Optional[Processor]:
        """
        Assign a new process to a processor.
        
        This is called when a new process arrives in the system.
        The algorithm decides which processor should receive it.
        
        Args:
            process: The process to assign
            processors: List of available processors
            
        Returns:
            The selected processor, or None if assignment failed
        """
        pass
    
    @abstractmethod
    def check_for_migration(self, processors: List[Processor], current_time: int) -> List[MigrationRecord]:
        """
        Check if any processes should be migrated.
        
        This is called periodically to rebalance the load.
        Returns a list of migration records to execute.
        
        Args:
            processors: List of all processors
            current_time: Current simulation time
            
        Returns:
            List of MigrationRecord objects describing migrations to perform
        """
        pass
    
    def get_migration_candidates(self, processor: Processor) -> List[Process]:
        """
        Get processes that can be migrated from a processor.
        
        Only processes in READY state (waiting in queue) can be migrated.
        The currently running process should not be migrated mid-execution.
        
        Args:
            processor: The processor to check
            
        Returns:
            List of migratable processes
        """
        candidates = []
        for process in processor.ready_queue:
            if process.state == ProcessState.READY:
                candidates.append(process)
        return candidates
    
    def execute_migration(self, migration: MigrationRecord, 
                          processors: List[Processor], 
                          current_time: int) -> bool:
        """
        Execute a process migration.
        
        Args:
            migration: The migration record describing the move
            processors: List of all processors
            current_time: Current simulation time
            
        Returns:
            True if migration was successful
        """
        source = next((p for p in processors if p.processor_id == migration.source_processor), None)
        dest = next((p for p in processors if p.processor_id == migration.destination_processor), None)
        
        if not source or not dest:
            return False
        
        # Find and remove the process from source
        process = None
        for p in source.ready_queue:
            if p.pid == migration.process_id:
                process = p
                break
        
        if not process:
            return False
        
        # Remove from source
        source.ready_queue.remove(process)
        source.statistics.processes_migrated_out += 1
        
        # Update process state
        process.state = ProcessState.MIGRATING
        process.migration_count += 1
        process.add_history_entry(current_time, "MIGRATED", migration.source_processor, migration.destination_processor)
        
        # Add to destination
        process.state = ProcessState.READY
        process.processor_id = dest.processor_id
        dest.add_process(process)
        dest.statistics.processes_migrated_in += 1
        
        # Record the migration
        self.migration_history.append(migration)
        self.migration_count += 1
        
        return True
    
    def get_load_statistics(self, processors: List[Processor]) -> Dict[str, float]:
        """
        Calculate load statistics across all processors.
        
        Args:
            processors: List of processors
            
        Returns:
            Dictionary with min, max, avg, std load values
        """
        if not processors:
            return {'min': 0, 'max': 0, 'avg': 0, 'std': 0, 'range': 0}
        
        loads = [p.get_load() for p in processors]
        avg_load = sum(loads) / len(loads)
        variance = sum((l - avg_load) ** 2 for l in loads) / len(loads)
        
        return {
            'min': min(loads),
            'max': max(loads),
            'avg': avg_load,
            'std': variance ** 0.5,
            'range': max(loads) - min(loads)
        }
    
    def reset(self):
        """Reset the balancer state."""
        self.migration_history.clear()
        self.assignment_count = 0
        self.migration_count = 0


class RoundRobinBalancer(LoadBalancer):
    """
    Round Robin Load Balancing Algorithm
    
    The simplest load balancing approach: assigns processes to processors
    in a cyclic order. Process 1 goes to P0, process 2 to P1, etc.
    
    Advantages:
    - Simple to implement
    - Fair distribution over time
    - Low overhead (O(1) assignment)
    
    Disadvantages:
    - Doesn't consider current load
    - Can create imbalance if processes have different sizes
    - No dynamic rebalancing
    
    Best for: Homogeneous workloads with similar process sizes
    """
    
    def __init__(self, config: SimulationConfig = None):
        """Initialize Round Robin balancer."""
        super().__init__(config)
        self._current_index = 0
    
    @property
    def algorithm_type(self) -> LoadBalancingAlgorithm:
        return LoadBalancingAlgorithm.ROUND_ROBIN
    
    def assign_process(self, process: Process, processors: List[Processor]) -> Optional[Processor]:
        """
        Assign process to the next processor in round-robin order.
        
        Uses modular arithmetic to cycle through processors:
        next_index = (current_index + 1) % num_processors
        
        Args:
            process: Process to assign
            processors: Available processors
            
        Returns:
            Selected processor
        """
        if not processors:
            return None
        
        # Select processor at current index
        selected = processors[self._current_index]
        
        # Move to next processor for next assignment
        self._current_index = (self._current_index + 1) % len(processors)
        
        # Add process to selected processor
        selected.add_process(process)
        process.processor_id = selected.processor_id
        self.assignment_count += 1
        
        return selected
    
    def check_for_migration(self, processors: List[Processor], current_time: int) -> List[MigrationRecord]:
        """
        Round Robin doesn't perform migrations.
        
        This is a static algorithm - it doesn't rebalance after initial assignment.
        
        Returns:
            Empty list (no migrations)
        """
        # Round Robin is a static algorithm - no migration
        return []
    
    def reset(self):
        """Reset balancer state."""
        super().reset()
        self._current_index = 0


class LeastLoadedBalancer(LoadBalancer):
    """
    Least Loaded (Minimum Load) Load Balancing Algorithm
    
    Always assigns new processes to the processor with the lowest current load.
    This is a greedy approach that considers current system state.
    
    Load Calculation:
    - Queue-based: Number of processes in ready queue
    - Work-based: Sum of remaining execution times
    
    Advantages:
    - Adapts to current load
    - Good for heterogeneous workloads
    - Better balance than Round Robin
    
    Disadvantages:
    - Higher overhead (O(n) to find minimum)
    - May cause "herd behavior" - all new processes go to same processor
    - Doesn't consider process migration
    
    Best for: Variable workloads, different process sizes
    """
    
    def __init__(self, config: SimulationConfig = None, use_work_based: bool = True):
        """
        Initialize Least Loaded balancer.
        
        Args:
            config: Simulation configuration
            use_work_based: If True, use remaining work; if False, use queue size
        """
        super().__init__(config)
        self.use_work_based = use_work_based
    
    @property
    def algorithm_type(self) -> LoadBalancingAlgorithm:
        return LoadBalancingAlgorithm.LEAST_LOADED
    
    def _get_processor_load(self, processor: Processor) -> float:
        """
        Get the load value for a processor.
        
        Args:
            processor: The processor to evaluate
            
        Returns:
            Load value (lower is better)
        """
        if self.use_work_based:
            return processor.get_load()  # Uses total remaining time
        else:
            return processor.get_queue_size()  # Just count processes
    
    def assign_process(self, process: Process, processors: List[Processor]) -> Optional[Processor]:
        """
        Assign process to the processor with minimum load.
        
        Algorithm:
        1. Calculate load for each processor
        2. Find processor with minimum load
        3. If tie, select first one (deterministic)
        
        Args:
            process: Process to assign
            processors: Available processors
            
        Returns:
            Processor with lowest load
        """
        if not processors:
            return None
        
        # Find processor with minimum load
        min_load = float('inf')
        selected = processors[0]
        
        for proc in processors:
            load = self._get_processor_load(proc)
            if load < min_load:
                min_load = load
                selected = proc
        
        # Add process to selected processor
        selected.add_process(process)
        process.processor_id = selected.processor_id
        self.assignment_count += 1
        
        return selected
    
    def check_for_migration(self, processors: List[Processor], current_time: int) -> List[MigrationRecord]:
        """
        Least Loaded can optionally migrate for extreme imbalances.
        
        Migration is triggered when:
        - Load difference exceeds a threshold
        - Both source and destination have migratable processes
        
        Returns:
            List of migration records (usually empty or single migration)
        """
        # Basic implementation: no migration for pure Least Loaded
        # Migration is handled by Threshold Based algorithm
        return []


class ThresholdBasedBalancer(LoadBalancer):
    """
    Threshold-Based Load Balancing Algorithm
    
    Combines initial assignment (like Least Loaded) with dynamic migration
    when load imbalance exceeds a configurable threshold.
    
    Key Concepts:
    - High Threshold: Processor is overloaded if load > this
    - Low Threshold: Processor is underloaded if load < this
    - Migration occurs from high to low processors
    
    Migration Policy:
    - Sender-Initiated: Overloaded processors push work
    - Uses threshold difference to prevent oscillation
    
    Advantages:
    - Dynamic rebalancing
    - Prevents extreme imbalance
    - Configurable sensitivity
    
    Disadvantages:
    - Migration has overhead
    - Can cause thrashing if thresholds wrong
    - More complex implementation
    
    Best for: Dynamic workloads, real-time systems
    """
    
    def __init__(self, config: SimulationConfig = None,
                 high_threshold: float = None,
                 low_threshold: float = None,
                 migration_cooldown: int = 5):
        """
        Initialize Threshold Based balancer.
        
        Args:
            config: Simulation configuration
            high_threshold: Load level considered "overloaded"
            low_threshold: Load level considered "underloaded"
            migration_cooldown: Minimum time between migrations per processor
        """
        super().__init__(config)
        
        self.high_threshold = high_threshold or (config.load_threshold if config else 0.8)
        self.low_threshold = low_threshold or (self.high_threshold * 0.5)
        self.migration_cooldown = migration_cooldown
        
        # Track last migration time per processor to prevent thrashing
        self._last_migration: Dict[int, int] = {}
    
    @property
    def algorithm_type(self) -> LoadBalancingAlgorithm:
        return LoadBalancingAlgorithm.THRESHOLD_BASED
    
    def assign_process(self, process: Process, processors: List[Processor]) -> Optional[Processor]:
        """
        Assign process using Least Loaded strategy.
        
        Initial assignment uses Least Loaded for good starting balance.
        Rebalancing is handled by check_for_migration().
        
        Args:
            process: Process to assign
            processors: Available processors
            
        Returns:
            Processor with lowest load
        """
        if not processors:
            return None
        
        # Use Least Loaded for initial assignment
        min_load = float('inf')
        selected = processors[0]
        
        for proc in processors:
            load = proc.get_load()
            if load < min_load:
                min_load = load
                selected = proc
        
        selected.add_process(process)
        process.processor_id = selected.processor_id
        self.assignment_count += 1
        
        return selected
    
    def _get_normalized_load(self, processor: Processor, max_load: float) -> float:
        """
        Get normalized load (0.0 to 1.0) for threshold comparison.
        
        Args:
            processor: The processor
            max_load: Maximum load across all processors for normalization
            
        Returns:
            Normalized load value
        """
        if max_load <= 0:
            return 0.0
        return processor.get_load() / max_load
    
    def _can_migrate(self, processor_id: int, current_time: int) -> bool:
        """
        Check if processor is allowed to participate in migration.
        
        Cooldown prevents thrashing (repeated migrations back and forth).
        
        Args:
            processor_id: ID of the processor
            current_time: Current simulation time
            
        Returns:
            True if migration is allowed
        """
        last_time = self._last_migration.get(processor_id, -self.migration_cooldown)
        return (current_time - last_time) >= self.migration_cooldown
    
    def check_for_migration(self, processors: List[Processor], current_time: int) -> List[MigrationRecord]:
        """
        Check for load imbalance and create migration plan.
        
        Algorithm (Sender-Initiated):
        1. Find overloaded processors (load > high_threshold)
        2. Find underloaded processors (load < low_threshold)
        3. For each overloaded processor:
           a. Get migration candidates
           b. Select best underloaded destination
           c. Create migration record
        4. Return list of migrations to execute
        
        Args:
            processors: All processors
            current_time: Current simulation time
            
        Returns:
            List of migration records
        """
        migrations = []
        
        if len(processors) < 2:
            return migrations
        
        # Calculate loads and find max for normalization
        loads = [(p, p.get_load()) for p in processors]
        max_load = max(l for _, l in loads) if loads else 0
        
        if max_load <= 0:
            return migrations
        
        # Categorize processors
        overloaded = []
        underloaded = []
        
        for proc, load in loads:
            norm_load = load / max_load if max_load > 0 else 0
            
            if norm_load > self.high_threshold and self._can_migrate(proc.processor_id, current_time):
                overloaded.append((proc, load))
            elif norm_load < self.low_threshold and self._can_migrate(proc.processor_id, current_time):
                underloaded.append((proc, load))
        
        # Sort: most overloaded first, least loaded first
        overloaded.sort(key=lambda x: x[1], reverse=True)
        underloaded.sort(key=lambda x: x[1])
        
        # Create migration plan
        for source_proc, source_load in overloaded:
            if not underloaded:
                break
            
            candidates = self.get_migration_candidates(source_proc)
            if not candidates:
                continue
            
            # Select process to migrate (smallest first to minimize disruption)
            candidates.sort(key=lambda p: p.remaining_time)
            process_to_migrate = candidates[0]
            
            # Find best destination
            dest_proc, dest_load = underloaded[0]
            
            # Only migrate if it improves balance
            load_diff = source_load - dest_load
            if load_diff <= 0:
                continue
            
            # Create migration record
            migration = MigrationRecord(
                process_id=process_to_migrate.pid,
                source_processor=source_proc.processor_id,
                destination_processor=dest_proc.processor_id,
                time=current_time,
                reason=f"Load imbalance: {source_load:.1f} vs {dest_load:.1f}",
                source_load_before=source_load,
                destination_load_before=dest_load
            )
            migrations.append(migration)
            
            # Update last migration time
            self._last_migration[source_proc.processor_id] = current_time
            self._last_migration[dest_proc.processor_id] = current_time
            
            # Update simulated loads for next iteration
            underloaded[0] = (dest_proc, dest_load + process_to_migrate.remaining_time)
            underloaded.sort(key=lambda x: x[1])
        
        return migrations
    
    def reset(self):
        """Reset balancer state."""
        super().reset()
        self._last_migration.clear()


class LoadBalancerFactory:
    """
    Factory class for creating load balancer instances.
    
    Uses the Factory Pattern to create appropriate balancer
    based on the algorithm type enum.
    """
    
    # Lazy import for AI balancers to avoid circular imports
    _qlearning_balancer_class = None
    _dqn_balancer_class = None
    
    @classmethod
    def _get_qlearning_class(cls):
        """Lazy load Q-Learning balancer to avoid circular imports."""
        if cls._qlearning_balancer_class is None:
            from ai_balancer import QLearningBalancer
            cls._qlearning_balancer_class = QLearningBalancer
        return cls._qlearning_balancer_class
    
    @classmethod
    def _get_dqn_class(cls):
        """Lazy load DQN balancer to avoid circular imports."""
        if cls._dqn_balancer_class is None:
            from dqn_balancer import DQNBalancer
            cls._dqn_balancer_class = DQNBalancer
        return cls._dqn_balancer_class
    
    @staticmethod
    def create(algorithm: LoadBalancingAlgorithm, 
               config: SimulationConfig = None,
               num_processors: int = 4) -> LoadBalancer:
        """
        Create a load balancer instance.
        
        Args:
            algorithm: The algorithm type to create
            config: Optional configuration
            num_processors: Number of processors (for AI algorithms)
            
        Returns:
            LoadBalancer instance
            
        Raises:
            ValueError: If algorithm is not supported
        """
        if algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
            return RoundRobinBalancer(config)
        elif algorithm == LoadBalancingAlgorithm.LEAST_LOADED:
            return LeastLoadedBalancer(config)
        elif algorithm == LoadBalancingAlgorithm.THRESHOLD_BASED:
            return ThresholdBasedBalancer(config)
        elif algorithm == LoadBalancingAlgorithm.Q_LEARNING:
            QLearningBalancer = LoadBalancerFactory._get_qlearning_class()
            return QLearningBalancer(config=config, num_processors=num_processors)
        elif algorithm == LoadBalancingAlgorithm.DQN:
            DQNBalancer = LoadBalancerFactory._get_dqn_class()
            return DQNBalancer(config=config, num_processors=num_processors)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    @staticmethod
    def get_all_algorithms() -> List[LoadBalancingAlgorithm]:
        """Get list of all supported algorithms."""
        return list(LoadBalancingAlgorithm)
    
    @staticmethod
    def get_algorithm_descriptions() -> Dict[str, str]:
        """Get descriptions of all algorithms."""
        return {
            LoadBalancingAlgorithm.ROUND_ROBIN.value: 
                "Simple cyclic distribution - fair but doesn't consider load",
            LoadBalancingAlgorithm.LEAST_LOADED.value: 
                "Assigns to processor with minimum load - adaptive but no migration",
            LoadBalancingAlgorithm.THRESHOLD_BASED.value: 
                "Dynamic migration when imbalance exceeds threshold - most sophisticated",
            LoadBalancingAlgorithm.Q_LEARNING.value:
                "AI-powered adaptive balancing using reinforcement learning",
            LoadBalancingAlgorithm.DQN.value:
                "Deep Q-Network with neural network function approximation"
        }


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Load Balancer Module Test")
    print("=" * 70)
    
    from process import ProcessGenerator, Process
    from processor import ProcessorManager
    
    # Create test setup
    generator = ProcessGenerator()
    processes = generator.generate_predefined_test_set()
    manager = ProcessorManager(num_processors=4)
    processors = list(manager)
    
    print("\n" + "-" * 70)
    print("1. Testing Round Robin Balancer")
    print("-" * 70)
    
    rr_balancer = RoundRobinBalancer()
    print(f"Algorithm: {rr_balancer.name}")
    
    # Assign all processes
    for p in processes:
        selected = rr_balancer.assign_process(p, processors)
        print(f"   Process {p.pid} -> Processor {selected.processor_id}")
    
    print(f"\n   Processor loads after assignment:")
    for proc in processors:
        print(f"   - Processor {proc.processor_id}: queue={proc.get_queue_size()}, load={proc.get_load():.1f}")
    
    # Check for migration (should be empty)
    migrations = rr_balancer.check_for_migration(processors, current_time=0)
    print(f"\n   Migrations suggested: {len(migrations)}")
    
    # Reset for next test
    for proc in processors:
        proc.ready_queue.clear()
    
    print("\n" + "-" * 70)
    print("2. Testing Least Loaded Balancer")
    print("-" * 70)
    
    ll_balancer = LeastLoadedBalancer()
    print(f"Algorithm: {ll_balancer.name}")
    
    # Reset processes
    for p in processes:
        p.processor_id = None
    
    # Assign all processes
    for p in processes:
        selected = ll_balancer.assign_process(p, processors)
        print(f"   Process {p.pid} (burst={p.burst_time}) -> Processor {selected.processor_id} (load={selected.get_load():.1f})")
    
    print(f"\n   Processor loads after assignment:")
    for proc in processors:
        print(f"   - Processor {proc.processor_id}: queue={proc.get_queue_size()}, load={proc.get_load():.1f}")
    
    # Reset for next test
    for proc in processors:
        proc.ready_queue.clear()
    
    print("\n" + "-" * 70)
    print("3. Testing Threshold Based Balancer")
    print("-" * 70)
    
    tb_balancer = ThresholdBasedBalancer(high_threshold=0.7, low_threshold=0.3)
    print(f"Algorithm: {tb_balancer.name}")
    print(f"   High threshold: {tb_balancer.high_threshold}")
    print(f"   Low threshold: {tb_balancer.low_threshold}")
    
    # Reset processes
    for p in processes:
        p.processor_id = None
        p.state = ProcessState.NEW
    
    # Create imbalanced load manually for testing migration
    for i, p in enumerate(processes):
        if i < 6:  # First 6 processes to processor 0
            processors[0].add_process(p)
            p.processor_id = 0
            p.state = ProcessState.READY
        elif i < 8:  # Next 2 to processor 1
            processors[1].add_process(p)
            p.processor_id = 1
            p.state = ProcessState.READY
        # Processor 2 and 3 get nothing
    
    print(f"\n   Initial loads (imbalanced):")
    for proc in processors:
        print(f"   - Processor {proc.processor_id}: queue={proc.get_queue_size()}, load={proc.get_load():.1f}")
    
    # Check for migration
    migrations = tb_balancer.check_for_migration(processors, current_time=0)
    print(f"\n   Migrations suggested: {len(migrations)}")
    for m in migrations:
        print(f"   - Process {m.process_id}: P{m.source_processor} -> P{m.destination_processor}")
        print(f"     Reason: {m.reason}")
    
    # Execute migrations
    for migration in migrations:
        success = tb_balancer.execute_migration(migration, processors, current_time=0)
        print(f"   - Migration executed: {success}")
    
    print(f"\n   Loads after migration:")
    for proc in processors:
        print(f"   - Processor {proc.processor_id}: queue={proc.get_queue_size()}, load={proc.get_load():.1f}")
    
    print("\n" + "-" * 70)
    print("4. Testing LoadBalancerFactory")
    print("-" * 70)
    
    for algo in LoadBalancerFactory.get_all_algorithms():
        balancer = LoadBalancerFactory.create(algo)
        print(f"   Created: {balancer.name} ({balancer.__class__.__name__})")
    
    print("\n   Algorithm Descriptions:")
    for name, desc in LoadBalancerFactory.get_algorithm_descriptions().items():
        print(f"   - {name}: {desc}")
    
    print("\n" + "=" * 70)
    print("All load balancer tests completed successfully!")
    print("=" * 70)
