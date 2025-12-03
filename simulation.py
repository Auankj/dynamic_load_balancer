"""
Simulation Engine Module for Dynamic Load Balancing Simulator

This module provides the core simulation engine that orchestrates all
components: processes, processors, load balancers, and metrics collection.

The simulation follows a discrete event model:
1. Initialize system with processors and load balancer
2. At each time step:
   - Arrival: New processes arrive
   - Assignment: Load balancer assigns new processes
   - Execution: Each processor executes one time unit
   - Migration: Load balancer checks for migrations
   - Metrics: Record statistics
3. Continue until all processes complete or max time reached

OS Concepts:
- Discrete Event Simulation models OS behavior step by step
- Time slicing in CPU scheduling
- Process state transitions
- Load monitoring and balancing decisions

Author: Student
Date: December 2024
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
from enum import Enum
from collections import deque
import time
import threading
import copy

from config import (
    ProcessState,
    LoadBalancingAlgorithm,
    ProcessPriority,
    SimulationConfig,
    GUIConfig,
    DEFAULT_SIMULATION_CONFIG,
    DEFAULT_GUI_CONFIG
)
from process import Process, ProcessGenerator
from processor import Processor, ProcessorManager
from load_balancer import (
    LoadBalancer,
    LoadBalancerFactory,
    MigrationRecord,
    RoundRobinBalancer,
    LeastLoadedBalancer,
    ThresholdBasedBalancer
)
from metrics import (
    MetricsCalculator,
    MetricsComparator,
    ProcessMetrics,
    ProcessorMetrics,
    SystemMetrics
)


class SimulationState(Enum):
    """
    States of the simulation engine.
    
    State Transitions:
    IDLE -> RUNNING (start)
    RUNNING -> PAUSED (pause)
    PAUSED -> RUNNING (resume)
    RUNNING -> COMPLETED (all done)
    RUNNING -> STOPPED (user stop)
    PAUSED -> STOPPED (user stop)
    """
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    STOPPED = "stopped"


@dataclass
class SimulationSnapshot:
    """
    Snapshot of simulation state at a point in time.
    
    Used for visualization and debugging.
    """
    time: int
    processor_loads: List[float]
    processor_queues: List[int]
    active_processes: List[int]  # PIDs of running processes
    completed_count: int
    total_processes: int
    migrations_this_step: int


@dataclass
class SimulationResult:
    """
    Complete results of a simulation run.
    
    Contains all data needed for analysis and comparison.
    """
    algorithm: LoadBalancingAlgorithm
    config: SimulationConfig
    system_metrics: SystemMetrics
    process_metrics: List[ProcessMetrics]
    processor_metrics: List[ProcessorMetrics]
    snapshots: List[SimulationSnapshot]
    total_time: int
    execution_duration: float  # Real wall-clock time
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            'algorithm': self.algorithm.value,
            'total_time': self.total_time,
            'execution_duration': round(self.execution_duration, 3),
            'system_metrics': self.system_metrics.to_dict() if self.system_metrics else {},
            'process_count': len(self.process_metrics),
            'processor_count': len(self.processor_metrics)
        }


class SimulationEngine:
    """
    Core simulation engine for load balancing simulation.
    
    This class manages the entire simulation lifecycle:
    - Initialization with configuration
    - Process generation and arrival
    - Load balancing decisions
    - Processor execution
    - Metrics collection
    - State management
    
    Usage:
        engine = SimulationEngine(config)
        engine.initialize()
        result = engine.run()
        # or for step-by-step:
        engine.initialize()
        while not engine.is_complete():
            engine.step()
        result = engine.get_result()
    
    Thread Safety:
        The engine uses locks to allow safe GUI updates from another thread.
    """
    
    def __init__(self, config: SimulationConfig = None, 
                 gui_config: GUIConfig = None):
        """
        Initialize the simulation engine.
        
        Args:
            config: Simulation configuration
            gui_config: GUI-specific configuration
        """
        self.config = config or DEFAULT_SIMULATION_CONFIG
        self.gui_config = gui_config or DEFAULT_GUI_CONFIG
        
        # Core components
        self.processor_manager: Optional[ProcessorManager] = None
        self.load_balancer: Optional[LoadBalancer] = None
        self.metrics_calculator: Optional[MetricsCalculator] = None
        self.process_generator: Optional[ProcessGenerator] = None
        
        # Process tracking
        self.all_processes: List[Process] = []
        self.pending_processes: deque = deque()  # Processes not yet arrived
        self.active_processes: List[Process] = []  # In system but not completed
        self.completed_processes: List[Process] = []
        
        # Simulation state
        self.state = SimulationState.IDLE
        self.current_time = 0
        self.max_time = 0
        
        # History for visualization
        self.snapshots: List[SimulationSnapshot] = []
        
        # Callbacks for GUI updates
        self._on_step_callback: Optional[Callable] = None
        self._on_complete_callback: Optional[Callable] = None
        self._on_process_complete_callback: Optional[Callable] = None
        
        # Thread safety
        self._lock = threading.Lock()
        self._stop_requested = False
        
        # Performance tracking
        self._start_wall_time: float = 0
        self._end_wall_time: float = 0
    
    def set_callbacks(self, 
                      on_step: Callable = None,
                      on_complete: Callable = None,
                      on_process_complete: Callable = None):
        """
        Set callback functions for events.
        
        Args:
            on_step: Called after each simulation step
            on_complete: Called when simulation completes
            on_process_complete: Called when a process completes
        """
        self._on_step_callback = on_step
        self._on_complete_callback = on_complete
        self._on_process_complete_callback = on_process_complete
    
    def initialize(self, algorithm: LoadBalancingAlgorithm = None,
                   processes: List[Process] = None) -> bool:
        """
        Initialize the simulation with all components.
        
        Args:
            algorithm: Load balancing algorithm to use (defaults to config)
            processes: Pre-generated processes (generates if None)
            
        Returns:
            True if initialization successful
        """
        with self._lock:
            try:
                # Reset state
                self._reset_state()
                
                # Create processor manager
                self.processor_manager = ProcessorManager(
                    num_processors=self.config.num_processors,
                    config=self.config
                )
                
                # Create load balancer
                algo = algorithm or self.config.default_algorithm
                self.load_balancer = LoadBalancerFactory.create(
                    algo, self.config, 
                    num_processors=self.config.num_processors
                )
                
                # Create metrics calculator
                self.metrics_calculator = MetricsCalculator(algo)
                
                # Generate or use provided processes
                if processes:
                    self.all_processes = list(processes)
                else:
                    self.process_generator = ProcessGenerator(config=self.config)
                    self.all_processes = self.process_generator.generate_processes(
                        self.config.num_processes
                    )
                
                # Reset all processes to initial state
                for p in self.all_processes:
                    p.state = ProcessState.NEW
                    p.processor_id = None
                    p.start_time = None
                    p.completion_time = None
                    p.waiting_time = 0
                    p.migration_count = 0
                    p.execution_history.clear()
                
                # Sort by arrival time and add to pending queue
                self.all_processes.sort(key=lambda p: p.arrival_time)
                self.pending_processes = deque(self.all_processes)
                self.active_processes = []
                self.completed_processes = []
                
                # Calculate max simulation time
                max_arrival = max(p.arrival_time for p in self.all_processes)
                max_burst = sum(p.burst_time for p in self.all_processes)
                self.max_time = max_arrival + max_burst + self.config.num_processors * 10
                
                self.state = SimulationState.IDLE
                self.current_time = 0
                self._stop_requested = False
                
                return True
                
            except Exception as e:
                print(f"Initialization error: {e}")
                return False
    
    def _reset_state(self):
        """Reset all simulation state."""
        self.all_processes.clear()
        self.pending_processes.clear()
        self.active_processes.clear()
        self.completed_processes.clear()
        self.snapshots.clear()
        self.current_time = 0
        self.state = SimulationState.IDLE
        self._stop_requested = False
        
        if self.processor_manager:
            self.processor_manager.reset_all()
        if self.load_balancer:
            self.load_balancer.reset()
        if self.metrics_calculator:
            self.metrics_calculator.reset()
    
    def step(self) -> bool:
        """
        Execute one simulation time step.
        
        This is the main simulation loop body:
        1. Process arrivals
        2. Execute time slice on each processor
        3. Check for migrations
        4. Record metrics
        5. Check for completion
        
        Returns:
            True if simulation should continue, False if done
        """
        if self.state not in (SimulationState.RUNNING, SimulationState.IDLE):
            return False
        
        if self.state == SimulationState.IDLE:
            self.state = SimulationState.RUNNING
            self._start_wall_time = time.time()
        
        with self._lock:
            if self._stop_requested:
                self.state = SimulationState.STOPPED
                return False
            
            # Step 1: Process arrivals
            self._handle_arrivals()
            
            # Step 2: Execute on all processors
            completed_this_step = self._execute_time_step()
            
            # Step 3: Check for migrations
            migrations_this_step = self._handle_migrations()
            
            # Step 4: Record metrics snapshot
            self._record_snapshot(migrations_this_step)
            
            # Step 5: Advance time
            self.current_time += 1
            
            # Step 6: Check completion
            if self._is_simulation_complete():
                self.state = SimulationState.COMPLETED
                self._end_wall_time = time.time()
                self._finalize_metrics()
                if self._on_complete_callback:
                    self._on_complete_callback()
                return False
            
            # Safety check for max time
            if self.current_time >= self.max_time:
                self.state = SimulationState.COMPLETED
                self._end_wall_time = time.time()
                self._finalize_metrics()
                return False
            
            # Invoke step callback
            if self._on_step_callback:
                self._on_step_callback(self.current_time)
            
            return True
    
    def _handle_arrivals(self):
        """
        Handle process arrivals at current time.
        
        Processes whose arrival_time equals current_time are:
        1. Removed from pending queue
        2. Assigned to a processor by load balancer
        3. Added to active processes list
        """
        while self.pending_processes and self.pending_processes[0].arrival_time <= self.current_time:
            process = self.pending_processes.popleft()
            
            # Use load balancer to assign
            processors = list(self.processor_manager)
            selected = self.load_balancer.assign_process(process, processors)
            
            if selected:
                process.state = ProcessState.READY
                self.active_processes.append(process)
            else:
                # This shouldn't happen with proper implementation
                print(f"Warning: Could not assign process {process.pid}")
    
    def _execute_time_step(self) -> List[Process]:
        """
        Execute one time unit on all processors.
        
        Returns:
            List of processes completed this step
        """
        completed = []
        
        for processor in self.processor_manager:
            result = processor.execute_time_slice(self.current_time)
            
            # Check if a process was completed (result is a dict)
            if result and result.get('completed', False):
                process = result.get('process')
                if process and process.state == ProcessState.COMPLETED:
                    completed.append(process)
                    
                    # Move from active to completed
                    if process in self.active_processes:
                        self.active_processes.remove(process)
                    self.completed_processes.append(process)
                    
                    # Provide feedback to AI load balancer if applicable
                    if hasattr(self.load_balancer, 'process_completed'):
                        processors = list(self.processor_manager)
                        self.load_balancer.process_completed(process, processors)
                    
                    # Callback
                    if self._on_process_complete_callback:
                        self._on_process_complete_callback(process)
        
        return completed
    
    def _handle_migrations(self) -> int:
        """
        Check and execute process migrations.
        
        Returns:
            Number of migrations performed
        """
        processors = list(self.processor_manager)
        migrations = self.load_balancer.check_for_migration(processors, self.current_time)
        
        executed = 0
        for migration in migrations:
            if self.load_balancer.execute_migration(migration, processors, self.current_time):
                executed += 1
        
        return executed
    
    def _record_snapshot(self, migrations: int):
        """
        Record current state snapshot.
        
        Args:
            migrations: Number of migrations this step
        """
        processors = list(self.processor_manager)
        
        snapshot = SimulationSnapshot(
            time=self.current_time,
            processor_loads=[p.get_load() for p in processors],
            processor_queues=[p.get_queue_size() for p in processors],
            active_processes=[p.current_process.pid if p.current_process else -1 for p in processors],
            completed_count=len(self.completed_processes),
            total_processes=len(self.all_processes),
            migrations_this_step=migrations
        )
        self.snapshots.append(snapshot)
        
        # Also record for metrics time series
        self.metrics_calculator.record_time_point(processors, len(self.completed_processes))
    
    def _is_simulation_complete(self) -> bool:
        """
        Check if simulation is complete.
        
        Complete when:
        - All processes have completed, OR
        - Max time reached
        """
        return (len(self.completed_processes) >= len(self.all_processes) or 
                self.current_time >= self.max_time)
    
    def _finalize_metrics(self):
        """Finalize and calculate all metrics."""
        processors = list(self.processor_manager)
        
        self.metrics_calculator.collect_process_metrics(self.all_processes)
        self.metrics_calculator.collect_processor_metrics(processors, self.current_time)
        self.metrics_calculator.calculate_system_metrics()
    
    def run(self, with_delay: bool = False) -> SimulationResult:
        """
        Run complete simulation.
        
        Args:
            with_delay: If True, add small delay between steps (for visualization)
            
        Returns:
            SimulationResult containing all data
        """
        self.state = SimulationState.RUNNING
        self._start_wall_time = time.time()
        
        while self.step():
            if with_delay:
                time.sleep(self.gui_config.animation_speed / 1000.0)
        
        return self.get_result()
    
    def run_async(self, callback: Callable = None):
        """
        Run simulation in background thread.
        
        Args:
            callback: Function to call when complete
        """
        def _run():
            result = self.run()
            if callback:
                callback(result)
        
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return thread
    
    def pause(self):
        """Pause the simulation."""
        if self.state == SimulationState.RUNNING:
            self.state = SimulationState.PAUSED
    
    def resume(self):
        """Resume paused simulation."""
        if self.state == SimulationState.PAUSED:
            self.state = SimulationState.RUNNING
    
    def stop(self):
        """Stop the simulation."""
        self._stop_requested = True
        self.state = SimulationState.STOPPED
    
    def is_complete(self) -> bool:
        """Check if simulation is complete."""
        return self.state in (SimulationState.COMPLETED, SimulationState.STOPPED)
    
    def is_running(self) -> bool:
        """Check if simulation is running."""
        return self.state == SimulationState.RUNNING
    
    def get_result(self) -> SimulationResult:
        """
        Get simulation results.
        
        Returns:
            SimulationResult with all data
        """
        if self._end_wall_time == 0:
            self._end_wall_time = time.time()
        
        return SimulationResult(
            algorithm=self.load_balancer.algorithm_type if self.load_balancer else LoadBalancingAlgorithm.ROUND_ROBIN,
            config=self.config,
            system_metrics=self.metrics_calculator.system_metrics if self.metrics_calculator else SystemMetrics(),
            process_metrics=self.metrics_calculator.process_metrics if self.metrics_calculator else [],
            processor_metrics=self.metrics_calculator.processor_metrics if self.metrics_calculator else [],
            snapshots=self.snapshots.copy(),
            total_time=self.current_time,
            execution_duration=self._end_wall_time - self._start_wall_time
        )
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current simulation state for display.
        
        Returns:
            Dictionary with current state information
        """
        with self._lock:
            processors = list(self.processor_manager) if self.processor_manager else []
            
            return {
                'time': self.current_time,
                'state': self.state.value,
                'total_processes': len(self.all_processes),
                'pending': len(self.pending_processes),
                'active': len(self.active_processes),
                'completed': len(self.completed_processes),
                'processors': [
                    {
                        'id': p.processor_id,
                        'load': p.get_load(),
                        'queue_size': p.get_queue_size(),
                        'current_process': p.current_process.pid if p.current_process else None,
                        'utilization': p.statistics.total_execution_time / max(1, self.current_time)
                    }
                    for p in processors
                ],
                'algorithm': self.load_balancer.name if self.load_balancer else 'None',
                'migrations': self.load_balancer.migration_count if self.load_balancer else 0
            }


class BatchSimulator:
    """
    Run multiple simulations for algorithm comparison.
    
    This class helps compare different load balancing algorithms
    by running the same workload with each algorithm.
    """
    
    def __init__(self, config: SimulationConfig = None):
        """
        Initialize batch simulator.
        
        Args:
            config: Base configuration for simulations
        """
        self.config = config or DEFAULT_SIMULATION_CONFIG
        self.results: Dict[LoadBalancingAlgorithm, SimulationResult] = {}
        self.comparator = MetricsComparator()
    
    def run_comparison(self, algorithms: List[LoadBalancingAlgorithm] = None,
                       processes: List[Process] = None) -> Dict[str, SimulationResult]:
        """
        Run simulation with multiple algorithms.
        
        Args:
            algorithms: List of algorithms to compare (defaults to all)
            processes: Processes to use (generates if None)
            
        Returns:
            Dictionary mapping algorithm names to results
        """
        if algorithms is None:
            algorithms = list(LoadBalancingAlgorithm)
        
        # Generate processes once for fair comparison
        if processes is None:
            generator = ProcessGenerator(config=self.config)
            base_processes = generator.generate_processes(self.config.num_processes)
        else:
            base_processes = processes
        
        self.results.clear()
        self.comparator.clear()
        
        for algo in algorithms:
            # Deep copy processes for each run
            test_processes = [
                Process(
                    pid=p.pid,
                    arrival_time=p.arrival_time,
                    burst_time=p.burst_time,
                    priority=p.priority
                )
                for p in base_processes
            ]
            
            # Run simulation
            engine = SimulationEngine(self.config)
            engine.initialize(algorithm=algo, processes=test_processes)
            result = engine.run()
            
            self.results[algo] = result
            self.comparator.add_result(algo, result.system_metrics)
        
        return {algo.value: result for algo, result in self.results.items()}
    
    def get_comparison_report(self) -> str:
        """Generate comparison report."""
        return self.comparator.generate_report()
    
    def get_best_algorithm(self, metric: str = 'avg_turnaround_time') -> str:
        """Get best algorithm for a metric."""
        return self.comparator.get_best_algorithm(metric)


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Simulation Engine Module Test")
    print("=" * 70)
    
    # Create configuration
    config = SimulationConfig(
        num_processors=4,
        num_processes=15,
        time_quantum=3,
        min_burst_time=3,
        max_burst_time=12,
        min_arrival_time=0,
        max_arrival_time=10
    )
    
    print(f"\nConfiguration:")
    print(f"  Processors: {config.num_processors}")
    print(f"  Processes: {config.num_processes}")
    print(f"  Time Quantum: {config.time_quantum}")
    
    print("\n" + "-" * 70)
    print("1. Testing Single Simulation Run")
    print("-" * 70)
    
    engine = SimulationEngine(config)
    engine.initialize(algorithm=LoadBalancingAlgorithm.LEAST_LOADED)
    
    print(f"\nInitial state:")
    state = engine.get_current_state()
    print(f"  Time: {state['time']}")
    print(f"  Pending: {state['pending']}")
    print(f"  Algorithm: {state['algorithm']}")
    
    # Run simulation
    print("\nRunning simulation...")
    result = engine.run()
    
    print(f"\nSimulation complete:")
    print(f"  Total time: {result.total_time} time units")
    print(f"  Execution duration: {result.execution_duration:.3f} seconds")
    print(f"  Completed: {result.system_metrics.completed_processes}/{result.system_metrics.total_processes}")
    
    print(f"\nSystem Metrics:")
    print(f"  Avg Turnaround: {result.system_metrics.avg_turnaround_time:.2f}")
    print(f"  Avg Waiting: {result.system_metrics.avg_waiting_time:.2f}")
    print(f"  Avg Utilization: {result.system_metrics.avg_utilization*100:.1f}%")
    print(f"  Load Balance Index: {result.system_metrics.load_balance_index:.4f}")
    print(f"  Jain's Fairness: {result.system_metrics.jains_fairness_index:.4f}")
    
    print("\n" + "-" * 70)
    print("2. Testing Step-by-Step Execution")
    print("-" * 70)
    
    engine2 = SimulationEngine(config)
    engine2.initialize(algorithm=LoadBalancingAlgorithm.ROUND_ROBIN)
    
    print("\nFirst 5 steps:")
    for i in range(5):
        continue_sim = engine2.step()
        state = engine2.get_current_state()
        print(f"  Step {state['time']}: pending={state['pending']}, active={state['active']}, completed={state['completed']}")
        
        if not continue_sim:
            break
    
    # Complete the rest
    while engine2.step():
        pass
    
    result2 = engine2.get_result()
    print(f"\nRound Robin completed in {result2.total_time} time units")
    
    print("\n" + "-" * 70)
    print("3. Testing Algorithm Comparison")
    print("-" * 70)
    
    batch = BatchSimulator(config)
    results = batch.run_comparison()
    
    print("\nComparison Results:")
    for algo_name, result in results.items():
        metrics = result.system_metrics
        print(f"\n  {algo_name}:")
        print(f"    Time: {result.total_time}")
        print(f"    Avg Turnaround: {metrics.avg_turnaround_time:.2f}")
        print(f"    Avg Waiting: {metrics.avg_waiting_time:.2f}")
        print(f"    Migrations: {metrics.total_migrations}")
    
    print(f"\n  Best for turnaround: {batch.get_best_algorithm('avg_turnaround_time')}")
    print(f"  Best for fairness: {batch.get_best_algorithm('jains_fairness_index')}")
    
    print("\n" + "=" * 70)
    print("All simulation tests completed successfully!")
    print("=" * 70)
