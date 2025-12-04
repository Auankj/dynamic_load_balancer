"""
Enhanced Simulation Engine for Dynamic Load Balancing Simulator

This module provides the production-grade simulation engine with:
- Advanced process and processor models
- Multiple scheduling policies
- Real-time metrics and visualization data
- Comprehensive event logging
- Performance analytics

Author: Student  
Date: December 2024
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
from enum import Enum
from collections import deque
import time
import threading
import random
import math
import statistics

from config import (
    ProcessState,
    LoadBalancingAlgorithm,
    ProcessPriority,
    SimulationConfig,
    GUIConfig,
    DEFAULT_SIMULATION_CONFIG,
    DEFAULT_GUI_CONFIG
)
from advanced_simulation import (
    AdvancedProcess,
    AdvancedProcessor,
    ProcessType,
    ProcessorState,
    SchedulingPolicy,
    WorkloadPattern,
    AdvancedWorkloadGenerator,
    AdvancedMetrics,
    AdvancedMetricsCalculator,
    ProcessorCapabilities,
    ThermalState,
    ProcessDeadline
)
from load_balancer import LoadBalancerFactory


class EnhancedSimulationState(Enum):
    """Enhanced simulation states."""
    IDLE = "idle"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STEPPING = "stepping"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class SimulationEvent:
    """
    Represents a discrete event in the simulation.
    """
    time: int
    event_type: str
    process_id: Optional[int] = None
    processor_id: Optional[int] = None
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'time': self.time,
            'type': self.event_type,
            'pid': self.process_id,
            'cpu': self.processor_id,
            'details': self.details
        }


@dataclass 
class EnhancedSnapshot:
    """
    Comprehensive snapshot of simulation state.
    """
    time: int
    
    # Processor states
    processor_loads: List[float]
    processor_queues: List[int]
    processor_temps: List[float]
    processor_powers: List[float]
    processor_states: List[str]
    current_processes: List[Optional[int]]
    
    # Process states
    active_count: int
    pending_count: int
    completed_count: int
    io_waiting_count: int
    
    # Performance
    migrations_this_step: int
    context_switches_this_step: int
    deadline_misses_this_step: int
    
    # Aggregate metrics
    avg_utilization: float
    load_variance: float
    throughput: float


@dataclass
class EnhancedSimulationResult:
    """
    Complete results from an enhanced simulation run.
    """
    algorithm: LoadBalancingAlgorithm
    config: SimulationConfig
    workload_pattern: WorkloadPattern
    scheduling_policy: SchedulingPolicy
    
    # Timing
    total_simulation_time: int
    wall_clock_duration: float
    
    # Process data
    process_count: int
    completed_count: int
    processes: List[Dict[str, Any]]
    
    # Processor data  
    processor_count: int
    processor_stats: List[Dict[str, Any]]
    
    # Metrics
    metrics: AdvancedMetrics
    
    # Event log
    events: List[SimulationEvent]
    
    # Snapshots for visualization
    snapshots: List[EnhancedSnapshot]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'algorithm': self.algorithm.value,
            'workload_pattern': self.workload_pattern.value,
            'scheduling_policy': self.scheduling_policy.value,
            'simulation_time': self.total_simulation_time,
            'wall_clock_ms': round(self.wall_clock_duration * 1000, 1),
            'processes': {
                'total': self.process_count,
                'completed': self.completed_count
            },
            'processors': self.processor_count,
            'metrics': self.metrics.to_dict() if self.metrics else {}
        }


class EnhancedSimulationEngine:
    """
    Production-grade simulation engine with advanced features.
    
    Features:
    - Advanced process model with I/O, memory, deadlines
    - Realistic processor model with thermal/power management
    - Multiple scheduling policies (MLFQ, EDF, Priority Aging)
    - Comprehensive event logging
    - Real-time metrics collection
    - Multiple workload patterns
    """
    
    def __init__(
        self,
        config: SimulationConfig = None,
        gui_config: GUIConfig = None,
        scheduling_policy: SchedulingPolicy = SchedulingPolicy.ROUND_ROBIN,
        workload_pattern: WorkloadPattern = WorkloadPattern.UNIFORM
    ):
        """Initialize the enhanced simulation engine."""
        self.config = config or DEFAULT_SIMULATION_CONFIG
        self.gui_config = gui_config or DEFAULT_GUI_CONFIG
        self.scheduling_policy = scheduling_policy
        self.workload_pattern = workload_pattern
        
        # Core components
        self.processors: List[AdvancedProcessor] = []
        self.load_balancer = None
        self.metrics_calculator = AdvancedMetricsCalculator()
        self.workload_generator = AdvancedWorkloadGenerator(self.config)
        
        # Process tracking
        self.all_processes: List[AdvancedProcess] = []
        self.pending_processes: deque = deque()
        self.active_processes: List[AdvancedProcess] = []
        self.completed_processes: List[AdvancedProcess] = []
        
        # Simulation state
        self.state = EnhancedSimulationState.IDLE
        self.current_time = 0
        self.max_time = 0
        
        # History
        self.snapshots: List[EnhancedSnapshot] = []
        self.events: List[SimulationEvent] = []
        
        # Callbacks
        self._on_step_callback: Optional[Callable] = None
        self._on_complete_callback: Optional[Callable] = None
        self._on_process_complete_callback: Optional[Callable] = None
        self._on_event_callback: Optional[Callable] = None
        
        # Thread safety
        self._lock = threading.Lock()
        self._stop_requested = False
        
        # Performance tracking
        self._start_wall_time: float = 0
        self._end_wall_time: float = 0
        
        # Statistics per step
        self._migrations_this_step = 0
        self._context_switches_this_step = 0
        self._deadline_misses_this_step = 0
    
    def set_callbacks(
        self,
        on_step: Callable = None,
        on_complete: Callable = None,
        on_process_complete: Callable = None,
        on_event: Callable = None
    ):
        """Set callback functions for events."""
        self._on_step_callback = on_step
        self._on_complete_callback = on_complete
        self._on_process_complete_callback = on_process_complete
        self._on_event_callback = on_event
    
    def initialize(
        self,
        algorithm: LoadBalancingAlgorithm = None,
        processes: List[AdvancedProcess] = None,
        process_mix: Dict[ProcessType, float] = None
    ) -> bool:
        """
        Initialize the simulation with all components.
        
        Args:
            algorithm: Load balancing algorithm to use
            processes: Pre-generated processes (generates if None)
            process_mix: Mix of process types for workload generation
        """
        with self._lock:
            try:
                self.state = EnhancedSimulationState.INITIALIZING
                self._reset_state()
                
                # Create processors with advanced features
                self._create_processors()
                
                # Create load balancer
                algo = algorithm or self.config.default_algorithm
                self.load_balancer = LoadBalancerFactory.create(
                    algo, self.config,
                    num_processors=self.config.num_processors
                )
                
                # Generate or use provided processes
                if processes:
                    self.all_processes = list(processes)
                else:
                    self.all_processes = self.workload_generator.generate_workload(
                        self.config.num_processes,
                        self.workload_pattern,
                        process_mix
                    )
                
                # Sort by arrival time
                self.all_processes.sort(key=lambda p: p.arrival_time)
                self.pending_processes = deque(self.all_processes)
                
                # Calculate max simulation time
                max_arrival = max(p.arrival_time for p in self.all_processes)
                total_burst = sum(p.burst_time for p in self.all_processes)
                self.max_time = max_arrival + total_burst + self.config.num_processors * 20
                
                # Log initialization
                self._log_event("INIT", details={
                    'processors': len(self.processors),
                    'processes': len(self.all_processes),
                    'algorithm': algo.value,
                    'pattern': self.workload_pattern.value,
                    'policy': self.scheduling_policy.value
                })
                
                self.state = EnhancedSimulationState.IDLE
                return True
                
            except Exception as e:
                self.state = EnhancedSimulationState.ERROR
                print(f"Initialization error: {e}")
                import traceback
                traceback.print_exc()
                return False
    
    def _create_processors(self):
        """Create processors with varied capabilities."""
        self.processors = []
        
        for i in range(self.config.num_processors):
            # Vary processor capabilities slightly
            capabilities = ProcessorCapabilities(
                base_speed=0.9 + random.random() * 0.2,  # 0.9-1.1
                boost_speed=1.3 + random.random() * 0.2,  # 1.3-1.5
                cache_size_mb=4 + (i % 3) * 4,  # 4, 8, or 12 MB
                numa_node=i // 4,  # NUMA nodes of 4 processors each
                core_type="performance" if i < self.config.num_processors // 2 else "efficiency"
            )
            
            processor = AdvancedProcessor(
                processor_id=i,
                capabilities=capabilities,
                scheduling_policy=self.scheduling_policy,
                time_quantum=self.config.time_quantum
            )
            self.processors.append(processor)
    
    def _reset_state(self):
        """Reset all simulation state."""
        self.all_processes.clear()
        self.pending_processes.clear()
        self.active_processes.clear()
        self.completed_processes.clear()
        self.snapshots.clear()
        self.events.clear()
        self.current_time = 0
        self._stop_requested = False
        self._migrations_this_step = 0
        self._context_switches_this_step = 0
        self._deadline_misses_this_step = 0
        
        for proc in self.processors:
            proc.reset()
        
        if self.load_balancer:
            self.load_balancer.reset()
        
        self.metrics_calculator.reset()
    
    def step(self) -> bool:
        """
        Execute one simulation time step.
        
        Returns:
            True if simulation should continue, False if done
        """
        if self.state not in (EnhancedSimulationState.RUNNING, 
                               EnhancedSimulationState.IDLE,
                               EnhancedSimulationState.STEPPING):
            return False
        
        if self.state == EnhancedSimulationState.IDLE:
            self.state = EnhancedSimulationState.RUNNING
            self._start_wall_time = time.time()
        
        with self._lock:
            if self._stop_requested:
                self.state = EnhancedSimulationState.STOPPED
                return False
            
            # Reset step counters
            self._migrations_this_step = 0
            self._context_switches_this_step = 0
            self._deadline_misses_this_step = 0
            
            # Step 1: Process arrivals
            self._handle_arrivals()
            
            # Step 2: Execute on all processors
            self._execute_time_step()
            
            # Step 3: Handle migrations
            self._handle_migrations()
            
            # Step 4: Update waiting times with aging
            self._update_waiting_processes()
            
            # Step 5: Record snapshot
            self._record_snapshot()
            
            # Step 6: Record metrics sample
            self.metrics_calculator.record_sample(
                self.processors,
                len(self.completed_processes),
                self.current_time
            )
            
            # Step 7: Advance time
            self.current_time += 1
            
            # Step 8: Check completion
            if self._is_simulation_complete():
                self.state = EnhancedSimulationState.COMPLETED
                self._end_wall_time = time.time()
                self._finalize()
                if self._on_complete_callback:
                    self._on_complete_callback()
                return False
            
            # Safety check
            if self.current_time >= self.max_time:
                self.state = EnhancedSimulationState.COMPLETED
                self._end_wall_time = time.time()
                self._finalize()
                return False
            
            # Invoke step callback
            if self._on_step_callback:
                self._on_step_callback(self.current_time)
            
            return True
    
    def _handle_arrivals(self):
        """Handle process arrivals at current time."""
        while (self.pending_processes and 
               self.pending_processes[0].arrival_time <= self.current_time):
            process = self.pending_processes.popleft()
            
            # Use load balancer to assign
            # Convert to basic Process interface if needed
            selected_idx = self._get_assignment(process)
            
            if selected_idx is not None:
                self.processors[selected_idx].add_process(process)
                process.state = ProcessState.READY
                self.active_processes.append(process)
                
                self._log_event("ARRIVE", process.pid, selected_idx, {
                    'type': process.process_type.value,
                    'burst': process.burst_time,
                    'priority': process.priority.name
                })
    
    def _get_assignment(self, process: AdvancedProcess) -> Optional[int]:
        """Get processor assignment from load balancer."""
        # Create a simple wrapper or use the load balancer's logic
        if hasattr(self.load_balancer, 'assign_process'):
            # For AI balancers, we might need adaptation
            # For now, use simple logic based on algorithm type
            algo = self.load_balancer.algorithm_type
            
            if algo == LoadBalancingAlgorithm.ROUND_ROBIN:
                idx = self.current_time % len(self.processors)
            elif algo == LoadBalancingAlgorithm.LEAST_LOADED:
                idx = min(range(len(self.processors)), 
                         key=lambda i: self.processors[i].get_load())
            elif algo == LoadBalancingAlgorithm.THRESHOLD_BASED:
                idx = min(range(len(self.processors)),
                         key=lambda i: self.processors[i].get_load())
            elif algo in (LoadBalancingAlgorithm.Q_LEARNING, LoadBalancingAlgorithm.DQN):
                # Use AI balancer with custom state
                try:
                    from process import Process
                    basic_process = Process(
                        pid=process.pid,
                        arrival_time=process.arrival_time,
                        burst_time=process.burst_time,
                        priority=process.priority
                    )
                    # Get processors as basic Processor objects  
                    from processor import Processor
                    basic_processors = []
                    for p in self.processors:
                        bp = Processor(p.processor_id, self.config)
                        # Copy load info
                        bp.ready_queue = deque([None] * p.get_queue_size())
                        basic_processors.append(bp)
                    
                    selected = self.load_balancer.assign_process(basic_process, basic_processors)
                    idx = selected.processor_id if selected else 0
                except:
                    idx = min(range(len(self.processors)),
                             key=lambda i: self.processors[i].get_load())
            else:
                idx = 0
            
            return idx
        return 0
    
    def _execute_time_step(self):
        """Execute one time unit on all processors."""
        for processor in self.processors:
            result = processor.execute_time_slice(self.current_time)
            
            if result['executed']:
                process = result['process']
                
                if result['completed']:
                    self._complete_process(process, processor.processor_id)
                
                if result['preempted']:
                    self._context_switches_this_step += 1
                    self._log_event("PREEMPT", process.pid, processor.processor_id)
                
                if result['io_triggered']:
                    self._log_event("IO_START", process.pid, processor.processor_id)
                
                if result['throttled']:
                    self._log_event("THROTTLE", processor_id=processor.processor_id)
            
            self._context_switches_this_step += 1 if result.get('executed') else 0
    
    def _complete_process(self, process: AdvancedProcess, processor_id: int):
        """Handle process completion."""
        if process in self.active_processes:
            self.active_processes.remove(process)
        self.completed_processes.append(process)
        
        # Check deadline
        if process.deadline and process.check_deadline(self.current_time):
            self._deadline_misses_this_step += 1
            self._log_event("DEADLINE_MISS", process.pid, processor_id)
        
        # Provide feedback to AI load balancer
        if hasattr(self.load_balancer, 'process_completed'):
            try:
                from process import Process
                basic_process = Process(
                    pid=process.pid,
                    arrival_time=process.arrival_time,
                    burst_time=process.burst_time,
                    priority=process.priority
                )
                basic_process.completion_time = process.completion_time
                basic_process.start_time = process.start_time
                self.load_balancer.process_completed(basic_process, [])
            except:
                pass
        
        self._log_event("COMPLETE", process.pid, processor_id, {
            'turnaround': process.get_turnaround_time(),
            'waiting': process.waiting_time,
            'io_wait': process.io_wait_time
        })
        
        if self._on_process_complete_callback:
            self._on_process_complete_callback(process)
    
    def _handle_migrations(self):
        """Check and execute process migrations."""
        if not hasattr(self.load_balancer, 'check_for_migration'):
            return
        
        # Get load statistics
        loads = [p.get_load() for p in self.processors]
        if not loads:
            return
        
        max_load = max(loads)
        min_load = min(loads)
        
        # Only migrate if significant imbalance
        if max_load - min_load < 3.0:  # Threshold
            return
        
        max_idx = loads.index(max_load)
        min_idx = loads.index(min_load)
        
        # Try to migrate a process
        source = self.processors[max_idx]
        if source.ready_queue:
            process = source.ready_queue.popleft()
            process.apply_migration_penalty()
            self.processors[min_idx].add_process(process, is_migration=True)
            
            self._migrations_this_step += 1
            self._log_event("MIGRATE", process.pid, min_idx, {
                'from': max_idx,
                'to': min_idx
            })
    
    def _update_waiting_processes(self):
        """Update waiting times and apply priority aging."""
        for processor in self.processors:
            for process in processor.ready_queue:
                process.waiting_time += 1
                
                # Apply priority aging
                if self.scheduling_policy == SchedulingPolicy.PRIORITY_AGING:
                    process.age_priority()
    
    def _record_snapshot(self):
        """Record current state snapshot."""
        utils = [p.get_utilization(max(1, self.current_time)) for p in self.processors]
        loads = [p.get_load() for p in self.processors]
        
        snapshot = EnhancedSnapshot(
            time=self.current_time,
            processor_loads=loads,
            processor_queues=[p.get_queue_size() for p in self.processors],
            processor_temps=[p.thermal.temperature for p in self.processors],
            processor_powers=[p.power.power_consumption for p in self.processors],
            processor_states=[p.state.value for p in self.processors],
            current_processes=[p.current_process.pid if p.current_process else None 
                              for p in self.processors],
            active_count=len(self.active_processes),
            pending_count=len(self.pending_processes),
            completed_count=len(self.completed_processes),
            io_waiting_count=sum(len(p.io_waiting_queue) for p in self.processors),
            migrations_this_step=self._migrations_this_step,
            context_switches_this_step=self._context_switches_this_step,
            deadline_misses_this_step=self._deadline_misses_this_step,
            avg_utilization=statistics.mean(utils) if utils else 0,
            load_variance=statistics.variance(loads) if len(loads) > 1 else 0,
            throughput=len(self.completed_processes) / max(1, self.current_time)
        )
        self.snapshots.append(snapshot)
    
    def _log_event(self, event_type: str, process_id: int = None,
                   processor_id: int = None, details: Dict = None):
        """Log a simulation event."""
        event = SimulationEvent(
            time=self.current_time,
            event_type=event_type,
            process_id=process_id,
            processor_id=processor_id,
            details=details or {}
        )
        self.events.append(event)
        
        if self._on_event_callback:
            self._on_event_callback(event)
    
    def _is_simulation_complete(self) -> bool:
        """Check if simulation is complete."""
        return (len(self.completed_processes) >= len(self.all_processes) or
                self.current_time >= self.max_time)
    
    def _finalize(self):
        """Finalize simulation and calculate metrics."""
        # Calculate comprehensive metrics
        self.metrics_calculator.calculate(
            self.all_processes,
            self.processors,
            self.current_time
        )
    
    def run(self, with_delay: bool = False, delay_ms: int = 50) -> EnhancedSimulationResult:
        """
        Run complete simulation.
        
        Args:
            with_delay: Add delay between steps for visualization
            delay_ms: Delay in milliseconds
        """
        self.state = EnhancedSimulationState.RUNNING
        self._start_wall_time = time.time()
        
        while self.step():
            if with_delay:
                time.sleep(delay_ms / 1000.0)
        
        return self.get_result()
    
    def run_async(self, callback: Callable = None, delay_ms: int = 50):
        """Run simulation in background thread."""
        def _run():
            result = self.run(with_delay=True, delay_ms=delay_ms)
            if callback:
                callback(result)
        
        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return thread
    
    def pause(self):
        """Pause simulation."""
        if self.state == EnhancedSimulationState.RUNNING:
            self.state = EnhancedSimulationState.PAUSED
    
    def resume(self):
        """Resume paused simulation."""
        if self.state == EnhancedSimulationState.PAUSED:
            self.state = EnhancedSimulationState.RUNNING
    
    def stop(self):
        """Stop simulation."""
        self._stop_requested = True
        self.state = EnhancedSimulationState.STOPPED
    
    def step_once(self) -> bool:
        """Execute a single step (for manual stepping)."""
        self.state = EnhancedSimulationState.STEPPING
        result = self.step()
        if not result:
            self.state = EnhancedSimulationState.COMPLETED
        else:
            self.state = EnhancedSimulationState.PAUSED
        return result
    
    def is_complete(self) -> bool:
        """Check if simulation is complete."""
        return self.state in (EnhancedSimulationState.COMPLETED,
                              EnhancedSimulationState.STOPPED)
    
    def is_running(self) -> bool:
        """Check if simulation is running."""
        return self.state == EnhancedSimulationState.RUNNING
    
    def get_result(self) -> EnhancedSimulationResult:
        """Get complete simulation results."""
        if self._end_wall_time == 0:
            self._end_wall_time = time.time()
        
        return EnhancedSimulationResult(
            algorithm=self.load_balancer.algorithm_type if self.load_balancer else LoadBalancingAlgorithm.ROUND_ROBIN,
            config=self.config,
            workload_pattern=self.workload_pattern,
            scheduling_policy=self.scheduling_policy,
            total_simulation_time=self.current_time,
            wall_clock_duration=self._end_wall_time - self._start_wall_time,
            process_count=len(self.all_processes),
            completed_count=len(self.completed_processes),
            processes=[p.to_dict() for p in self.all_processes],
            processor_count=len(self.processors),
            processor_stats=[p.get_statistics() for p in self.processors],
            metrics=self.metrics_calculator.metrics,
            events=self.events,
            snapshots=self.snapshots
        )
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current simulation state for display."""
        with self._lock:
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
                        'utilization': p.get_utilization(max(1, self.current_time)),
                        'temperature': p.thermal.temperature,
                        'power': p.power.power_consumption,
                        'state': p.state.value,
                        'io_waiting': len(p.io_waiting_queue)
                    }
                    for p in self.processors
                ],
                'algorithm': self.load_balancer.name if self.load_balancer else 'None',
                'workload_pattern': self.workload_pattern.value,
                'scheduling_policy': self.scheduling_policy.value,
                'migrations': sum(p.migration_count for p in self.all_processes),
                'context_switches': sum(p.context_switches for p in self.processors),
                'deadline_misses': sum(p.deadline_misses for p in self.processors)
            }
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        if not self.completed_processes:
            return {'completed': 0}
        
        turnarounds = [p.get_turnaround_time() for p in self.completed_processes 
                       if p.get_turnaround_time()]
        waitings = [p.waiting_time for p in self.completed_processes]
        
        return {
            'completed': len(self.completed_processes),
            'avg_turnaround': statistics.mean(turnarounds) if turnarounds else 0,
            'avg_waiting': statistics.mean(waitings) if waitings else 0,
            'throughput': len(self.completed_processes) / max(1, self.current_time),
            'avg_utilization': statistics.mean([p.get_utilization(max(1, self.current_time)) 
                                                 for p in self.processors])
        }


# =============================================================================
# ENHANCED BATCH SIMULATOR
# =============================================================================

class EnhancedBatchSimulator:
    """
    Run multiple enhanced simulations for comprehensive comparison.
    """
    
    def __init__(self, config: SimulationConfig = None):
        self.config = config or DEFAULT_SIMULATION_CONFIG
        self.results: Dict[str, EnhancedSimulationResult] = {}
    
    def run_comparison(
        self,
        algorithms: List[LoadBalancingAlgorithm] = None,
        patterns: List[WorkloadPattern] = None,
        policies: List[SchedulingPolicy] = None,
        runs_per_config: int = 1
    ) -> Dict[str, EnhancedSimulationResult]:
        """
        Run comprehensive comparison across configurations.
        """
        if algorithms is None:
            algorithms = [LoadBalancingAlgorithm.ROUND_ROBIN,
                          LoadBalancingAlgorithm.LEAST_LOADED,
                          LoadBalancingAlgorithm.THRESHOLD]
        
        if patterns is None:
            patterns = [WorkloadPattern.UNIFORM]
        
        if policies is None:
            policies = [SchedulingPolicy.ROUND_ROBIN]
        
        self.results.clear()
        
        for algo in algorithms:
            for pattern in patterns:
                for policy in policies:
                    for run in range(runs_per_config):
                        key = f"{algo.value}_{pattern.value}_{policy.value}"
                        if runs_per_config > 1:
                            key += f"_run{run}"
                        
                        engine = EnhancedSimulationEngine(
                            self.config,
                            scheduling_policy=policy,
                            workload_pattern=pattern
                        )
                        engine.initialize(algorithm=algo)
                        result = engine.run()
                        self.results[key] = result
        
        return self.results
    
    def get_summary_table(self) -> List[Dict[str, Any]]:
        """Get summary table of all results."""
        table = []
        for key, result in self.results.items():
            row = {
                'config': key,
                'algorithm': result.algorithm.value,
                'pattern': result.workload_pattern.value,
                'policy': result.scheduling_policy.value,
                'time': result.total_simulation_time,
                'completed': result.completed_count,
                **result.metrics.to_dict()
            }
            table.append(row)
        return table
    
    def get_best_configuration(self, metric: str = 'avg_turnaround') -> str:
        """Get best configuration for a given metric."""
        if not self.results:
            return "No results"
        
        # Lower is better for most metrics
        lower_better = ['avg_turnaround', 'avg_waiting', 'avg_response',
                       'turnaround_p95', 'deadline_miss_rate']
        
        best = None
        best_value = None
        
        for key, result in self.results.items():
            metrics = result.metrics.to_dict()
            value = metrics.get(metric, 0)
            
            if best is None:
                best = key
                best_value = value
            elif metric in lower_better:
                if value < best_value:
                    best = key
                    best_value = value
            else:
                if value > best_value:
                    best = key
                    best_value = value
        
        return best


# =============================================================================
# MODULE TEST  
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Enhanced Simulation Engine Test")
    print("=" * 70)
    
    # Create configuration
    config = SimulationConfig(
        num_processors=4,
        num_processes=20,
        time_quantum=3,
        min_burst_time=3,
        max_burst_time=15,
        min_arrival_time=0,
        max_arrival_time=30
    )
    
    print(f"\nConfiguration:")
    print(f"  Processors: {config.num_processors}")
    print(f"  Processes: {config.num_processes}")
    
    # Test 1: Basic simulation
    print("\n" + "-" * 70)
    print("1. Testing Basic Enhanced Simulation")
    print("-" * 70)
    
    engine = EnhancedSimulationEngine(
        config,
        scheduling_policy=SchedulingPolicy.ROUND_ROBIN,
        workload_pattern=WorkloadPattern.UNIFORM
    )
    engine.initialize(algorithm=LoadBalancingAlgorithm.LEAST_LOADED)
    
    print(f"\nRunning simulation...")
    result = engine.run()
    
    print(f"\nResults:")
    print(f"  Simulation time: {result.total_simulation_time}")
    print(f"  Wall clock: {result.wall_clock_duration:.3f}s")
    print(f"  Completed: {result.completed_count}/{result.process_count}")
    print(f"\nMetrics:")
    for key, value in result.metrics.to_dict().items():
        print(f"  {key}: {value}")
    
    # Test 2: Different workload patterns
    print("\n" + "-" * 70)
    print("2. Testing Different Workload Patterns")
    print("-" * 70)
    
    for pattern in [WorkloadPattern.UNIFORM, WorkloadPattern.BURSTY, 
                    WorkloadPattern.SPIKE]:
        engine = EnhancedSimulationEngine(
            config,
            workload_pattern=pattern
        )
        engine.initialize(algorithm=LoadBalancingAlgorithm.LEAST_LOADED)
        result = engine.run()
        
        print(f"\n  {pattern.value}:")
        print(f"    Time: {result.total_simulation_time}")
        print(f"    Avg Turnaround: {result.metrics.avg_turnaround:.2f}")
        print(f"    Throughput: {result.metrics.throughput:.4f}")
    
    # Test 3: Different scheduling policies
    print("\n" + "-" * 70)
    print("3. Testing Different Scheduling Policies")
    print("-" * 70)
    
    for policy in [SchedulingPolicy.ROUND_ROBIN, SchedulingPolicy.SJF,
                   SchedulingPolicy.MLFQ]:
        engine = EnhancedSimulationEngine(
            config,
            scheduling_policy=policy
        )
        engine.initialize()
        result = engine.run()
        
        print(f"\n  {policy.value}:")
        print(f"    Avg Turnaround: {result.metrics.avg_turnaround:.2f}")
        print(f"    Avg Waiting: {result.metrics.avg_waiting:.2f}")
        print(f"    Context Switches: {result.metrics.context_switches}")
    
    # Test 4: Step-by-step execution
    print("\n" + "-" * 70)
    print("4. Testing Step-by-Step Execution")
    print("-" * 70)
    
    engine = EnhancedSimulationEngine(config)
    engine.initialize()
    
    print("\nFirst 5 steps:")
    for i in range(5):
        continue_sim = engine.step_once()
        state = engine.get_current_state()
        print(f"  Step {state['time']}: pending={state['pending']}, "
              f"active={state['active']}, completed={state['completed']}")
        if not continue_sim:
            break
    
    # Test 5: Batch comparison
    print("\n" + "-" * 70)
    print("5. Testing Batch Comparison")
    print("-" * 70)
    
    batch = EnhancedBatchSimulator(config)
    results = batch.run_comparison(
        algorithms=[LoadBalancingAlgorithm.ROUND_ROBIN, 
                    LoadBalancingAlgorithm.LEAST_LOADED],
        patterns=[WorkloadPattern.UNIFORM, WorkloadPattern.BURSTY]
    )
    
    print(f"\nCompared {len(results)} configurations:")
    for row in batch.get_summary_table():
        print(f"  {row['config']}: turnaround={row['avg_turnaround']:.2f}")
    
    best = batch.get_best_configuration('avg_turnaround')
    print(f"\nBest for turnaround: {best}")
    
    print("\n" + "=" * 70)
    print("All enhanced simulation tests completed!")
    print("=" * 70)
