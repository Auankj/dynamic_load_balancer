"""
Advanced Simulation Module for Dynamic Load Balancing Simulator

This module provides production-grade enhancements:
- Advanced process models (I/O bursts, memory, CPU/IO-bound types)
- Realistic processor models (thermal throttling, power states)
- Advanced scheduling (priority aging, multi-level feedback queues)
- Comprehensive workload scenarios
- Enhanced metrics and analytics

Author: Student
Date: December 2024
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple, Set
from enum import Enum, auto
from collections import deque
import random
import math
import time
import threading
import statistics
from abc import ABC, abstractmethod

from config import (
    ProcessState,
    ProcessPriority,
    LoadBalancingAlgorithm,
    SimulationConfig,
    DEFAULT_SIMULATION_CONFIG
)


# =============================================================================
# ADVANCED ENUMS AND CONSTANTS
# =============================================================================

class ProcessType(Enum):
    """
    Process types based on resource usage patterns.
    
    Real OS processes vary in their CPU vs I/O needs:
    - CPU-bound: Heavy computation (video encoding, scientific computing)
    - I/O-bound: Frequent I/O waits (web servers, databases)
    - Mixed: Combination of both (typical applications)
    - Real-time: Strict timing requirements (audio/video streaming)
    - Batch: Background processing (backups, analytics)
    """
    CPU_BOUND = "cpu_bound"
    IO_BOUND = "io_bound"
    MIXED = "mixed"
    REAL_TIME = "real_time"
    BATCH = "batch"
    INTERACTIVE = "interactive"


class ProcessorState(Enum):
    """
    Processor power and performance states.
    
    Modern CPUs have various P-states and C-states:
    - ACTIVE: Full performance mode
    - IDLE: Low power, quick wake
    - THROTTLED: Reduced performance (thermal/power limits)
    - BOOSTED: Turbo/boost mode (short bursts)
    """
    ACTIVE = "active"
    IDLE = "idle"
    THROTTLED = "throttled"
    BOOSTED = "boosted"
    DEEP_SLEEP = "deep_sleep"


class SchedulingPolicy(Enum):
    """
    Advanced scheduling policies.
    """
    FCFS = "fcfs"                       # First Come First Serve
    ROUND_ROBIN = "round_robin"         # Time-sliced round robin
    PRIORITY = "priority"               # Static priority
    PRIORITY_AGING = "priority_aging"   # Priority with aging
    MLFQ = "mlfq"                        # Multi-level Feedback Queue
    EDF = "edf"                          # Earliest Deadline First
    SJF = "sjf"                          # Shortest Job First
    SRTF = "srtf"                        # Shortest Remaining Time First


class WorkloadPattern(Enum):
    """
    Workload arrival patterns for realistic simulation.
    """
    UNIFORM = "uniform"           # Steady arrivals
    BURSTY = "bursty"             # Clustered arrivals
    POISSON = "poisson"           # Random with Poisson distribution
    DIURNAL = "diurnal"           # Day/night pattern
    SPIKE = "spike"               # Sudden traffic spike
    GRADUAL_RAMP = "gradual_ramp" # Gradually increasing
    WAVE = "wave"                 # Oscillating pattern


# =============================================================================
# ADVANCED PROCESS MODEL
# =============================================================================

@dataclass
class IOBurst:
    """
    Represents an I/O operation that blocks the process.
    
    In real systems, processes alternate between CPU and I/O bursts.
    """
    duration: int              # I/O operation time
    io_type: str = "disk"      # Type: disk, network, user, etc.
    blocking: bool = True      # Whether process must wait


@dataclass
class MemoryRequirement:
    """
    Memory requirements for a process.
    """
    memory_mb: int = 64        # Required memory in MB
    peak_memory_mb: int = 128  # Peak memory usage
    shared_memory: bool = False


@dataclass 
class ProcessDeadline:
    """
    Deadline information for real-time processes.
    """
    deadline: int              # Absolute deadline time
    period: int = 0            # For periodic tasks
    is_hard: bool = False      # Hard vs soft deadline


@dataclass
class AdvancedProcess:
    """
    Enhanced process model with realistic characteristics.
    
    Adds to the basic Process:
    - Process type (CPU/IO-bound, real-time, etc.)
    - I/O burst patterns
    - Memory requirements
    - Deadline constraints
    - Affinity and NUMA awareness
    - Priority aging
    """
    pid: int
    arrival_time: int = 0
    burst_time: int = 10
    priority: ProcessPriority = ProcessPriority.MEDIUM
    process_type: ProcessType = ProcessType.MIXED
    
    # I/O modeling
    io_bursts: List[IOBurst] = field(default_factory=list)
    io_probability: float = 0.0  # Chance of I/O at each step
    current_io_remaining: int = 0
    
    # Memory
    memory: MemoryRequirement = field(default_factory=MemoryRequirement)
    
    # Real-time constraints
    deadline: Optional[ProcessDeadline] = None
    
    # Execution tracking
    remaining_time: int = field(init=False)
    state: ProcessState = ProcessState.NEW
    processor_id: Optional[int] = None
    start_time: Optional[int] = None
    completion_time: Optional[int] = None
    waiting_time: int = 0
    io_wait_time: int = 0
    migration_count: int = 0
    
    # Advanced tracking
    effective_priority: int = field(init=False)  # For aging
    last_run_time: int = 0
    cpu_bursts_completed: int = 0
    context_switches: int = 0
    cache_warmth: float = 0.0  # Simulated cache state (0-1)
    
    # Affinity
    preferred_processor: Optional[int] = None
    processor_affinity_mask: Set[int] = field(default_factory=set)
    
    # Execution history
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        self.remaining_time = self.burst_time
        self.effective_priority = self.priority.value
        
        # Set I/O probability based on process type
        if self.process_type == ProcessType.IO_BOUND:
            self.io_probability = 0.3
        elif self.process_type == ProcessType.CPU_BOUND:
            self.io_probability = 0.05
        elif self.process_type == ProcessType.INTERACTIVE:
            self.io_probability = 0.2
        else:
            self.io_probability = 0.1
    
    def execute(self, time_units: int, current_time: int) -> Tuple[int, bool]:
        """
        Execute process with I/O modeling.
        
        Returns:
            Tuple of (time_executed, triggered_io)
        """
        if self.state != ProcessState.RUNNING:
            return 0, False
        
        # Check for I/O during execution
        if self.io_probability > 0 and random.random() < self.io_probability:
            io_duration = random.randint(2, 8)
            self.current_io_remaining = io_duration
            self.state = ProcessState.WAITING
            return 0, True
        
        actual = min(time_units, self.remaining_time)
        self.remaining_time -= actual
        self.last_run_time = current_time
        self.cpu_bursts_completed += 1
        
        # Warm up cache
        self.cache_warmth = min(1.0, self.cache_warmth + 0.1)
        
        return actual, False
    
    def process_io(self, time_units: int = 1) -> bool:
        """
        Process I/O wait time.
        
        Returns:
            True if I/O completed
        """
        if self.current_io_remaining > 0:
            self.current_io_remaining -= time_units
            self.io_wait_time += time_units
            
            if self.current_io_remaining <= 0:
                self.current_io_remaining = 0
                self.state = ProcessState.READY
                return True
        return False
    
    def age_priority(self, aging_factor: float = 0.1, max_boost: int = 2):
        """
        Apply priority aging to prevent starvation.
        
        Processes waiting too long get temporary priority boost.
        """
        # Lower number = higher priority
        boost = min(max_boost, int(self.waiting_time * aging_factor))
        self.effective_priority = max(0, self.priority.value - boost)
    
    def apply_migration_penalty(self, penalty_time: int = 2):
        """
        Apply penalty for cache cold after migration.
        """
        self.cache_warmth = 0.0
        self.migration_count += 1
        # Could add execution time penalty here
    
    def check_deadline(self, current_time: int) -> bool:
        """
        Check if deadline is met.
        
        Returns:
            True if deadline missed
        """
        if self.deadline:
            return current_time > self.deadline.deadline
        return False
    
    def is_completed(self) -> bool:
        return self.remaining_time <= 0
    
    def get_turnaround_time(self) -> Optional[int]:
        if self.completion_time is not None:
            return self.completion_time - self.arrival_time
        return None
    
    def get_response_time(self) -> Optional[int]:
        if self.start_time is not None:
            return self.start_time - self.arrival_time
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'pid': self.pid,
            'type': self.process_type.value,
            'arrival': self.arrival_time,
            'burst': self.burst_time,
            'remaining': self.remaining_time,
            'priority': self.priority.name,
            'effective_priority': self.effective_priority,
            'state': self.state.name,
            'processor': self.processor_id,
            'io_wait': self.io_wait_time,
            'waiting': self.waiting_time,
            'migrations': self.migration_count,
            'cache_warmth': round(self.cache_warmth, 2),
            'deadline_miss': self.check_deadline(self.completion_time or 0) if self.deadline else None
        }


# =============================================================================
# ADVANCED PROCESSOR MODEL
# =============================================================================

@dataclass
class ThermalState:
    """
    Thermal management for processor.
    """
    temperature: float = 40.0       # Current temp in Celsius
    max_temp: float = 85.0          # Throttle threshold
    cooling_rate: float = 0.5       # Degrees per idle tick
    heating_rate: float = 0.3       # Degrees per execution tick
    
    def update(self, is_executing: bool) -> float:
        if is_executing:
            self.temperature = min(self.max_temp + 10, 
                                   self.temperature + self.heating_rate)
        else:
            self.temperature = max(35.0, self.temperature - self.cooling_rate)
        return self.temperature
    
    def is_throttled(self) -> bool:
        return self.temperature >= self.max_temp


@dataclass
class PowerState:
    """
    Power management for processor.
    """
    current_state: ProcessorState = ProcessorState.IDLE
    power_consumption: float = 0.0  # Watts
    idle_power: float = 5.0
    active_power: float = 65.0
    boost_power: float = 95.0
    
    def update(self, state: ProcessorState):
        self.current_state = state
        power_map = {
            ProcessorState.IDLE: self.idle_power,
            ProcessorState.ACTIVE: self.active_power,
            ProcessorState.BOOSTED: self.boost_power,
            ProcessorState.THROTTLED: self.active_power * 0.7,
            ProcessorState.DEEP_SLEEP: 1.0
        }
        self.power_consumption = power_map.get(state, self.active_power)


@dataclass
class ProcessorCapabilities:
    """
    Processor capabilities and characteristics.
    """
    base_speed: float = 1.0         # Base clock multiplier
    boost_speed: float = 1.4        # Boost clock multiplier
    cache_size_mb: int = 8          # L3 cache size
    numa_node: int = 0              # NUMA node ID
    core_type: str = "performance"  # performance/efficiency


class AdvancedProcessor:
    """
    Enhanced processor model with realistic characteristics.
    
    Features:
    - Thermal throttling simulation
    - Power state management
    - Cache effects
    - Variable execution speed
    - NUMA awareness
    - Multi-level scheduling queues
    """
    
    def __init__(
        self,
        processor_id: int,
        capabilities: ProcessorCapabilities = None,
        scheduling_policy: SchedulingPolicy = SchedulingPolicy.ROUND_ROBIN,
        time_quantum: int = 3
    ):
        self.processor_id = processor_id
        self.capabilities = capabilities or ProcessorCapabilities()
        self.scheduling_policy = scheduling_policy
        self.time_quantum = time_quantum
        
        # Process queues
        self.ready_queue: deque = deque()
        self.io_waiting_queue: deque = deque()  # Processes waiting for I/O
        self.current_process: Optional[AdvancedProcess] = None
        self._current_quantum_used: int = 0
        
        # Multi-level feedback queues (if MLFQ policy)
        self.mlfq_queues: List[deque] = [deque() for _ in range(4)]
        self.mlfq_quantums = [2, 4, 8, 16]  # Time quantums per level
        
        # State management
        self.thermal = ThermalState()
        self.power = PowerState()
        self.state = ProcessorState.IDLE
        
        # Statistics
        self.total_execution_time: int = 0
        self.total_idle_time: int = 0
        self.total_io_wait_time: int = 0
        self.processes_completed: int = 0
        self.context_switches: int = 0
        self.deadline_misses: int = 0
        self.total_energy: float = 0.0
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        
        # Execution history
        self.execution_history: List[Dict] = []
        self._last_execution_start: Optional[int] = None
    
    def get_effective_speed(self) -> float:
        """
        Get current effective execution speed.
        
        Considers thermal throttling, boost mode, etc.
        """
        if self.thermal.is_throttled():
            return self.capabilities.base_speed * 0.7
        elif self.state == ProcessorState.BOOSTED:
            return self.capabilities.boost_speed
        else:
            return self.capabilities.base_speed
    
    def add_process(self, process: AdvancedProcess, is_migration: bool = False):
        """Add process to appropriate queue."""
        if is_migration:
            process.apply_migration_penalty()
            self.cache_misses += 1
        else:
            self.cache_hits += 1
        
        process.processor_id = self.processor_id
        process.state = ProcessState.READY
        
        if self.scheduling_policy == SchedulingPolicy.MLFQ:
            # New processes start at highest priority queue
            self.mlfq_queues[0].append(process)
        elif self.scheduling_policy == SchedulingPolicy.PRIORITY:
            # Insert sorted by priority
            inserted = False
            for i, p in enumerate(self.ready_queue):
                if process.effective_priority < p.effective_priority:
                    self.ready_queue.insert(i, process)
                    inserted = True
                    break
            if not inserted:
                self.ready_queue.append(process)
        else:
            self.ready_queue.append(process)
    
    def execute_time_slice(self, current_time: int) -> Dict[str, Any]:
        """
        Execute one time slice with advanced features.
        """
        result = {
            'executed': False,
            'process': None,
            'time_executed': 0,
            'completed': False,
            'io_triggered': False,
            'preempted': False,
            'throttled': False
        }
        
        # Process I/O completions
        self._process_io_queue(current_time)
        
        # Apply priority aging to waiting processes
        if self.scheduling_policy == SchedulingPolicy.PRIORITY_AGING:
            self._apply_aging()
        
        # Get next process if needed
        if self.current_process is None:
            self._select_next_process(current_time)
        
        if self.current_process is None:
            # Processor is idle
            self.total_idle_time += 1
            self.state = ProcessorState.IDLE
            self.thermal.update(False)
            self.power.update(ProcessorState.IDLE)
            self.total_energy += self.power.power_consumption
            return result
        
        # Check for thermal throttling
        if self.thermal.is_throttled():
            self.state = ProcessorState.THROTTLED
            result['throttled'] = True
        else:
            self.state = ProcessorState.ACTIVE
        
        # Execute current process
        process = self.current_process
        if process.state != ProcessState.RUNNING:
            process.state = ProcessState.RUNNING
            if process.start_time is None:
                process.start_time = current_time
            self._last_execution_start = current_time
        
        # Calculate effective execution time based on speed
        effective_time = int(1 * self.get_effective_speed())
        executed, io_triggered = process.execute(effective_time, current_time)
        
        if io_triggered:
            # Process went to I/O
            result['io_triggered'] = True
            self.io_waiting_queue.append(process)
            self.current_process = None
            self._current_quantum_used = 0
            return result
        
        self.total_execution_time += executed
        self._current_quantum_used += 1
        self.thermal.update(True)
        self.power.update(self.state)
        self.total_energy += self.power.power_consumption
        
        result['executed'] = True
        result['process'] = process
        result['time_executed'] = executed
        
        # Check completion
        if process.remaining_time <= 0:
            self._complete_process(process, current_time)
            result['completed'] = True
        elif self._should_preempt():
            self._preempt_process(process, current_time)
            result['preempted'] = True
        
        return result
    
    def _process_io_queue(self, current_time: int):
        """Process I/O waiting queue."""
        completed_io = []
        for process in self.io_waiting_queue:
            if process.process_io():
                completed_io.append(process)
        
        for process in completed_io:
            self.io_waiting_queue.remove(process)
            self.add_process(process)
            self.total_io_wait_time += process.current_io_remaining
    
    def _apply_aging(self):
        """Apply priority aging to prevent starvation."""
        for process in self.ready_queue:
            process.age_priority()
        
        # Re-sort if needed
        if self.scheduling_policy == SchedulingPolicy.PRIORITY_AGING:
            sorted_queue = sorted(self.ready_queue, 
                                  key=lambda p: p.effective_priority)
            self.ready_queue = deque(sorted_queue)
    
    def _select_next_process(self, current_time: int):
        """Select next process based on scheduling policy."""
        self.current_process = None
        self._current_quantum_used = 0
        
        if self.scheduling_policy == SchedulingPolicy.MLFQ:
            # Select from highest non-empty queue
            for level, queue in enumerate(self.mlfq_queues):
                if queue:
                    self.current_process = queue.popleft()
                    self.time_quantum = self.mlfq_quantums[level]
                    break
        elif self.scheduling_policy == SchedulingPolicy.SJF:
            if self.ready_queue:
                # Select shortest job
                shortest = min(self.ready_queue, key=lambda p: p.remaining_time)
                self.ready_queue.remove(shortest)
                self.current_process = shortest
        elif self.scheduling_policy == SchedulingPolicy.EDF:
            if self.ready_queue:
                # Select earliest deadline
                with_deadline = [p for p in self.ready_queue if p.deadline]
                if with_deadline:
                    earliest = min(with_deadline, 
                                   key=lambda p: p.deadline.deadline)
                    self.ready_queue.remove(earliest)
                    self.current_process = earliest
                elif self.ready_queue:
                    self.current_process = self.ready_queue.popleft()
        else:
            # FCFS, Round Robin, Priority
            if self.ready_queue:
                self.current_process = self.ready_queue.popleft()
        
        if self.current_process:
            self.context_switches += 1
    
    def _should_preempt(self) -> bool:
        """Check if current process should be preempted."""
        if self.scheduling_policy in (SchedulingPolicy.FCFS, 
                                       SchedulingPolicy.SJF):
            return False  # Non-preemptive
        
        if self.scheduling_policy == SchedulingPolicy.SRTF:
            # Preempt if shorter job arrived
            if self.ready_queue:
                shortest = min(self.ready_queue, key=lambda p: p.remaining_time)
                if shortest.remaining_time < self.current_process.remaining_time:
                    return True
        
        return self._current_quantum_used >= self.time_quantum
    
    def _preempt_process(self, process: AdvancedProcess, current_time: int):
        """Preempt current process."""
        if self._last_execution_start is not None:
            self.execution_history.append({
                'pid': process.pid,
                'start': self._last_execution_start,
                'end': current_time,
                'type': 'preempt'
            })
        
        process.state = ProcessState.READY
        process.cache_warmth *= 0.5  # Cache cools down
        
        if self.scheduling_policy == SchedulingPolicy.MLFQ:
            # Demote to next level
            current_level = self._find_mlfq_level(process)
            next_level = min(current_level + 1, len(self.mlfq_queues) - 1)
            self.mlfq_queues[next_level].append(process)
        else:
            self.ready_queue.append(process)
        
        self.current_process = None
        self._current_quantum_used = 0
        self._last_execution_start = None
    
    def _find_mlfq_level(self, process: AdvancedProcess) -> int:
        """Find which MLFQ level a process was in."""
        # Based on how many times it's been preempted
        return min(process.context_switches, len(self.mlfq_queues) - 1)
    
    def _complete_process(self, process: AdvancedProcess, current_time: int):
        """Complete current process."""
        if self._last_execution_start is not None:
            self.execution_history.append({
                'pid': process.pid,
                'start': self._last_execution_start,
                'end': current_time,
                'type': 'complete'
            })
        
        process.state = ProcessState.COMPLETED
        process.completion_time = current_time
        
        # Check deadline
        if process.deadline and process.check_deadline(current_time):
            self.deadline_misses += 1
        
        self.processes_completed += 1
        self.current_process = None
        self._current_quantum_used = 0
        self._last_execution_start = None
    
    def get_load(self) -> float:
        """Calculate processor load."""
        queue_count = len(self.ready_queue) + len(self.io_waiting_queue)
        if self.current_process:
            queue_count += 1
        
        work_remaining = sum(p.remaining_time for p in self.ready_queue)
        if self.current_process:
            work_remaining += self.current_process.remaining_time
        
        return queue_count * 0.3 + work_remaining * 0.1
    
    def get_queue_size(self) -> int:
        return len(self.ready_queue) + (1 if self.current_process else 0)
    
    def is_idle(self) -> bool:
        return (self.current_process is None and 
                len(self.ready_queue) == 0 and 
                len(self.io_waiting_queue) == 0)
    
    def get_utilization(self, total_time: int) -> float:
        if total_time <= 0:
            return 0.0
        return min(1.0, self.total_execution_time / total_time)
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            'processor_id': self.processor_id,
            'execution_time': self.total_execution_time,
            'idle_time': self.total_idle_time,
            'io_wait_time': self.total_io_wait_time,
            'completed': self.processes_completed,
            'context_switches': self.context_switches,
            'deadline_misses': self.deadline_misses,
            'total_energy_wh': round(self.total_energy / 3600, 2),
            'temperature': round(self.thermal.temperature, 1),
            'power_state': self.state.value,
            'cache_hit_rate': self.cache_hits / max(1, self.cache_hits + self.cache_misses)
        }
    
    def reset(self):
        """Reset processor state."""
        self.ready_queue.clear()
        self.io_waiting_queue.clear()
        for q in self.mlfq_queues:
            q.clear()
        self.current_process = None
        self._current_quantum_used = 0
        self.thermal = ThermalState()
        self.power = PowerState()
        self.state = ProcessorState.IDLE
        self.total_execution_time = 0
        self.total_idle_time = 0
        self.total_io_wait_time = 0
        self.processes_completed = 0
        self.context_switches = 0
        self.deadline_misses = 0
        self.total_energy = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        self.execution_history.clear()


# =============================================================================
# ADVANCED WORKLOAD GENERATOR
# =============================================================================

class AdvancedWorkloadGenerator:
    """
    Generates realistic workloads with various patterns.
    """
    
    def __init__(self, config: SimulationConfig = None):
        self.config = config or DEFAULT_SIMULATION_CONFIG
        self._next_pid = 1
    
    def reset(self):
        self._next_pid = 1
    
    def generate_workload(
        self,
        count: int,
        pattern: WorkloadPattern = WorkloadPattern.UNIFORM,
        process_mix: Dict[ProcessType, float] = None
    ) -> List[AdvancedProcess]:
        """
        Generate a workload with specified pattern and process mix.
        
        Args:
            count: Number of processes
            pattern: Arrival pattern
            process_mix: Distribution of process types (defaults to realistic mix)
        """
        if process_mix is None:
            process_mix = {
                ProcessType.CPU_BOUND: 0.2,
                ProcessType.IO_BOUND: 0.3,
                ProcessType.MIXED: 0.3,
                ProcessType.INTERACTIVE: 0.15,
                ProcessType.BATCH: 0.05
            }
        
        # Generate arrival times based on pattern
        arrivals = self._generate_arrivals(count, pattern)
        
        # Generate processes
        processes = []
        for i, arrival in enumerate(arrivals):
            ptype = self._select_process_type(process_mix)
            process = self._create_process(arrival, ptype)
            processes.append(process)
        
        return sorted(processes, key=lambda p: p.arrival_time)
    
    def _generate_arrivals(self, count: int, pattern: WorkloadPattern) -> List[int]:
        """Generate arrival times based on pattern."""
        arrivals = []
        max_time = self.config.max_arrival_time
        
        if pattern == WorkloadPattern.UNIFORM:
            # Evenly distributed
            interval = max(1, max_time // count)
            for i in range(count):
                arrivals.append(i * interval + random.randint(0, interval//2))
        
        elif pattern == WorkloadPattern.BURSTY:
            # Clustered bursts
            num_bursts = max(1, count // 5)
            burst_starts = sorted(random.sample(range(max_time), num_bursts))
            for i in range(count):
                burst = burst_starts[i % len(burst_starts)]
                arrivals.append(burst + random.randint(0, 3))
        
        elif pattern == WorkloadPattern.POISSON:
            # Poisson process
            rate = count / max_time  # λ
            current_time = 0
            while len(arrivals) < count:
                # Exponential inter-arrival times
                interval = int(-math.log(1 - random.random()) / rate)
                current_time += max(1, interval)
                if current_time <= max_time:
                    arrivals.append(current_time)
                else:
                    arrivals.append(random.randint(0, max_time))
        
        elif pattern == WorkloadPattern.SPIKE:
            # Most arrivals in a short window
            spike_start = max_time // 3
            spike_end = spike_start + max_time // 6
            for i in range(count):
                if random.random() < 0.7:  # 70% in spike
                    arrivals.append(random.randint(spike_start, spike_end))
                else:
                    arrivals.append(random.randint(0, max_time))
        
        elif pattern == WorkloadPattern.DIURNAL:
            # Day/night pattern (sinusoidal)
            for i in range(count):
                # Higher probability during "peak hours"
                phase = (i / count) * 2 * math.pi
                probability = (1 + math.sin(phase)) / 2
                base = int(i * max_time / count)
                offset = int((1 - probability) * max_time / 4)
                arrivals.append(max(0, min(max_time, base - offset)))
        
        elif pattern == WorkloadPattern.GRADUAL_RAMP:
            # Increasing rate
            for i in range(count):
                # Quadratic distribution (more at the end)
                t = (i / count) ** 2
                arrivals.append(int(t * max_time))
        
        elif pattern == WorkloadPattern.WAVE:
            # Oscillating pattern
            wavelength = max_time / 3
            for i in range(count):
                base = i * max_time // count
                wave = int(wavelength/4 * math.sin(2 * math.pi * base / wavelength))
                arrivals.append(max(0, base + wave))
        
        else:
            arrivals = [random.randint(0, max_time) for _ in range(count)]
        
        return arrivals[:count]
    
    def _select_process_type(self, mix: Dict[ProcessType, float]) -> ProcessType:
        """Select process type based on distribution."""
        r = random.random()
        cumulative = 0.0
        for ptype, prob in mix.items():
            cumulative += prob
            if r <= cumulative:
                return ptype
        return ProcessType.MIXED
    
    def _create_process(self, arrival: int, ptype: ProcessType) -> AdvancedProcess:
        """Create a process of specified type."""
        # Burst time varies by type
        if ptype == ProcessType.CPU_BOUND:
            burst = random.randint(15, 30)
            priority = random.choice([ProcessPriority.MEDIUM, ProcessPriority.LOW])
        elif ptype == ProcessType.IO_BOUND:
            burst = random.randint(5, 15)
            priority = ProcessPriority.MEDIUM
        elif ptype == ProcessType.INTERACTIVE:
            burst = random.randint(2, 8)
            priority = ProcessPriority.HIGH
        elif ptype == ProcessType.REAL_TIME:
            burst = random.randint(3, 10)
            priority = ProcessPriority.CRITICAL
        elif ptype == ProcessType.BATCH:
            burst = random.randint(20, 50)
            priority = ProcessPriority.LOW
        else:  # MIXED
            burst = random.randint(self.config.min_burst_time, 
                                   self.config.max_burst_time)
            priority = random.choice(list(ProcessPriority))
        
        # Memory requirements vary by type
        if ptype == ProcessType.BATCH:
            memory = MemoryRequirement(memory_mb=256, peak_memory_mb=512)
        elif ptype == ProcessType.INTERACTIVE:
            memory = MemoryRequirement(memory_mb=32, peak_memory_mb=64)
        else:
            memory = MemoryRequirement(memory_mb=64, peak_memory_mb=128)
        
        # Deadline for real-time processes
        deadline = None
        if ptype == ProcessType.REAL_TIME:
            deadline = ProcessDeadline(
                deadline=arrival + burst * 2,
                is_hard=random.random() < 0.3
            )
        
        process = AdvancedProcess(
            pid=self._next_pid,
            arrival_time=arrival,
            burst_time=burst,
            priority=priority,
            process_type=ptype,
            memory=memory,
            deadline=deadline
        )
        self._next_pid += 1
        return process
    
    def generate_stress_test(self, count: int = 100) -> List[AdvancedProcess]:
        """Generate a stress test workload."""
        return self.generate_workload(
            count,
            pattern=WorkloadPattern.SPIKE,
            process_mix={
                ProcessType.CPU_BOUND: 0.5,
                ProcessType.IO_BOUND: 0.2,
                ProcessType.MIXED: 0.2,
                ProcessType.REAL_TIME: 0.1
            }
        )
    
    def generate_real_time_test(self, count: int = 50) -> List[AdvancedProcess]:
        """Generate workload with many real-time processes."""
        return self.generate_workload(
            count,
            pattern=WorkloadPattern.UNIFORM,
            process_mix={
                ProcessType.REAL_TIME: 0.6,
                ProcessType.INTERACTIVE: 0.3,
                ProcessType.CPU_BOUND: 0.1
            }
        )
    
    def generate_io_heavy(self, count: int = 50) -> List[AdvancedProcess]:
        """Generate I/O-heavy workload."""
        return self.generate_workload(
            count,
            pattern=WorkloadPattern.BURSTY,
            process_mix={
                ProcessType.IO_BOUND: 0.6,
                ProcessType.INTERACTIVE: 0.2,
                ProcessType.MIXED: 0.2
            }
        )


# =============================================================================
# ADVANCED METRICS
# =============================================================================

@dataclass
class AdvancedMetrics:
    """
    Comprehensive metrics for simulation analysis.
    """
    # Basic metrics
    total_processes: int = 0
    completed_processes: int = 0
    avg_turnaround: float = 0.0
    avg_waiting: float = 0.0
    avg_response: float = 0.0
    
    # Throughput
    throughput: float = 0.0  # Processes per time unit
    
    # Percentiles
    turnaround_p50: float = 0.0
    turnaround_p95: float = 0.0
    turnaround_p99: float = 0.0
    response_p50: float = 0.0
    response_p95: float = 0.0
    response_p99: float = 0.0
    
    # Fairness
    jains_fairness: float = 0.0
    max_wait_time: int = 0
    starvation_count: int = 0  # Processes waiting > threshold
    
    # Efficiency
    avg_utilization: float = 0.0
    total_energy_wh: float = 0.0
    migrations: int = 0
    context_switches: int = 0
    
    # Real-time
    deadline_miss_rate: float = 0.0
    deadline_misses: int = 0
    
    # I/O
    total_io_wait: int = 0
    avg_io_wait: float = 0.0
    
    # Cache
    avg_cache_hit_rate: float = 0.0
    
    # Time series for charts
    utilization_history: List[float] = field(default_factory=list)
    queue_length_history: List[float] = field(default_factory=list)
    throughput_history: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'total_processes': self.total_processes,
            'completed': self.completed_processes,
            'avg_turnaround': round(self.avg_turnaround, 2),
            'avg_waiting': round(self.avg_waiting, 2),
            'avg_response': round(self.avg_response, 2),
            'throughput': round(self.throughput, 4),
            'turnaround_p95': round(self.turnaround_p95, 2),
            'response_p95': round(self.response_p95, 2),
            'jains_fairness': round(self.jains_fairness, 4),
            'avg_utilization': round(self.avg_utilization * 100, 1),
            'total_energy_wh': round(self.total_energy_wh, 2),
            'migrations': self.migrations,
            'context_switches': self.context_switches,
            'deadline_miss_rate': round(self.deadline_miss_rate * 100, 1),
            'avg_cache_hit_rate': round(self.avg_cache_hit_rate * 100, 1)
        }


class AdvancedMetricsCalculator:
    """
    Calculate comprehensive metrics from simulation data.
    """
    
    def __init__(self):
        self.metrics = AdvancedMetrics()
        self._utilization_samples: List[float] = []
        self._queue_samples: List[float] = []
        self._completed_over_time: List[int] = []
    
    def record_sample(
        self,
        processors: List[AdvancedProcessor],
        completed_count: int,
        current_time: int
    ):
        """Record metrics sample at current time."""
        # Utilization
        utils = [p.get_utilization(max(1, current_time)) for p in processors]
        avg_util = sum(utils) / len(utils) if utils else 0
        self._utilization_samples.append(avg_util)
        
        # Queue lengths
        total_queue = sum(p.get_queue_size() for p in processors)
        avg_queue = total_queue / len(processors) if processors else 0
        self._queue_samples.append(avg_queue)
        
        # Throughput tracking
        self._completed_over_time.append(completed_count)
    
    def calculate(
        self,
        processes: List[AdvancedProcess],
        processors: List[AdvancedProcessor],
        total_time: int
    ) -> AdvancedMetrics:
        """Calculate all metrics."""
        m = self.metrics
        
        # Basic counts
        m.total_processes = len(processes)
        completed = [p for p in processes if p.is_completed()]
        m.completed_processes = len(completed)
        
        if not completed:
            return m
        
        # Timing metrics
        turnarounds = [p.get_turnaround_time() for p in completed if p.get_turnaround_time()]
        waitings = [p.waiting_time for p in completed]
        responses = [p.get_response_time() for p in completed if p.get_response_time()]
        
        if turnarounds:
            m.avg_turnaround = statistics.mean(turnarounds)
            m.turnaround_p50 = statistics.median(turnarounds)
            sorted_ta = sorted(turnarounds)
            m.turnaround_p95 = sorted_ta[int(len(sorted_ta) * 0.95)] if len(sorted_ta) > 1 else sorted_ta[0]
            m.turnaround_p99 = sorted_ta[int(len(sorted_ta) * 0.99)] if len(sorted_ta) > 1 else sorted_ta[0]
        
        if waitings:
            m.avg_waiting = statistics.mean(waitings)
            m.max_wait_time = max(waitings)
            m.starvation_count = sum(1 for w in waitings if w > m.avg_turnaround * 2)
        
        if responses:
            m.avg_response = statistics.mean(responses)
            m.response_p50 = statistics.median(responses)
            sorted_r = sorted(responses)
            m.response_p95 = sorted_r[int(len(sorted_r) * 0.95)] if len(sorted_r) > 1 else sorted_r[0]
            m.response_p99 = sorted_r[int(len(sorted_r) * 0.99)] if len(sorted_r) > 1 else sorted_r[0]
        
        # Throughput
        m.throughput = m.completed_processes / max(1, total_time)
        
        # Fairness (Jain's index on waiting times)
        if waitings:
            sum_x = sum(waitings)
            sum_x2 = sum(w * w for w in waitings)
            n = len(waitings)
            if sum_x2 > 0:
                m.jains_fairness = (sum_x ** 2) / (n * sum_x2)
        
        # Processor metrics
        utils = [p.get_utilization(total_time) for p in processors]
        m.avg_utilization = statistics.mean(utils) if utils else 0
        m.total_energy_wh = sum(p.total_energy / 3600 for p in processors)
        m.context_switches = sum(p.context_switches for p in processors)
        m.migrations = sum(p.migration_count for p in processes)
        
        # Cache
        cache_hits = sum(p.cache_hits for p in processors)
        cache_total = cache_hits + sum(p.cache_misses for p in processors)
        m.avg_cache_hit_rate = cache_hits / max(1, cache_total)
        
        # Deadline metrics
        deadline_processes = [p for p in completed if p.deadline]
        if deadline_processes:
            misses = sum(1 for p in deadline_processes 
                        if p.check_deadline(p.completion_time or 0))
            m.deadline_misses = misses
            m.deadline_miss_rate = misses / len(deadline_processes)
        
        # I/O metrics
        io_waits = [p.io_wait_time for p in completed if p.io_wait_time > 0]
        if io_waits:
            m.total_io_wait = sum(io_waits)
            m.avg_io_wait = statistics.mean(io_waits)
        
        # Time series
        m.utilization_history = self._utilization_samples.copy()
        m.queue_length_history = self._queue_samples.copy()
        
        # Calculate throughput over time (moving average)
        if len(self._completed_over_time) > 1:
            window = min(10, len(self._completed_over_time))
            for i in range(len(self._completed_over_time)):
                start = max(0, i - window)
                rate = (self._completed_over_time[i] - self._completed_over_time[start]) / max(1, i - start)
                m.throughput_history.append(rate)
        
        return m
    
    def reset(self):
        self.metrics = AdvancedMetrics()
        self._utilization_samples.clear()
        self._queue_samples.clear()
        self._completed_over_time.clear()


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Advanced Simulation Module Test")
    print("=" * 70)
    
    # Test Advanced Process
    print("\n1. Testing AdvancedProcess")
    process = AdvancedProcess(
        pid=1,
        arrival_time=0,
        burst_time=10,
        process_type=ProcessType.IO_BOUND
    )
    print(f"   Created: {process.to_dict()}")
    
    # Test Advanced Processor
    print("\n2. Testing AdvancedProcessor")
    processor = AdvancedProcessor(
        processor_id=0,
        scheduling_policy=SchedulingPolicy.MLFQ
    )
    processor.add_process(process)
    print(f"   Queue size: {processor.get_queue_size()}")
    
    # Execute a few steps
    for t in range(5):
        result = processor.execute_time_slice(t)
        print(f"   Time {t}: executed={result['executed']}, io={result['io_triggered']}")
    
    print(f"   Stats: {processor.get_statistics()}")
    
    # Test Workload Generator
    print("\n3. Testing AdvancedWorkloadGenerator")
    generator = AdvancedWorkloadGenerator()
    
    for pattern in [WorkloadPattern.UNIFORM, WorkloadPattern.BURSTY, 
                    WorkloadPattern.SPIKE]:
        processes = generator.generate_workload(20, pattern)
        arrivals = [p.arrival_time for p in processes]
        types = [p.process_type.value for p in processes]
        print(f"   {pattern.value}: arrivals range={min(arrivals)}-{max(arrivals)}")
    
    # Test stress test
    stress = generator.generate_stress_test(50)
    print(f"   Stress test: {len(stress)} processes, "
          f"{sum(1 for p in stress if p.process_type == ProcessType.CPU_BOUND)} CPU-bound")
    
    # Test Metrics
    print("\n4. Testing AdvancedMetricsCalculator")
    calc = AdvancedMetricsCalculator()
    
    # Simulate some completed processes
    test_processes = []
    for i in range(10):
        p = AdvancedProcess(pid=i, arrival_time=i, burst_time=5+i)
        p.state = ProcessState.COMPLETED
        p.start_time = i + 1
        p.completion_time = i + 10
        p.waiting_time = 1
        test_processes.append(p)
    
    test_processors = [AdvancedProcessor(i) for i in range(4)]
    for proc in test_processors:
        proc.total_execution_time = 20
        proc.processes_completed = 3
    
    metrics = calc.calculate(test_processes, test_processors, 50)
    print(f"   Metrics: {metrics.to_dict()}")
    
    print("\n" + "=" * 70)
    print("All advanced simulation tests completed!")
    print("=" * 70)
