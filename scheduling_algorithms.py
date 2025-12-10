"""
Advanced CPU Scheduling Algorithms Module

This module implements production-grade CPU scheduling algorithms that go beyond
simple load balancing. These are the classic algorithms every OS student should know,
implemented with real-world considerations.

Algorithms Implemented:
1. FCFS (First Come First Served) - The OG scheduler
2. SJF (Shortest Job First) - Productivity king
3. SRTF (Shortest Remaining Time First) - Preemptive SJF
4. Priority Scheduling - VIP treatment
5. Multilevel Queue - Hierarchical scheduling
6. MLFQ (Multilevel Feedback Queue) - The genius adaptive algorithm
7. EDF (Earliest Deadline First) - Real-time champion

OS Concepts Demonstrated:
- Preemptive vs Non-preemptive scheduling
- Starvation and Aging mechanisms
- Context switching overhead
- Real-time scheduling guarantees
- Queue-based priority management

Author: AI Enhancement
Date: December 2024
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple, Deque
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto
import heapq
import time
import logging
import threading
from datetime import datetime, timedelta

from config import (
    LoadBalancingAlgorithm,
    ProcessState,
    ProcessPriority,
    SimulationConfig,
    DEFAULT_SIMULATION_CONFIG
)
from process import Process
from processor import Processor, ProcessorManager
from load_balancer import LoadBalancer, MigrationRecord

# Configure logging
logger = logging.getLogger(__name__)


# =============================================================================
# EXTENDED ENUMERATIONS
# =============================================================================

class SchedulingAlgorithm(Enum):
    """
    Extended scheduling algorithms enum.
    
    Includes classic OS scheduling algorithms beyond simple load balancing.
    """
    # Load Balancing Algorithms (existing)
    ROUND_ROBIN = "Round Robin"
    LEAST_LOADED = "Least Loaded First"
    THRESHOLD_BASED = "Threshold Based"
    Q_LEARNING = "AI (Q-Learning)"
    DQN = "AI (DQN)"
    
    # Classic CPU Scheduling Algorithms (new)
    FCFS = "FCFS (First Come First Served)"
    SJF = "SJF (Shortest Job First)"
    SRTF = "SRTF (Shortest Remaining Time First)"
    PRIORITY = "Priority Scheduling"
    PRIORITY_PREEMPTIVE = "Priority Scheduling (Preemptive)"
    MULTILEVEL_QUEUE = "Multilevel Queue"
    MLFQ = "MLFQ (Multilevel Feedback Queue)"
    EDF = "EDF (Earliest Deadline First)"


class QueueLevel(Enum):
    """
    Queue levels for Multilevel Queue Scheduling.
    
    Each level has different priority and typically different scheduling policy.
    """
    SYSTEM = 0        # Highest priority - OS processes
    INTERACTIVE = 1   # User-facing processes
    BATCH = 2         # Background computation
    IDLE = 3          # Lowest priority - run when nothing else


@dataclass
class SchedulingMetrics:
    """
    Metrics for evaluating scheduling algorithm performance.
    
    These are the key metrics OS researchers care about:
    - Throughput: Processes completed per unit time
    - Turnaround: Total time from arrival to completion
    - Waiting: Time spent in ready queue
    - Response: Time from arrival to first execution
    """
    total_processes: int = 0
    completed_processes: int = 0
    total_turnaround_time: float = 0.0
    total_waiting_time: float = 0.0
    total_response_time: float = 0.0
    context_switches: int = 0
    preemptions: int = 0
    starvation_events: int = 0
    deadlines_met: int = 0
    deadlines_missed: int = 0
    
    @property
    def avg_turnaround_time(self) -> float:
        """Average turnaround time per process."""
        if self.completed_processes == 0:
            return 0.0
        return self.total_turnaround_time / self.completed_processes
    
    @property
    def avg_waiting_time(self) -> float:
        """Average waiting time per process."""
        if self.completed_processes == 0:
            return 0.0
        return self.total_waiting_time / self.completed_processes
    
    @property
    def avg_response_time(self) -> float:
        """Average response time per process."""
        if self.completed_processes == 0:
            return 0.0
        return self.total_response_time / self.completed_processes
    
    @property
    def throughput(self) -> float:
        """Processes completed per unit time."""
        if self.total_turnaround_time == 0:
            return 0.0
        return self.completed_processes / self.total_turnaround_time
    
    @property
    def deadline_success_rate(self) -> float:
        """Percentage of deadlines met (for real-time scheduling)."""
        total = self.deadlines_met + self.deadlines_missed
        if total == 0:
            return 100.0
        return (self.deadlines_met / total) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'total_processes': self.total_processes,
            'completed_processes': self.completed_processes,
            'avg_turnaround_time': self.avg_turnaround_time,
            'avg_waiting_time': self.avg_waiting_time,
            'avg_response_time': self.avg_response_time,
            'throughput': self.throughput,
            'context_switches': self.context_switches,
            'preemptions': self.preemptions,
            'starvation_events': self.starvation_events,
            'deadline_success_rate': self.deadline_success_rate
        }


@dataclass
class ExtendedProcess:
    """
    Extended process with additional scheduling attributes.
    
    Wraps a Process with extra fields needed for advanced scheduling.
    """
    process: Process
    deadline: Optional[int] = None          # For EDF scheduling
    queue_level: QueueLevel = QueueLevel.BATCH  # For multilevel queues
    time_in_queue: int = 0                  # For aging mechanism
    first_response_time: Optional[int] = None  # When first scheduled
    last_scheduled_time: Optional[int] = None  # For MLFQ demotion
    cpu_bursts_in_quantum: int = 0          # For MLFQ
    aging_priority_boost: int = 0           # Accumulated from aging
    
    @property
    def effective_priority(self) -> int:
        """
        Calculate effective priority including aging boost.
        
        Lower number = higher priority (like ProcessPriority enum).
        """
        base_priority = self.process.priority.value
        return max(1, base_priority - self.aging_priority_boost)
    
    @property
    def remaining_time(self) -> int:
        """Remaining execution time."""
        return self.process.remaining_time
    
    @property
    def arrival_time(self) -> int:
        """Process arrival time."""
        return self.process.arrival_time
    
    @property
    def burst_time(self) -> int:
        """Original burst time."""
        return self.process.burst_time
    
    @property
    def pid(self) -> int:
        """Process ID."""
        return self.process.pid


# =============================================================================
# BASE SCHEDULER CLASS
# =============================================================================

class CPUScheduler(ABC):
    """
    Abstract base class for CPU scheduling algorithms.
    
    Extends the LoadBalancer concept with CPU scheduling semantics:
    - Manages ready queue(s)
    - Selects next process to run
    - Handles preemption
    - Tracks scheduling metrics
    """
    
    def __init__(self, config: SimulationConfig = None):
        """Initialize the scheduler."""
        self.config = config or DEFAULT_SIMULATION_CONFIG
        self.metrics = SchedulingMetrics()
        self.current_time = 0
        self.ready_queue: Deque[ExtendedProcess] = deque()
        self.running_process: Optional[ExtendedProcess] = None
        self._lock = threading.Lock()
    
    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Return the algorithm name."""
        pass
    
    @property
    @abstractmethod
    def is_preemptive(self) -> bool:
        """Whether this scheduler can preempt running processes."""
        pass
    
    @abstractmethod
    def select_next_process(self) -> Optional[ExtendedProcess]:
        """
        Select the next process to run from the ready queue.
        
        This is the core scheduling decision - each algorithm
        implements this differently.
        
        Returns:
            The selected process, or None if queue is empty
        """
        pass
    
    def add_process(self, process: Process, 
                    deadline: Optional[int] = None,
                    queue_level: QueueLevel = QueueLevel.BATCH) -> ExtendedProcess:
        """
        Add a new process to the scheduler.
        
        Args:
            process: The process to add
            deadline: Optional deadline for real-time scheduling
            queue_level: Queue level for multilevel scheduling
            
        Returns:
            The wrapped ExtendedProcess
        """
        with self._lock:
            ext_process = ExtendedProcess(
                process=process,
                deadline=deadline,
                queue_level=queue_level
            )
            self.ready_queue.append(ext_process)
            self.metrics.total_processes += 1
            
            # Check for preemption if scheduler supports it
            if self.is_preemptive and self.running_process:
                if self._should_preempt(ext_process):
                    self._preempt_current()
            
            return ext_process
    
    def _should_preempt(self, new_process: ExtendedProcess) -> bool:
        """
        Check if new process should preempt current.
        
        Override in subclasses for specific preemption logic.
        """
        return False
    
    def _preempt_current(self):
        """Preempt the currently running process."""
        if self.running_process:
            self.ready_queue.appendleft(self.running_process)
            self.running_process = None
            self.metrics.preemptions += 1
            self.metrics.context_switches += 1
    
    def tick(self, time_units: int = 1) -> Optional[ExtendedProcess]:
        """
        Advance time and execute scheduling.
        
        Args:
            time_units: Number of time units to simulate
            
        Returns:
            Currently running process (may have changed)
        """
        with self._lock:
            self.current_time += time_units
            
            # Age processes in queue (for anti-starvation)
            self._age_processes()
            
            # If no running process, select one
            if not self.running_process:
                self.running_process = self.select_next_process()
                if self.running_process:
                    self.metrics.context_switches += 1
                    if self.running_process.first_response_time is None:
                        self.running_process.first_response_time = self.current_time
                        response = self.current_time - self.running_process.arrival_time
                        self.metrics.total_response_time += response
            
            # Execute running process
            if self.running_process:
                self._execute_process(time_units)
            
            return self.running_process
    
    def _execute_process(self, time_units: int):
        """Execute the running process for given time units."""
        if not self.running_process:
            return
        
        proc = self.running_process.process
        executed = min(time_units, proc.remaining_time)
        proc.remaining_time -= executed
        
        # Check completion
        if proc.remaining_time <= 0:
            self._complete_process()
    
    def _complete_process(self):
        """Handle process completion."""
        if not self.running_process:
            return
        
        proc = self.running_process.process
        proc.state = ProcessState.COMPLETED
        proc.completion_time = self.current_time
        
        # Calculate metrics
        turnaround = self.current_time - proc.arrival_time
        waiting = turnaround - proc.burst_time
        
        self.metrics.completed_processes += 1
        self.metrics.total_turnaround_time += turnaround
        self.metrics.total_waiting_time += waiting
        
        # Check deadline (for EDF)
        if self.running_process.deadline is not None:
            if self.current_time <= self.running_process.deadline:
                self.metrics.deadlines_met += 1
            else:
                self.metrics.deadlines_missed += 1
        
        self.running_process = None
    
    def _age_processes(self):
        """
        Apply aging to prevent starvation.
        
        Processes waiting too long get priority boost.
        Override for algorithm-specific aging.
        """
        for ext_proc in self.ready_queue:
            ext_proc.time_in_queue += 1
            # Default: boost priority every 10 time units
            if ext_proc.time_in_queue % 10 == 0:
                ext_proc.aging_priority_boost += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scheduler statistics."""
        return {
            'algorithm': self.algorithm_name,
            'is_preemptive': self.is_preemptive,
            'current_time': self.current_time,
            'queue_size': len(self.ready_queue),
            'running': self.running_process.pid if self.running_process else None,
            **self.metrics.to_dict()
        }
    
    def reset(self):
        """Reset scheduler state."""
        with self._lock:
            self.ready_queue.clear()
            self.running_process = None
            self.current_time = 0
            self.metrics = SchedulingMetrics()


# =============================================================================
# FCFS - FIRST COME FIRST SERVED
# =============================================================================

class FCFSScheduler(CPUScheduler):
    """
    First Come First Served (FCFS) Scheduler
    
    The OG of schedulers. Whoever comes first gets the CPU first.
    
    Characteristics:
    - Non-preemptive: Once a process starts, it runs to completion
    - Simple FIFO queue
    - Fair in the sense of arrival order
    
    Pros:
    - Simple to implement
    - No starvation (everyone gets a turn eventually)
    - Minimal overhead
    
    Cons:
    - Convoy Effect: Short processes stuck behind long ones
    - Poor average waiting time
    - Not suitable for interactive systems
    
    Best For:
    - Batch processing systems
    - When process times are similar
    - Educational purposes (it's simple!)
    """
    
    @property
    def algorithm_name(self) -> str:
        return "FCFS (First Come First Served)"
    
    @property
    def is_preemptive(self) -> bool:
        return False  # FCFS never preempts
    
    def select_next_process(self) -> Optional[ExtendedProcess]:
        """
        Select process that arrived first (front of queue).
        
        FCFS uses a simple FIFO approach - first in, first out.
        """
        if not self.ready_queue:
            return None
        
        # Sort by arrival time to ensure true FCFS
        sorted_queue = sorted(self.ready_queue, key=lambda p: p.arrival_time)
        selected = sorted_queue[0]
        self.ready_queue.remove(selected)
        
        return selected


# =============================================================================
# SJF - SHORTEST JOB FIRST
# =============================================================================

class SJFScheduler(CPUScheduler):
    """
    Shortest Job First (SJF) Scheduler
    
    Picks the process with the shortest burst time. Productivity king!
    
    Characteristics:
    - Non-preemptive: Selected process runs to completion
    - Optimal average waiting time (provably!)
    - Requires knowledge of burst time
    
    Pros:
    - Minimum average waiting time (optimal!)
    - Good throughput
    - Efficient CPU utilization
    
    Cons:
    - Starvation: Long processes may never run
    - Need to know/estimate burst time
    - Not practical for interactive systems
    
    Real World:
    - OS doesn't actually know burst times
    - Uses exponential averaging to predict:
      τ(n+1) = α * t(n) + (1-α) * τ(n)
      where t(n) is actual burst, τ is prediction, α typically 0.5
    
    Best For:
    - Batch systems with known job sizes
    - Maximizing throughput
    """
    
    def __init__(self, config: SimulationConfig = None, 
                 use_prediction: bool = False,
                 alpha: float = 0.5):
        """
        Initialize SJF Scheduler.
        
        Args:
            config: Simulation configuration
            use_prediction: Use exponential averaging for burst prediction
            alpha: Weight for recent history (0-1)
        """
        super().__init__(config)
        self.use_prediction = use_prediction
        self.alpha = alpha
        self.burst_history: Dict[int, float] = {}  # pid -> predicted burst
    
    @property
    def algorithm_name(self) -> str:
        return "SJF (Shortest Job First)"
    
    @property
    def is_preemptive(self) -> bool:
        return False
    
    def _predict_burst(self, process: ExtendedProcess) -> float:
        """
        Predict burst time using exponential averaging.
        
        τ(n+1) = α * t(n) + (1-α) * τ(n)
        """
        if not self.use_prediction:
            return process.burst_time
        
        pid = process.pid
        if pid not in self.burst_history:
            # First time - use actual burst as initial estimate
            self.burst_history[pid] = process.burst_time
            return process.burst_time
        
        predicted = self.burst_history[pid]
        return predicted
    
    def select_next_process(self) -> Optional[ExtendedProcess]:
        """
        Select process with shortest (predicted) burst time.
        
        Uses remaining_time since we want shortest remaining work.
        """
        if not self.ready_queue:
            return None
        
        # Find shortest job
        # Use remaining_time to handle partially executed processes
        shortest = min(self.ready_queue, 
                       key=lambda p: self._predict_burst(p))
        self.ready_queue.remove(shortest)
        
        return shortest
    
    def _complete_process(self):
        """Override to update burst predictions."""
        if self.running_process and self.use_prediction:
            pid = self.running_process.pid
            actual_burst = self.running_process.burst_time
            
            if pid in self.burst_history:
                old_prediction = self.burst_history[pid]
                # Update prediction
                new_prediction = self.alpha * actual_burst + (1 - self.alpha) * old_prediction
                self.burst_history[pid] = new_prediction
            else:
                self.burst_history[pid] = actual_burst
        
        super()._complete_process()


# =============================================================================
# SRTF - SHORTEST REMAINING TIME FIRST
# =============================================================================

class SRTFScheduler(CPUScheduler):
    """
    Shortest Remaining Time First (SRTF) Scheduler
    
    The chaotic younger sibling of SJF. Preemptive version!
    
    Characteristics:
    - Preemptive: New shorter job can interrupt current
    - Always runs the process closest to completion
    - Optimal for minimizing average waiting time
    
    Pros:
    - Even better average waiting time than SJF
    - Short jobs get through FAST
    - Responsive to new arrivals
    
    Cons:
    - Higher overhead (more context switches)
    - Starvation of long processes (they get ghosted constantly)
    - Requires remaining time knowledge
    
    Best For:
    - Systems prioritizing short response times
    - When job sizes vary significantly
    """
    
    @property
    def algorithm_name(self) -> str:
        return "SRTF (Shortest Remaining Time First)"
    
    @property
    def is_preemptive(self) -> bool:
        return True
    
    def _should_preempt(self, new_process: ExtendedProcess) -> bool:
        """Preempt if new process has shorter remaining time."""
        if not self.running_process:
            return False
        return new_process.remaining_time < self.running_process.remaining_time
    
    def select_next_process(self) -> Optional[ExtendedProcess]:
        """Select process with shortest remaining time."""
        if not self.ready_queue:
            return None
        
        # Find process with shortest remaining time
        shortest = min(self.ready_queue, key=lambda p: p.remaining_time)
        self.ready_queue.remove(shortest)
        
        return shortest


# =============================================================================
# PRIORITY SCHEDULING
# =============================================================================

class PriorityScheduler(CPUScheduler):
    """
    Priority Scheduling
    
    CPU goes to the highest priority process. VIP treatment!
    
    Characteristics:
    - Comes in preemptive and non-preemptive flavors
    - Lower number = higher priority (convention)
    - Uses aging to prevent starvation
    
    Pros:
    - Critical processes get attention
    - Flexible - priority can mean anything
    - Good for mixed workloads
    
    Cons:
    - Starvation of low priority processes
    - Priority inversion problem
    - Requires careful priority assignment
    
    Starvation Solution: AGING
    - Gradually increase priority of waiting processes
    - Eventually everyone becomes high priority
    
    Priority Inversion:
    - Low priority holds resource needed by high priority
    - Solved with priority inheritance
    """
    
    def __init__(self, config: SimulationConfig = None,
                 preemptive: bool = False,
                 aging_interval: int = 5,
                 aging_boost: int = 1):
        """
        Initialize Priority Scheduler.
        
        Args:
            config: Simulation configuration
            preemptive: Enable preemptive mode
            aging_interval: Time units between aging boosts
            aging_boost: Priority boost per aging interval
        """
        super().__init__(config)
        self._preemptive = preemptive
        self.aging_interval = aging_interval
        self.aging_boost = aging_boost
    
    @property
    def algorithm_name(self) -> str:
        mode = "Preemptive" if self._preemptive else "Non-Preemptive"
        return f"Priority Scheduling ({mode})"
    
    @property
    def is_preemptive(self) -> bool:
        return self._preemptive
    
    def _should_preempt(self, new_process: ExtendedProcess) -> bool:
        """Preempt if new process has higher priority (lower number)."""
        if not self._preemptive or not self.running_process:
            return False
        return new_process.effective_priority < self.running_process.effective_priority
    
    def _age_processes(self):
        """
        Apply aging to prevent starvation.
        
        Every aging_interval, boost priority of waiting processes.
        """
        for ext_proc in self.ready_queue:
            ext_proc.time_in_queue += 1
            if ext_proc.time_in_queue % self.aging_interval == 0:
                ext_proc.aging_priority_boost += self.aging_boost
                if ext_proc.time_in_queue > 50:  # Waited too long
                    self.metrics.starvation_events += 1
    
    def select_next_process(self) -> Optional[ExtendedProcess]:
        """Select highest priority process (lowest effective priority number)."""
        if not self.ready_queue:
            return None
        
        # Find highest priority (lowest number)
        highest = min(self.ready_queue, key=lambda p: p.effective_priority)
        self.ready_queue.remove(highest)
        
        return highest


# =============================================================================
# MULTILEVEL QUEUE SCHEDULING
# =============================================================================

class MultilevelQueueScheduler(CPUScheduler):
    """
    Multilevel Queue Scheduling
    
    Think of it like a school with different sections!
    
    Structure:
    - Multiple queues with different priorities
    - Each queue can have its own scheduling algorithm
    - Processes are permanently assigned to queues
    
    Typical Queue Hierarchy:
    1. System Processes (highest priority)
    2. Interactive Processes
    3. Batch Processes
    4. Idle Processes (lowest)
    
    Characteristics:
    - Strict hierarchy: Lower queue runs only when higher is empty
    - No queue jumping (unlike MLFQ)
    - Each queue has its own scheduler
    
    Pros:
    - Good for categorized workloads
    - Predictable behavior
    - Can optimize each queue differently
    
    Cons:
    - Inflexible - processes stuck in their queue
    - Can starve lower queues
    - Requires process classification
    
    Best For:
    - Systems with clear process categories
    - Mixed OS/user workloads
    """
    
    def __init__(self, config: SimulationConfig = None,
                 queue_time_slices: Dict[QueueLevel, int] = None):
        """
        Initialize Multilevel Queue Scheduler.
        
        Args:
            config: Simulation configuration
            queue_time_slices: Time quantum for each queue level
        """
        super().__init__(config)
        
        # Separate queues for each level
        self.queues: Dict[QueueLevel, Deque[ExtendedProcess]] = {
            level: deque() for level in QueueLevel
        }
        
        # Time slices per queue (higher queue = more time)
        self.time_slices = queue_time_slices or {
            QueueLevel.SYSTEM: 8,
            QueueLevel.INTERACTIVE: 4,
            QueueLevel.BATCH: 2,
            QueueLevel.IDLE: 1
        }
        
        self.current_queue_time = 0
    
    @property
    def algorithm_name(self) -> str:
        return "Multilevel Queue Scheduling"
    
    @property
    def is_preemptive(self) -> bool:
        return True  # Higher priority queue preempts
    
    def add_process(self, process: Process, 
                    deadline: Optional[int] = None,
                    queue_level: QueueLevel = QueueLevel.BATCH) -> ExtendedProcess:
        """Add process to appropriate queue."""
        with self._lock:
            ext_process = ExtendedProcess(
                process=process,
                deadline=deadline,
                queue_level=queue_level
            )
            self.queues[queue_level].append(ext_process)
            self.metrics.total_processes += 1
            
            # Check for preemption from higher priority queue
            if self.running_process:
                if queue_level.value < self.running_process.queue_level.value:
                    self._preempt_current()
            
            return ext_process
    
    def _preempt_current(self):
        """Put current process back in its queue."""
        if self.running_process:
            level = self.running_process.queue_level
            self.queues[level].appendleft(self.running_process)
            self.running_process = None
            self.metrics.preemptions += 1
            self.metrics.context_switches += 1
    
    def select_next_process(self) -> Optional[ExtendedProcess]:
        """
        Select from highest priority non-empty queue.
        
        Strict priority: Only look at lower queues when higher are empty.
        """
        for level in QueueLevel:  # Ordered by priority
            if self.queues[level]:
                selected = self.queues[level].popleft()
                self.current_queue_time = self.time_slices[level]
                return selected
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics including per-queue info."""
        base_stats = super().get_statistics()
        base_stats['queue_sizes'] = {
            level.name: len(queue) for level, queue in self.queues.items()
        }
        return base_stats
    
    def reset(self):
        """Reset all queues."""
        with self._lock:
            for queue in self.queues.values():
                queue.clear()
            self.running_process = None
            self.current_time = 0
            self.metrics = SchedulingMetrics()


# =============================================================================
# MLFQ - MULTILEVEL FEEDBACK QUEUE
# =============================================================================

class MLFQScheduler(CPUScheduler):
    """
    Multilevel Feedback Queue (MLFQ) Scheduler
    
    The genius, adaptive version of multilevel queues!
    
    Key Innovation: Processes can MOVE between queues based on behavior
    
    Rules:
    1. New processes start at highest priority queue
    2. If process uses entire time slice: demote to lower queue
    3. If process voluntarily gives up CPU (I/O): stay or promote
    4. Periodic priority boost prevents starvation
    
    Why It's Brilliant:
    - Short jobs finish quickly (stay at high priority)
    - Long jobs gradually sink (CPU hogs get demoted)
    - Interactive processes stay responsive (I/O moves them up)
    - Learns process behavior automatically!
    
    Pros:
    - Adapts to unknown job sizes
    - Balances response time and throughput
    - No prior knowledge needed
    
    Cons:
    - Complex to implement correctly
    - Hard to tune parameters
    - Can be gamed by adversarial processes
    
    Anti-Gaming:
    - Track CPU usage across queue levels
    - Periodic priority resets
    - Rate limiting for I/O bursts
    
    Best For:
    - General purpose OS
    - Unknown workload characteristics
    - Mixed interactive/batch systems
    """
    
    def __init__(self, config: SimulationConfig = None,
                 num_queues: int = 4,
                 base_quantum: int = 2,
                 boost_interval: int = 50):
        """
        Initialize MLFQ Scheduler.
        
        Args:
            config: Simulation configuration
            num_queues: Number of priority levels
            base_quantum: Base time quantum (doubles per level)
            boost_interval: Time between priority resets
        """
        super().__init__(config)
        
        self.num_queues = num_queues
        self.base_quantum = base_quantum
        self.boost_interval = boost_interval
        
        # Priority queues (0 = highest priority)
        self.queues: List[Deque[ExtendedProcess]] = [
            deque() for _ in range(num_queues)
        ]
        
        # Time quantum per level (doubles each level)
        self.quantums = [base_quantum * (2 ** i) for i in range(num_queues)]
        
        self.current_quantum_remaining = 0
        self.time_since_boost = 0
    
    @property
    def algorithm_name(self) -> str:
        return f"MLFQ ({self.num_queues} levels)"
    
    @property
    def is_preemptive(self) -> bool:
        return True
    
    def add_process(self, process: Process, 
                    deadline: Optional[int] = None,
                    queue_level: QueueLevel = QueueLevel.BATCH) -> ExtendedProcess:
        """Add new process at highest priority queue."""
        with self._lock:
            ext_process = ExtendedProcess(
                process=process,
                deadline=deadline,
                queue_level=queue_level
            )
            ext_process.last_scheduled_time = self.current_time
            
            # New processes go to top queue
            self.queues[0].append(ext_process)
            self.metrics.total_processes += 1
            
            return ext_process
    
    def _get_process_queue_index(self, process: ExtendedProcess) -> int:
        """Determine which queue a process should be in."""
        for i, queue in enumerate(self.queues):
            if process in queue:
                return i
        return self.num_queues - 1  # Default to lowest
    
    def _demote_process(self, process: ExtendedProcess, current_level: int):
        """Demote process to lower priority queue."""
        if current_level < self.num_queues - 1:
            new_level = current_level + 1
            self.queues[new_level].append(process)
            logger.debug(f"MLFQ: Demoted P{process.pid} from Q{current_level} to Q{new_level}")
        else:
            # Already at lowest, just re-add
            self.queues[current_level].append(process)
    
    def _boost_all_priorities(self):
        """
        Periodic priority boost - anti-starvation mechanism.
        
        Move all processes to top queue to prevent starvation.
        """
        with self._lock:
            all_processes = []
            for queue in self.queues[1:]:  # Skip top queue
                all_processes.extend(queue)
                queue.clear()
            
            # Move everyone to top queue
            self.queues[0].extend(all_processes)
            
            # Reset timing
            for proc in all_processes:
                proc.cpu_bursts_in_quantum = 0
                proc.aging_priority_boost = 0
            
            logger.debug(f"MLFQ: Priority boost! Moved {len(all_processes)} processes to top queue")
    
    def tick(self, time_units: int = 1) -> Optional[ExtendedProcess]:
        """Execute with MLFQ-specific logic."""
        with self._lock:
            self.current_time += time_units
            self.time_since_boost += time_units
            
            # Periodic priority boost
            if self.time_since_boost >= self.boost_interval:
                self._boost_all_priorities()
                self.time_since_boost = 0
            
            # Track quantum usage
            if self.running_process:
                self.current_quantum_remaining -= time_units
                self.running_process.cpu_bursts_in_quantum += time_units
                
                # Time slice expired - demote
                if self.current_quantum_remaining <= 0:
                    current_level = self._get_process_queue_index(self.running_process)
                    if self.running_process.remaining_time > 0:
                        self._demote_process(self.running_process, current_level)
                    self.running_process = None
                    self.metrics.context_switches += 1
            
            # Select next process
            if not self.running_process:
                self.running_process = self.select_next_process()
                if self.running_process:
                    self.metrics.context_switches += 1
                    level = self._get_process_queue_index(self.running_process)
                    self.current_quantum_remaining = self.quantums[level]
                    
                    if self.running_process.first_response_time is None:
                        self.running_process.first_response_time = self.current_time
                        response = self.current_time - self.running_process.arrival_time
                        self.metrics.total_response_time += response
            
            # Execute
            if self.running_process:
                self._execute_process(time_units)
            
            return self.running_process
    
    def select_next_process(self) -> Optional[ExtendedProcess]:
        """Select from highest priority non-empty queue."""
        for i, queue in enumerate(self.queues):
            if queue:
                selected = queue.popleft()
                selected.cpu_bursts_in_quantum = 0
                return selected
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get MLFQ-specific statistics."""
        base_stats = super().get_statistics()
        base_stats['queue_sizes'] = [len(q) for q in self.queues]
        base_stats['quantums'] = self.quantums
        base_stats['boost_interval'] = self.boost_interval
        return base_stats
    
    def reset(self):
        """Reset MLFQ state."""
        with self._lock:
            for queue in self.queues:
                queue.clear()
            self.running_process = None
            self.current_time = 0
            self.current_quantum_remaining = 0
            self.time_since_boost = 0
            self.metrics = SchedulingMetrics()


# =============================================================================
# EDF - EARLIEST DEADLINE FIRST
# =============================================================================

class EDFScheduler(CPUScheduler):
    """
    Earliest Deadline First (EDF) Scheduler
    
    For real-time systems. Whoever has the nearest deadline gets the CPU!
    
    Characteristics:
    - Preemptive: New closer deadline preempts current
    - Dynamic priority based on deadline
    - Optimal for single processor real-time scheduling
    
    Theory:
    - Can achieve 100% CPU utilization with schedulable task set
    - Optimal: If any algorithm can meet deadlines, EDF can
    - Uses absolute deadlines for scheduling
    
    Schedulability Test:
    - Sum of (execution_time / period) <= 1 for all tasks
    - If this holds, EDF guarantees all deadlines met
    
    Pros:
    - Optimal for real-time on single processor
    - Better utilization than fixed-priority
    - Dynamic adaptation to workload
    
    Cons:
    - Harder to analyze than fixed-priority
    - Deadline miss cascade (domino effect)
    - Requires deadline information
    
    Best For:
    - Soft real-time systems
    - Multimedia applications
    - Control systems
    """
    
    def __init__(self, config: SimulationConfig = None,
                 default_deadline_slack: int = 20):
        """
        Initialize EDF Scheduler.
        
        Args:
            config: Simulation configuration
            default_deadline_slack: Default deadline = arrival + burst + slack
        """
        super().__init__(config)
        self.default_deadline_slack = default_deadline_slack
        self.deadline_heap: List[Tuple[int, ExtendedProcess]] = []  # (deadline, process)
    
    @property
    def algorithm_name(self) -> str:
        return "EDF (Earliest Deadline First)"
    
    @property
    def is_preemptive(self) -> bool:
        return True
    
    def add_process(self, process: Process, 
                    deadline: Optional[int] = None,
                    queue_level: QueueLevel = QueueLevel.BATCH) -> ExtendedProcess:
        """Add process with deadline to EDF heap."""
        with self._lock:
            # Calculate deadline if not provided
            if deadline is None:
                deadline = (process.arrival_time + 
                           process.burst_time + 
                           self.default_deadline_slack)
            
            ext_process = ExtendedProcess(
                process=process,
                deadline=deadline,
                queue_level=queue_level
            )
            
            # Use heap for efficient earliest deadline retrieval
            heapq.heappush(self.deadline_heap, (deadline, id(ext_process), ext_process))
            self.metrics.total_processes += 1
            
            # Check preemption
            if self.running_process and self.running_process.deadline:
                if deadline < self.running_process.deadline:
                    self._preempt_current()
            
            return ext_process
    
    def _preempt_current(self):
        """Preempt and reinsert into heap."""
        if self.running_process:
            deadline = self.running_process.deadline or float('inf')
            heapq.heappush(self.deadline_heap, 
                          (deadline, id(self.running_process), self.running_process))
            self.running_process = None
            self.metrics.preemptions += 1
            self.metrics.context_switches += 1
    
    def _should_preempt(self, new_process: ExtendedProcess) -> bool:
        """Preempt if new process has earlier deadline."""
        if not self.running_process or not self.running_process.deadline:
            return False
        if not new_process.deadline:
            return False
        return new_process.deadline < self.running_process.deadline
    
    def select_next_process(self) -> Optional[ExtendedProcess]:
        """Select process with earliest deadline."""
        while self.deadline_heap:
            deadline, _, process = heapq.heappop(self.deadline_heap)
            if process.remaining_time > 0:  # Still needs execution
                return process
        return None
    
    def _complete_process(self):
        """Handle completion with deadline tracking."""
        if not self.running_process:
            return
        
        if self.running_process.deadline:
            if self.current_time <= self.running_process.deadline:
                self.metrics.deadlines_met += 1
            else:
                self.metrics.deadlines_missed += 1
                logger.warning(
                    f"EDF: Process {self.running_process.pid} missed deadline! "
                    f"Completed at {self.current_time}, deadline was {self.running_process.deadline}"
                )
        
        super()._complete_process()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get EDF-specific statistics."""
        base_stats = super().get_statistics()
        base_stats['pending_deadlines'] = len(self.deadline_heap)
        base_stats['deadline_success_rate'] = self.metrics.deadline_success_rate
        return base_stats
    
    def reset(self):
        """Reset EDF state."""
        with self._lock:
            self.deadline_heap.clear()
            self.running_process = None
            self.current_time = 0
            self.metrics = SchedulingMetrics()


# =============================================================================
# LOAD BALANCER ADAPTERS
# =============================================================================

class FCFSBalancer(LoadBalancer):
    """FCFS-based load balancer adapter."""
    
    def __init__(self, config: SimulationConfig = None):
        super().__init__(config)
        self.schedulers: Dict[int, FCFSScheduler] = {}
    
    @property
    def algorithm_type(self) -> LoadBalancingAlgorithm:
        # Return a generic type since FCFS isn't in original enum
        return LoadBalancingAlgorithm.ROUND_ROBIN
    
    @property
    def name(self) -> str:
        return "FCFS (First Come First Served)"
    
    def _get_scheduler(self, processor_id: int) -> FCFSScheduler:
        """Get or create scheduler for processor."""
        if processor_id not in self.schedulers:
            self.schedulers[processor_id] = FCFSScheduler(self.config)
        return self.schedulers[processor_id]
    
    def assign_process(self, process: Process, 
                       processors: List[Processor]) -> Optional[Processor]:
        """Assign to processor with shortest queue (FCFS per processor)."""
        if not processors:
            return None
        
        # Find processor with shortest queue
        selected = min(processors, key=lambda p: p.get_queue_size())
        selected.add_process(process)
        process.processor_id = selected.processor_id
        self.assignment_count += 1
        
        return selected
    
    def check_for_migration(self, processors: List[Processor], 
                           current_time: int) -> List[MigrationRecord]:
        """FCFS doesn't migrate - respect arrival order."""
        return []


class SJFBalancer(LoadBalancer):
    """SJF-based load balancer adapter."""
    
    def __init__(self, config: SimulationConfig = None):
        super().__init__(config)
    
    @property
    def algorithm_type(self) -> LoadBalancingAlgorithm:
        return LoadBalancingAlgorithm.LEAST_LOADED
    
    @property
    def name(self) -> str:
        return "SJF (Shortest Job First)"
    
    def assign_process(self, process: Process, 
                       processors: List[Processor]) -> Optional[Processor]:
        """Assign to processor that can complete shortest job first."""
        if not processors:
            return None
        
        # Assign to least loaded (allows shortest jobs to complete faster)
        selected = min(processors, key=lambda p: p.get_load())
        selected.add_process(process)
        process.processor_id = selected.processor_id
        self.assignment_count += 1
        
        return selected
    
    def check_for_migration(self, processors: List[Processor], 
                           current_time: int) -> List[MigrationRecord]:
        """Migrate to balance for better SJF behavior."""
        return []  # Base implementation


class PriorityBalancer(LoadBalancer):
    """Priority-based load balancer adapter."""
    
    def __init__(self, config: SimulationConfig = None, preemptive: bool = False):
        super().__init__(config)
        self.preemptive = preemptive
    
    @property
    def algorithm_type(self) -> LoadBalancingAlgorithm:
        return LoadBalancingAlgorithm.THRESHOLD_BASED
    
    @property
    def name(self) -> str:
        mode = "Preemptive" if self.preemptive else "Non-Preemptive"
        return f"Priority Scheduling ({mode})"
    
    def assign_process(self, process: Process, 
                       processors: List[Processor]) -> Optional[Processor]:
        """Assign high priority to less loaded processors."""
        if not processors:
            return None
        
        # High priority processes go to least loaded
        if process.priority == ProcessPriority.HIGH:
            selected = min(processors, key=lambda p: p.get_load())
        else:
            # Lower priority can go anywhere
            selected = min(processors, key=lambda p: p.get_queue_size())
        
        selected.add_process(process)
        process.processor_id = selected.processor_id
        self.assignment_count += 1
        
        return selected
    
    def check_for_migration(self, processors: List[Processor], 
                           current_time: int) -> List[MigrationRecord]:
        """Could migrate low priority from overloaded processors."""
        return []


class MLFQBalancer(LoadBalancer):
    """MLFQ-based load balancer adapter."""
    
    def __init__(self, config: SimulationConfig = None, num_queues: int = 4):
        super().__init__(config)
        self.num_queues = num_queues
        self.process_history: Dict[int, int] = {}  # pid -> queue_level
    
    @property
    def algorithm_type(self) -> LoadBalancingAlgorithm:
        return LoadBalancingAlgorithm.THRESHOLD_BASED
    
    @property
    def name(self) -> str:
        return f"MLFQ ({self.num_queues} levels)"
    
    def assign_process(self, process: Process, 
                       processors: List[Processor]) -> Optional[Processor]:
        """Assign based on process history and current loads."""
        if not processors:
            return None
        
        # New processes or those at high queue level go to least loaded
        queue_level = self.process_history.get(process.pid, 0)
        
        if queue_level == 0:  # High priority
            selected = min(processors, key=lambda p: p.get_load())
        else:
            # Lower priority distributed more evenly
            selected = min(processors, key=lambda p: p.get_queue_size())
        
        selected.add_process(process)
        process.processor_id = selected.processor_id
        self.assignment_count += 1
        
        return selected
    
    def check_for_migration(self, processors: List[Processor], 
                           current_time: int) -> List[MigrationRecord]:
        """Balance based on queue levels."""
        return []


class EDFBalancer(LoadBalancer):
    """EDF-based load balancer adapter."""
    
    def __init__(self, config: SimulationConfig = None):
        super().__init__(config)
        self.deadlines: Dict[int, int] = {}  # pid -> deadline
    
    @property
    def algorithm_type(self) -> LoadBalancingAlgorithm:
        return LoadBalancingAlgorithm.THRESHOLD_BASED
    
    @property
    def name(self) -> str:
        return "EDF (Earliest Deadline First)"
    
    def assign_process(self, process: Process, 
                       processors: List[Processor]) -> Optional[Processor]:
        """Assign to processor that can meet deadline."""
        if not processors:
            return None
        
        # Assign to least loaded to maximize deadline success
        selected = min(processors, key=lambda p: p.get_load())
        selected.add_process(process)
        process.processor_id = selected.processor_id
        self.assignment_count += 1
        
        return selected
    
    def check_for_migration(self, processors: List[Processor], 
                           current_time: int) -> List[MigrationRecord]:
        """Could migrate to help meet deadlines."""
        return []


# =============================================================================
# SCHEDULER FACTORY
# =============================================================================

class SchedulerFactory:
    """Factory for creating scheduler instances."""
    
    @staticmethod
    def create_scheduler(algorithm: SchedulingAlgorithm, 
                         config: SimulationConfig = None,
                         **kwargs) -> CPUScheduler:
        """
        Create a scheduler instance.
        
        Args:
            algorithm: The scheduling algorithm
            config: Simulation configuration
            **kwargs: Algorithm-specific parameters
            
        Returns:
            CPUScheduler instance
        """
        schedulers = {
            SchedulingAlgorithm.FCFS: FCFSScheduler,
            SchedulingAlgorithm.SJF: SJFScheduler,
            SchedulingAlgorithm.SRTF: SRTFScheduler,
            SchedulingAlgorithm.PRIORITY: lambda c: PriorityScheduler(c, preemptive=False),
            SchedulingAlgorithm.PRIORITY_PREEMPTIVE: lambda c: PriorityScheduler(c, preemptive=True),
            SchedulingAlgorithm.MULTILEVEL_QUEUE: MultilevelQueueScheduler,
            SchedulingAlgorithm.MLFQ: MLFQScheduler,
            SchedulingAlgorithm.EDF: EDFScheduler,
        }
        
        if algorithm not in schedulers:
            raise ValueError(f"Unknown scheduling algorithm: {algorithm}")
        
        creator = schedulers[algorithm]
        if callable(creator) and not isinstance(creator, type):
            return creator(config)
        return creator(config, **kwargs)
    
    @staticmethod
    def create_balancer(algorithm: SchedulingAlgorithm,
                        config: SimulationConfig = None,
                        **kwargs) -> LoadBalancer:
        """
        Create a load balancer adapter for the scheduling algorithm.
        
        Args:
            algorithm: The scheduling algorithm
            config: Simulation configuration
            **kwargs: Algorithm-specific parameters
            
        Returns:
            LoadBalancer instance
        """
        balancers = {
            SchedulingAlgorithm.FCFS: FCFSBalancer,
            SchedulingAlgorithm.SJF: SJFBalancer,
            SchedulingAlgorithm.PRIORITY: lambda c: PriorityBalancer(c, preemptive=False),
            SchedulingAlgorithm.PRIORITY_PREEMPTIVE: lambda c: PriorityBalancer(c, preemptive=True),
            SchedulingAlgorithm.MLFQ: MLFQBalancer,
            SchedulingAlgorithm.EDF: EDFBalancer,
        }
        
        if algorithm not in balancers:
            raise ValueError(f"No load balancer adapter for: {algorithm}")
        
        creator = balancers[algorithm]
        if callable(creator) and not isinstance(creator, type):
            return creator(config)
        return creator(config, **kwargs)
    
    @staticmethod
    def get_all_algorithms() -> List[SchedulingAlgorithm]:
        """Get all available scheduling algorithms."""
        return list(SchedulingAlgorithm)
    
    @staticmethod
    def get_algorithm_info() -> Dict[str, Dict[str, Any]]:
        """Get detailed info about all algorithms."""
        return {
            SchedulingAlgorithm.FCFS.value: {
                'preemptive': False,
                'optimal_metric': None,
                'pros': ['Simple', 'No starvation', 'Low overhead'],
                'cons': ['Convoy effect', 'Poor waiting time'],
                'best_for': 'Batch processing, similar job sizes'
            },
            SchedulingAlgorithm.SJF.value: {
                'preemptive': False,
                'optimal_metric': 'Average waiting time',
                'pros': ['Optimal waiting time', 'Good throughput'],
                'cons': ['Starvation possible', 'Need burst time knowledge'],
                'best_for': 'Batch systems with known job sizes'
            },
            SchedulingAlgorithm.SRTF.value: {
                'preemptive': True,
                'optimal_metric': 'Average waiting time',
                'pros': ['Best waiting time', 'Responsive'],
                'cons': ['High overhead', 'Starvation of long jobs'],
                'best_for': 'Variable job sizes, response time priority'
            },
            SchedulingAlgorithm.PRIORITY.value: {
                'preemptive': False,
                'optimal_metric': None,
                'pros': ['Critical processes prioritized', 'Flexible'],
                'cons': ['Starvation possible', 'Priority inversion'],
                'best_for': 'Mixed priority workloads'
            },
            SchedulingAlgorithm.MLFQ.value: {
                'preemptive': True,
                'optimal_metric': 'Balanced response/throughput',
                'pros': ['Adaptive', 'No prior knowledge needed', 'Anti-starvation'],
                'cons': ['Complex', 'Hard to tune'],
                'best_for': 'General purpose OS, unknown workloads'
            },
            SchedulingAlgorithm.EDF.value: {
                'preemptive': True,
                'optimal_metric': 'Deadline success rate',
                'pros': ['Optimal for real-time', 'High utilization'],
                'cons': ['Deadline cascades', 'Needs deadline info'],
                'best_for': 'Real-time systems, multimedia'
            }
        }


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'SchedulingAlgorithm',
    'QueueLevel',
    
    # Metrics
    'SchedulingMetrics',
    'ExtendedProcess',
    
    # Base class
    'CPUScheduler',
    
    # Schedulers
    'FCFSScheduler',
    'SJFScheduler', 
    'SRTFScheduler',
    'PriorityScheduler',
    'MultilevelQueueScheduler',
    'MLFQScheduler',
    'EDFScheduler',
    
    # Load Balancer Adapters
    'FCFSBalancer',
    'SJFBalancer',
    'PriorityBalancer',
    'MLFQBalancer',
    'EDFBalancer',
    
    # Factory
    'SchedulerFactory',
]


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Advanced Scheduling Algorithms Module Test")
    print("=" * 70)
    
    from process import ProcessGenerator
    
    # Generate test processes
    generator = ProcessGenerator()
    processes = generator.generate_processes(10)
    
    # Test each scheduler
    schedulers_to_test = [
        ("FCFS", FCFSScheduler()),
        ("SJF", SJFScheduler()),
        ("SRTF", SRTFScheduler()),
        ("Priority (Non-Preemptive)", PriorityScheduler(preemptive=False)),
        ("Priority (Preemptive)", PriorityScheduler(preemptive=True)),
        ("Multilevel Queue", MultilevelQueueScheduler()),
        ("MLFQ", MLFQScheduler(num_queues=4)),
        ("EDF", EDFScheduler()),
    ]
    
    print("\n" + "-" * 70)
    print("Testing All Schedulers")
    print("-" * 70)
    
    results = []
    
    for name, scheduler in schedulers_to_test:
        print(f"\n📊 Testing {name}...")
        scheduler.reset()
        
        # Add processes
        for proc in processes:
            # Reset process state for fair comparison
            proc.remaining_time = proc.burst_time
            proc.state = ProcessState.NEW
            proc.start_time = None
            proc.completion_time = None
            
            scheduler.add_process(proc.copy() if hasattr(proc, 'copy') else Process(
                pid=proc.pid,
                arrival_time=proc.arrival_time,
                burst_time=proc.burst_time,
                priority=proc.priority
            ))
        
        # Run simulation
        max_time = 200
        while scheduler.current_time < max_time:
            running = scheduler.tick()
            if not running and not scheduler.ready_queue:
                break
        
        stats = scheduler.get_statistics()
        results.append((name, stats))
        
        print(f"   Completed: {stats['completed_processes']}/{stats['total_processes']}")
        print(f"   Avg Turnaround: {stats['avg_turnaround_time']:.2f}")
        print(f"   Avg Waiting: {stats['avg_waiting_time']:.2f}")
        print(f"   Context Switches: {stats['context_switches']}")
        print(f"   Preemptions: {stats['preemptions']}")
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("📈 ALGORITHM COMPARISON")
    print("=" * 70)
    print(f"{'Algorithm':<30} {'Avg Wait':<12} {'Avg Turn':<12} {'Switches':<10}")
    print("-" * 70)
    
    for name, stats in sorted(results, key=lambda x: x[1]['avg_waiting_time']):
        print(f"{name:<30} {stats['avg_waiting_time']:<12.2f} "
              f"{stats['avg_turnaround_time']:<12.2f} {stats['context_switches']:<10}")
    
    print("\n" + "=" * 70)
    print("✅ All scheduling algorithm tests completed!")
    print("=" * 70)
