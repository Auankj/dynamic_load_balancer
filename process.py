"""
Process Module for Dynamic Load Balancing Simulator

This module defines the Process class which represents a computational task
in the operating system. In real operating systems, a process is an instance
of a running program with its own memory space, registers, and state.

Key OS Concepts Demonstrated:
- Process Control Block (PCB): Data structure storing process information
- Process States: NEW, READY, RUNNING, WAITING, COMPLETED
- Process Attributes: PID, burst time, arrival time, priority

Author: Student
Date: December 2024
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
import random
import time

from config import (
    ProcessState, 
    ProcessPriority, 
    SimulationConfig,
    DEFAULT_SIMULATION_CONFIG
)


@dataclass
class Process:
    """
    Represents a process in the operating system simulation.
    
    This class models the Process Control Block (PCB), which in real operating
    systems contains all the information the OS needs to manage a process.
    
    In this simulation, we track:
    - Identification: PID (Process ID)
    - Timing: arrival_time, burst_time, remaining_time
    - Scheduling: priority, processor assignment
    - State: current process state
    - Metrics: start_time, completion_time for performance analysis
    
    Attributes:
        pid (int): Unique process identifier
        arrival_time (int): Time when process arrives in the system
        burst_time (int): Total CPU time required to complete the process
        remaining_time (int): CPU time still needed (decreases as process executes)
        priority (ProcessPriority): Process priority level
        state (ProcessState): Current state of the process
        processor_id (Optional[int]): ID of assigned processor (None if unassigned)
        start_time (Optional[int]): Time when process first started executing
        completion_time (Optional[int]): Time when process completed
        waiting_time (int): Total time spent waiting in ready queue
        migration_count (int): Number of times process was migrated between processors
        execution_history (List): Timeline of execution for Gantt chart
    """
    
    # Core Process Attributes (required for identification)
    pid: int
    
    # Timing Attributes
    arrival_time: int = 0
    burst_time: int = 10
    remaining_time: int = field(init=False)  # Calculated from burst_time
    
    # Scheduling Attributes
    priority: ProcessPriority = ProcessPriority.MEDIUM
    state: ProcessState = ProcessState.NEW
    processor_id: Optional[int] = None
    
    # Timing Metrics (set during execution)
    start_time: Optional[int] = None
    completion_time: Optional[int] = None
    waiting_time: int = 0
    
    # Migration Tracking
    migration_count: int = 0
    original_processor_id: Optional[int] = None  # First assigned processor
    
    # Execution History for Gantt Chart
    # Each entry: {'processor_id': int, 'start': int, 'end': int}
    execution_history: List[Dict[str, int]] = field(default_factory=list)
    
    # Internal tracking
    _last_execution_start: Optional[int] = field(default=None, repr=False)
    _created_at: float = field(default_factory=time.time, repr=False)
    
    def __post_init__(self):
        """
        Initialize calculated fields after dataclass initialization.
        
        remaining_time starts equal to burst_time and decreases as the
        process executes. This separation allows us to track progress
        while knowing the original burst time for metrics.
        """
        self.remaining_time = self.burst_time
        
    def __str__(self) -> str:
        """
        Human-readable string representation of the process.
        
        Returns:
            Formatted string with key process information
        """
        return (
            f"Process[PID={self.pid}, State={self.state.name}, "
            f"Burst={self.burst_time}, Remaining={self.remaining_time}, "
            f"Priority={self.priority.name}, Processor={self.processor_id}]"
        )
    
    def __repr__(self) -> str:
        """
        Detailed representation for debugging.
        
        Returns:
            String with all process attributes
        """
        return (
            f"Process(pid={self.pid}, arrival={self.arrival_time}, "
            f"burst={self.burst_time}, remaining={self.remaining_time}, "
            f"priority={self.priority}, state={self.state}, "
            f"processor={self.processor_id}, migrations={self.migration_count})"
        )
    
    def __lt__(self, other: 'Process') -> bool:
        """
        Compare processes for priority queue ordering.
        
        Higher priority (lower number) processes come first.
        If priorities are equal, earlier arrival time wins.
        
        Args:
            other: Another Process to compare with
            
        Returns:
            True if this process should come before other
        """
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.arrival_time < other.arrival_time
    
    def __eq__(self, other: object) -> bool:
        """
        Check equality based on PID.
        
        Args:
            other: Object to compare with
            
        Returns:
            True if PIDs match
        """
        if not isinstance(other, Process):
            return False
        return self.pid == other.pid
    
    def __hash__(self) -> int:
        """
        Hash based on PID for use in sets and dicts.
        
        Returns:
            Hash value
        """
        return hash(self.pid)
    
    # =========================================================================
    # State Management Methods
    # =========================================================================
    
    def set_ready(self) -> None:
        """
        Transition process to READY state.
        
        In OS terms: Process has been created and is waiting in the 
        ready queue for CPU time.
        """
        if self.state in (ProcessState.NEW, ProcessState.WAITING, ProcessState.MIGRATING):
            self.state = ProcessState.READY
    
    def set_running(self, current_time: int) -> None:
        """
        Transition process to RUNNING state.
        
        In OS terms: Process has been selected by the scheduler and
        is now executing on a CPU.
        
        Args:
            current_time: Current simulation time (for tracking start time)
        """
        if self.state == ProcessState.READY:
            self.state = ProcessState.RUNNING
            self._last_execution_start = current_time
            
            # Record first execution start time
            if self.start_time is None:
                self.start_time = current_time
    
    def set_waiting(self) -> None:
        """
        Transition process to WAITING state.
        
        In OS terms: Process is waiting for I/O or an event.
        Note: This simulation doesn't use WAITING state extensively,
        but it's included for completeness.
        """
        if self.state == ProcessState.RUNNING:
            self.state = ProcessState.WAITING
    
    def set_completed(self, current_time: int) -> None:
        """
        Transition process to COMPLETED state.
        
        In OS terms: Process has finished execution and its resources
        can be deallocated.
        
        Args:
            current_time: Current simulation time
        """
        if self.state == ProcessState.RUNNING:
            # Record final execution segment
            if self._last_execution_start is not None:
                self.execution_history.append({
                    'processor_id': self.processor_id,
                    'start': self._last_execution_start,
                    'end': current_time
                })
            
            self.state = ProcessState.COMPLETED
            self.completion_time = current_time
            self.remaining_time = 0
    
    def set_migrating(self, current_time: int) -> None:
        """
        Transition process to MIGRATING state.
        
        Custom state for load balancing: Process is being moved from
        one processor to another. This has an associated cost.
        
        Args:
            current_time: Current simulation time
        """
        if self.state in (ProcessState.READY, ProcessState.RUNNING):
            # If was running, record the execution segment
            if self.state == ProcessState.RUNNING and self._last_execution_start is not None:
                self.execution_history.append({
                    'processor_id': self.processor_id,
                    'start': self._last_execution_start,
                    'end': current_time
                })
            
            self.state = ProcessState.MIGRATING
            self.migration_count += 1
    
    # =========================================================================
    # Execution Methods
    # =========================================================================
    
    def execute(self, time_units: int, current_time: int) -> int:
        """
        Execute the process for a given number of time units.
        
        This simulates CPU execution. The process's remaining_time
        decreases by the actual execution time.
        
        Args:
            time_units: Number of time units to execute
            current_time: Current simulation time
            
        Returns:
            Actual time units executed (may be less if process completes)
        """
        if self.state != ProcessState.RUNNING:
            return 0
        
        # Calculate actual execution time (cannot exceed remaining time)
        actual_execution = min(time_units, self.remaining_time)
        self.remaining_time -= actual_execution
        
        return actual_execution
    
    def preempt(self, current_time: int) -> None:
        """
        Preempt the running process (move back to ready queue).
        
        In OS terms: The scheduler has decided to give the CPU to
        another process (time quantum expired, higher priority process arrived).
        
        Args:
            current_time: Current simulation time
        """
        if self.state == ProcessState.RUNNING:
            # Record execution segment
            if self._last_execution_start is not None:
                self.execution_history.append({
                    'processor_id': self.processor_id,
                    'start': self._last_execution_start,
                    'end': current_time
                })
                self._last_execution_start = None
            
            self.state = ProcessState.READY
    
    def assign_to_processor(self, processor_id: int) -> None:
        """
        Assign this process to a specific processor.
        
        Args:
            processor_id: ID of the processor to assign to
        """
        # Track original processor for migration statistics
        if self.original_processor_id is None:
            self.original_processor_id = processor_id
        
        self.processor_id = processor_id
        
        # If was migrating, now ready
        if self.state == ProcessState.MIGRATING:
            self.state = ProcessState.READY
    
    def update_waiting_time(self, time_units: int) -> None:
        """
        Update the process's waiting time.
        
        Called when process is in READY state but not executing.
        
        Args:
            time_units: Time units spent waiting
        """
        if self.state == ProcessState.READY:
            self.waiting_time += time_units
    
    def add_history_entry(self, time: int, event: str, 
                          source_processor: int = None, 
                          dest_processor: int = None) -> None:
        """
        Add an event to the process execution history.
        
        Used for tracking migrations and other events for visualization.
        
        Args:
            time: Time of the event
            event: Type of event (e.g., "MIGRATED", "STARTED", "COMPLETED")
            source_processor: Source processor ID (for migrations)
            dest_processor: Destination processor ID (for migrations)
        """
        entry = {
            'time': time,
            'event': event,
            'processor_id': self.processor_id
        }
        if source_processor is not None:
            entry['source_processor'] = source_processor
        if dest_processor is not None:
            entry['dest_processor'] = dest_processor
        
        self.execution_history.append(entry)
    
    # =========================================================================
    # Query Methods
    # =========================================================================
    
    def is_completed(self) -> bool:
        """
        Check if process has finished execution.
        
        Returns:
            True if process is in COMPLETED state
        """
        return self.state == ProcessState.COMPLETED
    
    def is_ready(self) -> bool:
        """
        Check if process is ready to run.
        
        Returns:
            True if process is in READY state
        """
        return self.state == ProcessState.READY
    
    def is_running(self) -> bool:
        """
        Check if process is currently running.
        
        Returns:
            True if process is in RUNNING state
        """
        return self.state == ProcessState.RUNNING
    
    def get_progress(self) -> float:
        """
        Get execution progress as a percentage.
        
        Returns:
            Float between 0.0 and 1.0 representing completion percentage
        """
        if self.burst_time == 0:
            return 1.0
        return (self.burst_time - self.remaining_time) / self.burst_time
    
    def get_turnaround_time(self) -> Optional[int]:
        """
        Calculate turnaround time for completed process.
        
        Turnaround Time = Completion Time - Arrival Time
        
        This is a key performance metric: total time from process arrival
        to completion, including waiting and execution time.
        
        Returns:
            Turnaround time if process is completed, None otherwise
        """
        if self.completion_time is not None:
            return self.completion_time - self.arrival_time
        return None
    
    def get_response_time(self) -> Optional[int]:
        """
        Calculate response time for the process.
        
        Response Time = First Start Time - Arrival Time
        
        This measures how quickly the system responds to a new process.
        
        Returns:
            Response time if process has started, None otherwise
        """
        if self.start_time is not None:
            return self.start_time - self.arrival_time
        return None
    
    def get_waiting_time(self) -> int:
        """
        Get total waiting time.
        
        Returns:
            Total time spent in ready queue
        """
        return self.waiting_time
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert process to dictionary for serialization or logging.
        
        Returns:
            Dictionary with all process information
        """
        return {
            'pid': self.pid,
            'arrival_time': self.arrival_time,
            'burst_time': self.burst_time,
            'remaining_time': self.remaining_time,
            'priority': self.priority.name,
            'state': self.state.name,
            'processor_id': self.processor_id,
            'start_time': self.start_time,
            'completion_time': self.completion_time,
            'waiting_time': self.waiting_time,
            'migration_count': self.migration_count,
            'turnaround_time': self.get_turnaround_time(),
            'response_time': self.get_response_time(),
            'execution_history': self.execution_history
        }


# =============================================================================
# PROCESS GENERATOR
# =============================================================================

class ProcessGenerator:
    """
    Factory class for generating processes with random or specified attributes.
    
    In real systems, processes are created by:
    - User applications being launched
    - System services starting
    - Fork() system calls creating child processes
    
    This generator simulates workload creation for testing load balancing algorithms.
    """
    
    def __init__(self, config: SimulationConfig = None):
        """
        Initialize the process generator with configuration.
        
        Args:
            config: SimulationConfig instance (uses default if None)
        """
        self.config = config or DEFAULT_SIMULATION_CONFIG
        self._next_pid = 1  # Auto-incrementing PID counter
    
    def reset(self) -> None:
        """Reset the PID counter for a new simulation run."""
        self._next_pid = 1
    
    def generate_process(
        self,
        arrival_time: Optional[int] = None,
        burst_time: Optional[int] = None,
        priority: Optional[ProcessPriority] = None
    ) -> Process:
        """
        Generate a single process with random or specified attributes.
        
        Args:
            arrival_time: Specific arrival time (random if None)
            burst_time: Specific burst time (random if None)
            priority: Specific priority (random if None)
            
        Returns:
            New Process instance
        """
        # Generate random values for unspecified attributes
        if arrival_time is None:
            arrival_time = random.randint(
                self.config.min_arrival_time,
                self.config.max_arrival_time
            )
        
        if burst_time is None:
            burst_time = random.randint(
                self.config.min_burst_time,
                self.config.max_burst_time
            )
        
        if priority is None:
            priority = random.choice(list(ProcessPriority))
        
        # Create and return the process
        process = Process(
            pid=self._next_pid,
            arrival_time=arrival_time,
            burst_time=burst_time,
            priority=priority
        )
        
        self._next_pid += 1
        return process
    
    def generate_processes(self, count: Optional[int] = None) -> List[Process]:
        """
        Generate multiple processes.
        
        Args:
            count: Number of processes to generate (uses config default if None)
            
        Returns:
            List of Process instances sorted by arrival time
        """
        if count is None:
            count = self.config.num_processes
        
        processes = [self.generate_process() for _ in range(count)]
        
        # Sort by arrival time for chronological simulation
        processes.sort(key=lambda p: (p.arrival_time, p.pid))
        
        return processes
    
    def generate_balanced_workload(self, count: int) -> List[Process]:
        """
        Generate processes with similar burst times (for testing).
        
        This creates a balanced workload where all processes have
        similar execution requirements - useful for testing Round Robin.
        
        Args:
            count: Number of processes to generate
            
        Returns:
            List of processes with similar burst times
        """
        avg_burst = (self.config.min_burst_time + self.config.max_burst_time) // 2
        variance = 2  # Small variance around average
        
        processes = []
        for _ in range(count):
            burst = max(1, avg_burst + random.randint(-variance, variance))
            processes.append(self.generate_process(burst_time=burst))
        
        processes.sort(key=lambda p: (p.arrival_time, p.pid))
        return processes
    
    def generate_unbalanced_workload(self, count: int) -> List[Process]:
        """
        Generate processes with highly varied burst times (for testing).
        
        This creates an unbalanced workload with mix of very short and
        very long processes - useful for testing Least Loaded algorithm.
        
        Args:
            count: Number of processes to generate
            
        Returns:
            List of processes with varied burst times
        """
        processes = []
        
        for i in range(count):
            # Alternate between short and long processes
            if i % 3 == 0:
                burst = random.randint(1, 3)  # Short
            elif i % 3 == 1:
                burst = random.randint(15, 25)  # Long
            else:
                burst = random.randint(5, 10)  # Medium
            
            processes.append(self.generate_process(burst_time=burst))
        
        processes.sort(key=lambda p: (p.arrival_time, p.pid))
        return processes
    
    def generate_burst_workload(self, count: int, arrival_spread: int = 5) -> List[Process]:
        """
        Generate processes that arrive in bursts (for testing).
        
        This simulates real-world scenarios where multiple requests
        arrive at similar times - useful for testing threshold-based balancing.
        
        Args:
            count: Number of processes to generate
            arrival_spread: Maximum time spread for each burst
            
        Returns:
            List of processes arriving in bursts
        """
        processes = []
        current_burst_start = 0
        burst_size = count // 4  # 4 bursts
        
        for i in range(count):
            # Every burst_size processes, start a new arrival burst
            if i > 0 and i % burst_size == 0:
                current_burst_start += 10  # Gap between bursts
            
            arrival = current_burst_start + random.randint(0, arrival_spread)
            processes.append(self.generate_process(arrival_time=arrival))
        
        processes.sort(key=lambda p: (p.arrival_time, p.pid))
        return processes
    
    def generate_predefined_test_set(self) -> List[Process]:
        """
        Generate a predefined set of processes for consistent testing.
        
        Returns:
            List of processes with known, predictable attributes
        """
        # Predefined processes for deterministic testing
        test_processes = [
            (0, 8, ProcessPriority.MEDIUM),   # P1: arrives at 0, needs 8 time units
            (1, 4, ProcessPriority.HIGH),     # P2: arrives at 1, needs 4 time units
            (2, 9, ProcessPriority.LOW),      # P3: arrives at 2, needs 9 time units
            (3, 5, ProcessPriority.MEDIUM),   # P4: arrives at 3, needs 5 time units
            (4, 2, ProcessPriority.HIGH),     # P5: arrives at 4, needs 2 time units
            (5, 6, ProcessPriority.MEDIUM),   # P6: arrives at 5, needs 6 time units
            (6, 3, ProcessPriority.LOW),      # P7: arrives at 6, needs 3 time units
            (7, 7, ProcessPriority.MEDIUM),   # P8: arrives at 7, needs 7 time units
            (10, 4, ProcessPriority.HIGH),    # P9: arrives at 10, needs 4 time units
            (12, 5, ProcessPriority.MEDIUM),  # P10: arrives at 12, needs 5 time units
        ]
        
        processes = []
        for arrival, burst, priority in test_processes:
            processes.append(self.generate_process(
                arrival_time=arrival,
                burst_time=burst,
                priority=priority
            ))
        
        return processes


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Process Module Test")
    print("=" * 60)
    
    # Test Process creation
    print("\n1. Creating a process manually:")
    p1 = Process(
        pid=1,
        arrival_time=0,
        burst_time=10,
        priority=ProcessPriority.HIGH
    )
    print(f"   {p1}")
    
    # Test state transitions
    print("\n2. Testing state transitions:")
    print(f"   Initial state: {p1.state.name}")
    
    p1.set_ready()
    print(f"   After set_ready(): {p1.state.name}")
    
    p1.assign_to_processor(0)
    print(f"   Assigned to processor: {p1.processor_id}")
    
    p1.set_running(current_time=0)
    print(f"   After set_running(): {p1.state.name}")
    
    # Test execution
    print("\n3. Testing execution:")
    executed = p1.execute(time_units=4, current_time=0)
    print(f"   Executed {executed} time units")
    print(f"   Remaining time: {p1.remaining_time}")
    print(f"   Progress: {p1.get_progress()*100:.1f}%")
    
    # Test preemption
    print("\n4. Testing preemption:")
    p1.preempt(current_time=4)
    print(f"   After preempt(): {p1.state.name}")
    
    # Test completion
    print("\n5. Testing completion:")
    p1.set_running(current_time=5)
    p1.execute(time_units=6, current_time=5)
    p1.set_completed(current_time=11)
    print(f"   Final state: {p1.state.name}")
    print(f"   Turnaround time: {p1.get_turnaround_time()}")
    print(f"   Execution history: {p1.execution_history}")
    
    # Test ProcessGenerator
    print("\n6. Testing ProcessGenerator:")
    generator = ProcessGenerator()
    
    print("\n   Random processes:")
    processes = generator.generate_processes(5)
    for p in processes:
        print(f"   {p}")
    
    print("\n   Predefined test set:")
    generator.reset()
    test_processes = generator.generate_predefined_test_set()
    for p in test_processes[:5]:
        print(f"   {p}")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
