"""
Integration Module for Enhanced Simulation System

This module bridges the advanced simulation capabilities with the existing
GUI and load balancing infrastructure, providing a unified interface for
production-grade simulation.

Features:
- Seamless integration with existing GUI
- Enhanced simulation modes (basic, advanced, real-time)
- Production metrics dashboard
- Scenario management
- Performance profiling

Author: Student
Date: December 2024
"""

import time
import threading
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum
from pathlib import Path
import logging

# Import existing modules
from config import SimulationConfig, LoadBalancingAlgorithm, ProcessState, ProcessPriority
from process import Process, ProcessGenerator
from processor import Processor, ProcessorManager
from load_balancer import LoadBalancerFactory
from simulation import SimulationEngine, SimulationState
from metrics import MetricsCalculator, SystemMetrics

# Import enhanced modules
from advanced_simulation import (
    ProcessType, WorkloadPattern,
    AdvancedProcess, AdvancedProcessor, PowerState,
    AdvancedWorkloadGenerator, AdvancedMetricsCalculator,
    SchedulingPolicy
)
from enhanced_simulation import (
    EnhancedSimulationEngine,
    EnhancedSimulationResult
)

logger = logging.getLogger(__name__)


# =============================================================================
# SCENARIO DEFINITIONS
# =============================================================================

class ScenarioType(Enum):
    """Predefined simulation scenarios for testing"""
    BASIC = "basic"
    CPU_INTENSIVE = "cpu_intensive"
    IO_INTENSIVE = "io_intensive"
    MIXED_WORKLOAD = "mixed_workload"
    BURSTY_TRAFFIC = "bursty_traffic"
    HIGH_PRIORITY = "high_priority"
    REAL_TIME = "real_time"
    STRESS_TEST = "stress_test"
    BALANCED = "balanced"
    CUSTOM = "custom"


@dataclass
class SimulationScenario:
    """Complete scenario configuration"""
    name: str
    description: str
    scenario_type: ScenarioType
    
    # Basic configuration
    num_processors: int = 4
    num_processes: int = 20
    time_quantum: int = 4
    
    # Advanced configuration
    workload_pattern: WorkloadPattern = WorkloadPattern.UNIFORM
    process_types: Dict[ProcessType, float] = field(default_factory=lambda: {
        ProcessType.CPU_BOUND: 0.3,
        ProcessType.IO_BOUND: 0.3,
        ProcessType.MIXED: 0.4
    })
    
    # Scheduling configuration
    scheduling_policy: SchedulingPolicy = SchedulingPolicy.ROUND_ROBIN
    enable_preemption: bool = True
    
    # Load balancing
    algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.LEAST_LOADED
    
    # Performance targets
    target_throughput: Optional[float] = None
    target_avg_turnaround: Optional[float] = None
    max_response_time: Optional[float] = None
    sla_deadline_ratio: float = 0.95
    
    def to_sim_config(self) -> SimulationConfig:
        """Convert scenario to standard simulation config"""
        return SimulationConfig(
            num_processors=self.num_processors,
            num_processes=self.num_processes,
            time_quantum=self.time_quantum
        )


# Predefined scenarios
PREDEFINED_SCENARIOS = {
    ScenarioType.BASIC: SimulationScenario(
        name="Basic Simulation",
        description="Standard simulation with default settings",
        scenario_type=ScenarioType.BASIC,
        num_processors=4,
        num_processes=20,
        workload_pattern=WorkloadPattern.UNIFORM,
        algorithm=LoadBalancingAlgorithm.ROUND_ROBIN
    ),
    ScenarioType.CPU_INTENSIVE: SimulationScenario(
        name="CPU Intensive",
        description="Heavy computation workload with long-running processes",
        scenario_type=ScenarioType.CPU_INTENSIVE,
        num_processors=8,
        num_processes=30,
        time_quantum=8,
        workload_pattern=WorkloadPattern.UNIFORM,
        process_types={ProcessType.CPU_BOUND: 0.8, ProcessType.MIXED: 0.2},
        algorithm=LoadBalancingAlgorithm.LEAST_LOADED
    ),
    ScenarioType.IO_INTENSIVE: SimulationScenario(
        name="I/O Intensive",
        description="I/O bound processes with frequent blocking",
        scenario_type=ScenarioType.IO_INTENSIVE,
        num_processors=4,
        num_processes=40,
        time_quantum=2,
        workload_pattern=WorkloadPattern.BURSTY,
        process_types={ProcessType.IO_BOUND: 0.7, ProcessType.MIXED: 0.3},
        algorithm=LoadBalancingAlgorithm.THRESHOLD_BASED
    ),
    ScenarioType.MIXED_WORKLOAD: SimulationScenario(
        name="Mixed Workload",
        description="Diverse process types simulating real-world scenarios",
        scenario_type=ScenarioType.MIXED_WORKLOAD,
        num_processors=6,
        num_processes=50,
        time_quantum=4,
        workload_pattern=WorkloadPattern.DIURNAL,
        process_types={
            ProcessType.CPU_BOUND: 0.2,
            ProcessType.IO_BOUND: 0.2,
            ProcessType.MIXED: 0.3,
            ProcessType.INTERACTIVE: 0.2,
            ProcessType.BATCH: 0.1
        },
        algorithm=LoadBalancingAlgorithm.DQN
    ),
    ScenarioType.BURSTY_TRAFFIC: SimulationScenario(
        name="Bursty Traffic",
        description="Sudden spikes in process arrivals",
        scenario_type=ScenarioType.BURSTY_TRAFFIC,
        num_processors=4,
        num_processes=60,
        time_quantum=3,
        workload_pattern=WorkloadPattern.SPIKE,
        algorithm=LoadBalancingAlgorithm.Q_LEARNING
    ),
    ScenarioType.REAL_TIME: SimulationScenario(
        name="Real-Time System",
        description="Hard real-time constraints with strict deadlines",
        scenario_type=ScenarioType.REAL_TIME,
        num_processors=8,
        num_processes=25,
        time_quantum=2,
        workload_pattern=WorkloadPattern.UNIFORM,
        process_types={ProcessType.REAL_TIME: 0.6, ProcessType.INTERACTIVE: 0.4},
        scheduling_policy=SchedulingPolicy.PRIORITY,
        enable_preemption=True,
        sla_deadline_ratio=0.99,
        algorithm=LoadBalancingAlgorithm.LEAST_LOADED
    ),
    ScenarioType.STRESS_TEST: SimulationScenario(
        name="Stress Test",
        description="Maximum load to test system limits",
        scenario_type=ScenarioType.STRESS_TEST,
        num_processors=4,
        num_processes=100,
        time_quantum=2,
        workload_pattern=WorkloadPattern.SPIKE,
        algorithm=LoadBalancingAlgorithm.THRESHOLD_BASED
    )
}


# =============================================================================
# INTEGRATED SIMULATION MANAGER
# =============================================================================

class IntegratedSimulationManager:
    """
    Unified manager for both basic and enhanced simulation modes.
    
    This class provides a seamless interface for the GUI to interact with
    either the standard SimulationEngine or the EnhancedSimulationEngine,
    while maintaining backward compatibility.
    """
    
    def __init__(self, use_enhanced: bool = False):
        self.use_enhanced = use_enhanced
        self.engine = None
        self.metrics_collector = None
        self.current_scenario: Optional[SimulationScenario] = None
        
        # Callbacks
        self.on_step: Optional[Callable[[Dict], None]] = None
        self.on_complete: Optional[Callable[[Dict], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        
        # State
        self.is_running = False
        self.is_paused = False
        self._simulation_thread: Optional[threading.Thread] = None
        
        # Performance tracking
        self.step_times: List[float] = []
        self.total_steps = 0
        
    def load_scenario(self, scenario: SimulationScenario):
        """Load a simulation scenario"""
        self.current_scenario = scenario
        logger.info(f"Loaded scenario: {scenario.name}")
        
    def load_predefined_scenario(self, scenario_type: ScenarioType):
        """Load a predefined scenario"""
        if scenario_type in PREDEFINED_SCENARIOS:
            self.load_scenario(PREDEFINED_SCENARIOS[scenario_type])
        else:
            raise ValueError(f"Unknown scenario type: {scenario_type}")
    
    def initialize(self, config: Optional[SimulationConfig] = None,
                   algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.ROUND_ROBIN) -> bool:
        """
        Initialize the simulation engine.
        
        Args:
            config: Simulation configuration (uses scenario config if None)
            algorithm: Load balancing algorithm to use
            
        Returns:
            bool: True if initialization successful
        """
        try:
            # Use scenario config if available
            if self.current_scenario:
                algorithm = self.current_scenario.algorithm
                
                if self.use_enhanced:
                    # Create enhanced config as a SimulationConfig (compatible)
                    enhanced_config = self.current_scenario.to_sim_config()
                    from config import DEFAULT_GUI_CONFIG
                    self.engine = EnhancedSimulationEngine(enhanced_config, DEFAULT_GUI_CONFIG)
                    
                    # Set workload pattern before initialization
                    self.engine.workload_pattern = self.current_scenario.workload_pattern
                    
                    # Initialize with algorithm and process mix
                    self.engine.initialize(
                        algorithm=algorithm,
                        process_mix=self.current_scenario.process_types
                    )
                    
                else:
                    # Standard engine
                    standard_config = SimulationConfig(
                        num_processors=self.current_scenario.num_processors,
                        num_processes=self.current_scenario.num_processes,
                        time_quantum=self.current_scenario.time_quantum
                    )
                    from config import DEFAULT_GUI_CONFIG
                    self.engine = SimulationEngine(standard_config, DEFAULT_GUI_CONFIG)
                    self.engine.initialize(algorithm=algorithm)
                    
            elif config:
                from config import DEFAULT_GUI_CONFIG
                self.engine = SimulationEngine(config, DEFAULT_GUI_CONFIG)
                self.engine.initialize(algorithm=algorithm)
            else:
                raise ValueError("Either config or scenario must be provided")
            
            # Initialize metrics collector
            if self.use_enhanced:
                self.metrics_collector = AdvancedMetricsCalculator()
            
            self.step_times.clear()
            self.total_steps = 0
            
            logger.info(f"Simulation initialized (enhanced={self.use_enhanced})")
            return True
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            import traceback
            traceback.print_exc()
            if self.on_error:
                self.on_error(str(e))
            return False
    
    def start(self):
        """Start the simulation in a background thread"""
        if self.is_running or not self.engine:
            return
            
        self.is_running = True
        self.is_paused = False
        
        self._simulation_thread = threading.Thread(
            target=self._run_simulation,
            daemon=True
        )
        self._simulation_thread.start()
        logger.info("Simulation started")
    
    def _run_simulation(self):
        """Run simulation loop in background thread"""
        try:
            while self.is_running:
                if self.is_paused:
                    time.sleep(0.1)
                    continue
                
                # Check completion
                if self.use_enhanced:
                    if self.engine.is_complete():
                        break
                else:
                    if self.engine.is_complete():
                        break
                
                # Execute step
                start_time = time.perf_counter()
                
                if self.use_enhanced:
                    self.engine.step()
                else:
                    self.engine.step()
                
                step_time = time.perf_counter() - start_time
                self.step_times.append(step_time)
                self.total_steps += 1
                
                # Notify callback
                if self.on_step:
                    state = self.get_state()
                    self.on_step(state)
                
                # Configurable delay
                time.sleep(0.01)
            
            # Simulation complete
            if self.on_complete:
                result = self.get_result()
                self.on_complete(result)
                
        except Exception as e:
            logger.error(f"Simulation error: {e}")
            if self.on_error:
                self.on_error(str(e))
        finally:
            self.is_running = False
    
    def pause(self):
        """Pause the simulation"""
        self.is_paused = True
        logger.info("Simulation paused")
    
    def resume(self):
        """Resume the simulation"""
        self.is_paused = False
        logger.info("Simulation resumed")
    
    def stop(self):
        """Stop the simulation"""
        self.is_running = False
        if self._simulation_thread:
            self._simulation_thread.join(timeout=1.0)
        logger.info("Simulation stopped")
    
    def get_state(self) -> Dict[str, Any]:
        """Get current simulation state"""
        if not self.engine:
            return {}
        
        if self.use_enhanced:
            state = self.engine.get_state()
            
            # Add performance metrics
            if self.step_times:
                state['avg_step_time_ms'] = sum(self.step_times) / len(self.step_times) * 1000
                state['steps_per_second'] = 1 / (state['avg_step_time_ms'] / 1000) if state['avg_step_time_ms'] > 0 else 0
            
            return state
        else:
            return self.engine.get_current_state()
    
    def get_result(self) -> Dict[str, Any]:
        """Get simulation results"""
        if not self.engine:
            return {}
        
        if self.use_enhanced:
            result = self.engine.get_result()
            return {
                'simulation_time': result.total_simulation_time,
                'wall_clock_time': result.wall_clock_duration,
                'processes_completed': result.completed_count,
                'total_processes': result.process_count,
                'metrics': result.metrics.to_dict() if result.metrics else {},
                'processor_stats': result.processor_stats,
                'scenario': self.current_scenario.name if self.current_scenario else None,
                'performance': {
                    'total_steps': self.total_steps,
                    'avg_step_time_ms': sum(self.step_times) / len(self.step_times) * 1000 if self.step_times else 0
                }
            }
        else:
            result = self.engine.get_result()
            return {
                'simulation_time': result.total_time,
                'processes_completed': len(result.process_metrics),  # Use process_metrics
                'total_processes': len(result.process_metrics),
                'metrics': asdict(result.system_metrics) if result.system_metrics else {},
                'scenario': self.current_scenario.name if self.current_scenario else None
            }


# =============================================================================
# PERFORMANCE ANALYZER
# =============================================================================

class PerformanceAnalyzer:
    """
    Analyzes simulation performance and generates reports.
    """
    
    @staticmethod
    def compare_algorithms(config: SimulationConfig, 
                           algorithms: List[LoadBalancingAlgorithm],
                           runs: int = 3) -> Dict[str, Dict]:
        """
        Compare multiple algorithms on the same configuration.
        
        Args:
            config: Base simulation configuration
            algorithms: List of algorithms to compare
            runs: Number of runs per algorithm for averaging
            
        Returns:
            Dict mapping algorithm names to averaged metrics
        """
        results = {}
        
        for algo in algorithms:
            algo_results = []
            
            for run in range(runs):
                manager = IntegratedSimulationManager(use_enhanced=True)
                
                # Create scenario from config
                scenario = SimulationScenario(
                    name=f"{algo.value}_test",
                    description="Algorithm comparison test",
                    scenario_type=ScenarioType.BASIC,
                    num_processors=config.num_processors,
                    num_processes=config.num_processes,
                    time_quantum=config.time_quantum,
                    algorithm=algo
                )
                
                manager.load_scenario(scenario)
                manager.initialize()
                
                # Run synchronously for comparison
                while not manager.engine.is_complete():
                    manager.engine.step()
                
                result = manager.get_result()
                algo_results.append(result['metrics'])
            
            # Average the results
            averaged = {}
            for key in algo_results[0].keys():
                values = [r.get(key, 0) for r in algo_results if isinstance(r.get(key), (int, float))]
                if values:
                    averaged[key] = sum(values) / len(values)
            
            results[algo.value] = averaged
        
        return results
    
    @staticmethod
    def analyze_scenario(scenario: SimulationScenario, 
                         runs: int = 5) -> Dict[str, Any]:
        """
        Analyze a scenario's performance characteristics.
        
        Args:
            scenario: Scenario to analyze
            runs: Number of runs for statistical analysis
            
        Returns:
            Dict containing performance analysis
        """
        all_metrics = []
        
        for _ in range(runs):
            manager = IntegratedSimulationManager(use_enhanced=True)
            manager.load_scenario(scenario)
            manager.initialize()
            
            while not manager.engine.is_complete():
                manager.engine.step()
            
            result = manager.get_result()
            all_metrics.append(result['metrics'])
        
        # Statistical analysis
        analysis = {
            'scenario': scenario.name,
            'runs': runs,
            'metrics': {}
        }
        
        for key in all_metrics[0].keys():
            values = [m.get(key, 0) for m in all_metrics if isinstance(m.get(key), (int, float))]
            if values:
                analysis['metrics'][key] = {
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'std': (sum((v - sum(values)/len(values))**2 for v in values) / len(values)) ** 0.5
                }
        
        return analysis


# =============================================================================
# SCENARIO BUILDER
# =============================================================================

class ScenarioBuilder:
    """
    Builder pattern for creating custom scenarios.
    """
    
    def __init__(self, name: str):
        self._scenario = SimulationScenario(
            name=name,
            description="Custom scenario",
            scenario_type=ScenarioType.CUSTOM
        )
    
    def with_processors(self, count: int) -> 'ScenarioBuilder':
        self._scenario.num_processors = count
        return self
    
    def with_processes(self, count: int) -> 'ScenarioBuilder':
        self._scenario.num_processes = count
        return self
    
    def with_time_quantum(self, quantum: int) -> 'ScenarioBuilder':
        self._scenario.time_quantum = quantum
        return self
    
    def with_workload(self, pattern: WorkloadPattern) -> 'ScenarioBuilder':
        self._scenario.workload_pattern = pattern
        return self
    
    def with_process_mix(self, mix: Dict[ProcessType, float]) -> 'ScenarioBuilder':
        self._scenario.process_types = mix
        return self
    
    def with_algorithm(self, algo: LoadBalancingAlgorithm) -> 'ScenarioBuilder':
        self._scenario.algorithm = algo
        return self
    
    def with_scheduling(self, policy: SchedulingPolicy) -> 'ScenarioBuilder':
        self._scenario.scheduling_policy = policy
        return self
    
    def with_preemption(self, enabled: bool) -> 'ScenarioBuilder':
        self._scenario.enable_preemption = enabled
        return self
    
    def with_description(self, desc: str) -> 'ScenarioBuilder':
        self._scenario.description = desc
        return self
    
    def with_sla_target(self, ratio: float) -> 'ScenarioBuilder':
        self._scenario.sla_deadline_ratio = ratio
        return self
    
    def build(self) -> SimulationScenario:
        return self._scenario


# =============================================================================
# DEMO AND TESTING
# =============================================================================

def run_integration_demo():
    """Demonstrate the integrated simulation system"""
    print("\n" + "="*70)
    print("INTEGRATED SIMULATION SYSTEM DEMO")
    print("="*70)
    
    # 1. Standard simulation with manager
    print("\n1. Testing Standard Mode...")
    manager = IntegratedSimulationManager(use_enhanced=False)
    
    config = SimulationConfig(num_processors=4, num_processes=15)
    manager.initialize(config, LoadBalancingAlgorithm.ROUND_ROBIN)
    
    # Run synchronously for demo
    steps = 0
    while not manager.engine.is_complete() and steps < 100:
        manager.engine.step()
        steps += 1
    
    result = manager.get_result()
    print(f"  Completed: {result['processes_completed']}/{result['total_processes']}")
    print(f"  Time: {result['simulation_time']}")
    
    # 2. Enhanced simulation with scenario
    print("\n2. Testing Enhanced Mode with Scenario...")
    manager2 = IntegratedSimulationManager(use_enhanced=True)
    manager2.load_predefined_scenario(ScenarioType.MIXED_WORKLOAD)
    manager2.initialize()
    
    steps = 0
    while not manager2.engine.is_complete() and steps < 200:
        manager2.engine.step()
        steps += 1
    
    result2 = manager2.get_result()
    print(f"  Scenario: {result2['scenario']}")
    print(f"  Completed: {result2['processes_completed']}/{result2['total_processes']}")
    if 'avg_turnaround' in result2['metrics']:
        print(f"  Avg Turnaround: {result2['metrics']['avg_turnaround']:.2f}")
    
    # 3. Custom scenario
    print("\n3. Testing Custom Scenario Builder...")
    custom = (ScenarioBuilder("High Performance Test")
              .with_processors(8)
              .with_processes(40)
              .with_workload(WorkloadPattern.BURSTY)
              .with_algorithm(LoadBalancingAlgorithm.DQN)
              .with_description("Custom high-performance test scenario")
              .build())
    
    print(f"  Built scenario: {custom.name}")
    print(f"  Processors: {custom.num_processors}")
    print(f"  Algorithm: {custom.algorithm.value}")
    
    # 4. Quick comparison
    print("\n4. Algorithm Quick Comparison...")
    print("  Comparing Round Robin vs Least Loaded...")
    
    config = SimulationConfig(num_processors=4, num_processes=20)
    
    for algo in [LoadBalancingAlgorithm.ROUND_ROBIN, LoadBalancingAlgorithm.LEAST_LOADED]:
        manager = IntegratedSimulationManager(use_enhanced=True)
        scenario = SimulationScenario(
            name=f"{algo.value}_test",
            description="Comparison test",
            scenario_type=ScenarioType.BASIC,
            num_processors=4,
            num_processes=20,
            algorithm=algo
        )
        manager.load_scenario(scenario)
        manager.initialize()
        
        steps = 0
        while not manager.engine.is_complete() and steps < 200:
            manager.engine.step()
            steps += 1
        
        result = manager.get_result()
        metrics = result['metrics']
        print(f"  {algo.value}:")
        print(f"    Turnaround: {metrics.get('avg_turnaround', 0):.2f}")
        print(f"    Throughput: {metrics.get('throughput', 0):.4f}")
    
    print("\n" + "="*70)
    print("Integration Demo Complete!")
    print("="*70)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_integration_demo()
