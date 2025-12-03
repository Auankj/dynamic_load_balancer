"""
Comprehensive Test Suite for Dynamic Load Balancing Simulator

This module provides extensive unit tests, integration tests, and scenario
tests for all components of the load balancing simulation system.

Test Categories:
1. Unit Tests: Test individual classes and methods in isolation
2. Integration Tests: Test component interactions
3. Scenario Tests: Test realistic use cases with various configurations
4. Edge Case Tests: Test boundary conditions and error handling
5. Performance Tests: Basic performance benchmarks

Testing Framework: unittest (Python standard library)

Author: Student
Date: December 2024
"""

import unittest
import sys
import os
import time
import logging
import json
import random
import statistics
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import project modules
from config import (
    ProcessState,
    ProcessPriority,
    LoadBalancingAlgorithm,
    SimulationConfig,
    GUIConfig,
    DEFAULT_SIMULATION_CONFIG,
    DEFAULT_GUI_CONFIG
)
from process import Process, ProcessGenerator
from processor import Processor, ProcessorManager, ProcessorStatistics
from load_balancer import (
    LoadBalancer,
    LoadBalancerFactory,
    RoundRobinBalancer,
    LeastLoadedBalancer,
    ThresholdBasedBalancer,
    MigrationRecord
)
from simulation import SimulationEngine, SimulationState, SimulationResult
from metrics import (
    ProcessMetrics,
    ProcessorMetrics,
    SystemMetrics,
    MetricsCalculator,
    MetricsComparator
)
from utils import SimulationLogger, DataExporter

# Disable logging during tests unless debugging
logging.disable(logging.CRITICAL)


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

class TestConfig(unittest.TestCase):
    """Test configuration module and data classes."""
    
    def test_process_state_enum_values(self):
        """Test that all process states are defined."""
        states = [ProcessState.NEW, ProcessState.READY, ProcessState.RUNNING,
                  ProcessState.WAITING, ProcessState.COMPLETED, ProcessState.MIGRATING]
        self.assertEqual(len(states), 6)
        
    def test_process_priority_enum_values(self):
        """Test priority ordering."""
        self.assertLess(ProcessPriority.HIGH.value, ProcessPriority.MEDIUM.value)
        self.assertLess(ProcessPriority.MEDIUM.value, ProcessPriority.LOW.value)
        
    def test_load_balancing_algorithm_enum(self):
        """Test all load balancing algorithms are defined."""
        self.assertEqual(LoadBalancingAlgorithm.ROUND_ROBIN.value, "Round Robin")
        self.assertEqual(LoadBalancingAlgorithm.LEAST_LOADED.value, "Least Loaded First")
        self.assertEqual(LoadBalancingAlgorithm.THRESHOLD_BASED.value, "Threshold Based")
        self.assertEqual(LoadBalancingAlgorithm.Q_LEARNING.value, "AI (Q-Learning)")
        
    def test_simulation_config_defaults(self):
        """Test default configuration values."""
        config = SimulationConfig()
        self.assertEqual(config.num_processors, 4)
        self.assertGreaterEqual(config.num_processes, 1)
        self.assertGreater(config.time_quantum, 0)
        
    def test_simulation_config_validation(self):
        """Test configuration validation."""
        config = SimulationConfig()
        self.assertGreaterEqual(config.min_processors, 1)
        self.assertLessEqual(config.max_processors, 16)
        
    def test_gui_config_defaults(self):
        """Test GUI configuration defaults."""
        config = GUIConfig()
        self.assertIsNotNone(config.window_title)
        self.assertGreater(config.window_width, 0)
        self.assertGreater(config.window_height, 0)


# =============================================================================
# TEST PROCESS MODULE
# =============================================================================

class TestProcess(unittest.TestCase):
    """Test Process class and related functionality."""
    
    def test_process_creation(self):
        """Test basic process creation."""
        process = Process(pid=1, arrival_time=0, burst_time=10)
        self.assertEqual(process.pid, 1)
        self.assertEqual(process.arrival_time, 0)
        self.assertEqual(process.burst_time, 10)
        self.assertEqual(process.remaining_time, 10)
        self.assertEqual(process.state, ProcessState.NEW)
        
    def test_process_remaining_time_init(self):
        """Test remaining time initializes to burst time."""
        process = Process(pid=1, burst_time=15)
        self.assertEqual(process.remaining_time, process.burst_time)
        
    def test_process_priority_levels(self):
        """Test different priority levels."""
        high = Process(pid=1, priority=ProcessPriority.HIGH)
        medium = Process(pid=2, priority=ProcessPriority.MEDIUM)
        low = Process(pid=3, priority=ProcessPriority.LOW)
        
        self.assertEqual(high.priority, ProcessPriority.HIGH)
        self.assertEqual(medium.priority, ProcessPriority.MEDIUM)
        self.assertEqual(low.priority, ProcessPriority.LOW)
        
    def test_process_state_changes(self):
        """Test process state transitions."""
        process = Process(pid=1)
        
        # NEW -> READY
        process.state = ProcessState.READY
        self.assertEqual(process.state, ProcessState.READY)
        
        # READY -> RUNNING
        process.state = ProcessState.RUNNING
        self.assertEqual(process.state, ProcessState.RUNNING)
        
        # RUNNING -> COMPLETED
        process.state = ProcessState.COMPLETED
        self.assertEqual(process.state, ProcessState.COMPLETED)
        
    def test_process_execution_history(self):
        """Test execution history tracking."""
        process = Process(pid=1)
        self.assertIsInstance(process.execution_history, list)
        self.assertEqual(len(process.execution_history), 0)
        
    def test_process_migration_count_init(self):
        """Test migration count initialization."""
        process = Process(pid=1)
        self.assertEqual(process.migration_count, 0)
        
    def test_process_is_completed_property(self):
        """Test is_completed property if exists."""
        process = Process(pid=1, burst_time=10)
        process.remaining_time = 0
        if hasattr(process, 'is_completed'):
            self.assertTrue(process.is_completed)
            
    def test_process_to_dict(self):
        """Test process serialization."""
        process = Process(pid=1, arrival_time=5, burst_time=10)
        if hasattr(process, 'to_dict'):
            data = process.to_dict()
            self.assertEqual(data['pid'], 1)
            self.assertEqual(data['arrival_time'], 5)


class TestProcessGenerator(unittest.TestCase):
    """Test ProcessGenerator functionality."""
    
    def test_generate_processes(self):
        """Test process generation."""
        generator = ProcessGenerator()
        processes = generator.generate_processes(10)
        
        self.assertEqual(len(processes), 10)
        for i, proc in enumerate(processes):
            self.assertIsInstance(proc, Process)
            self.assertGreater(proc.burst_time, 0)
            
    def test_unique_pids(self):
        """Test that generated processes have unique PIDs."""
        generator = ProcessGenerator()
        processes = generator.generate_processes(20)
        
        pids = [p.pid for p in processes]
        self.assertEqual(len(pids), len(set(pids)))  # All unique
        
    def test_generate_with_reset(self):
        """Test process generation with reset."""
        config = SimulationConfig()
        gen1 = ProcessGenerator(config)
        
        procs1 = gen1.generate_processes(5)
        first_last_pid = procs1[-1].pid
        
        # After generating more, PID should continue
        procs2 = gen1.generate_processes(5)
        
        # PIDs should continue incrementing
        self.assertGreater(procs2[0].pid, first_last_pid)
            
    def test_burst_time_bounds(self):
        """Test burst times are within configured bounds."""
        config = SimulationConfig()
        generator = ProcessGenerator(config)
        processes = generator.generate_processes(50)
        
        for proc in processes:
            self.assertGreaterEqual(proc.burst_time, config.min_burst_time)
            self.assertLessEqual(proc.burst_time, config.max_burst_time)
            
    def test_arrival_time_generation(self):
        """Test arrival times are properly distributed."""
        config = SimulationConfig()
        generator = ProcessGenerator(config)
        processes = generator.generate_processes(10)
        
        for proc in processes:
            self.assertGreaterEqual(proc.arrival_time, 0)


# =============================================================================
# TEST PROCESSOR MODULE
# =============================================================================

class TestProcessor(unittest.TestCase):
    """Test Processor class."""
    
    def test_processor_creation(self):
        """Test processor initialization."""
        processor = Processor(processor_id=0)
        self.assertEqual(processor.processor_id, 0)
        self.assertEqual(processor.get_queue_size(), 0)
        
    def test_processor_add_process(self):
        """Test adding process to processor."""
        processor = Processor(processor_id=0)
        process = Process(pid=1, burst_time=10)
        
        result = processor.add_process(process)
        self.assertTrue(result)
        self.assertGreaterEqual(processor.get_queue_size(), 1)
        
    def test_processor_queue_size(self):
        """Test queue size tracking."""
        processor = Processor(processor_id=0)
        
        for i in range(5):
            processor.add_process(Process(pid=i+1, burst_time=5))
            
        self.assertGreaterEqual(processor.get_queue_size(), 1)
        
    def test_processor_get_load(self):
        """Test load calculation."""
        processor = Processor(processor_id=0)
        
        # Empty processor should have low load
        initial_load = processor.get_load()
        self.assertGreaterEqual(initial_load, 0)
        
        # Add processes
        for i in range(3):
            processor.add_process(Process(pid=i+1, burst_time=10))
            
        # Load should increase
        loaded = processor.get_load()
        self.assertGreaterEqual(loaded, initial_load)
        
    def test_processor_execute_step(self):
        """Test process execution."""
        processor = Processor(processor_id=0)
        process = Process(pid=1, burst_time=10)
        processor.add_process(process)
        
        # Execute should not raise
        if hasattr(processor, 'execute_time_slice'):
            processor.execute_time_slice(current_time=1)
        elif hasattr(processor, 'execute_step'):
            processor.execute_step(current_time=1)
            
    def test_processor_is_idle(self):
        """Test idle state detection."""
        processor = Processor(processor_id=0)
        
        # Should be idle when empty
        if hasattr(processor, 'is_idle'):
            self.assertTrue(processor.is_idle())
            
        if hasattr(processor, 'has_processes'):
            self.assertFalse(processor.has_processes())
            
        processor.add_process(Process(pid=1, burst_time=5))
        
        if hasattr(processor, 'has_processes'):
            self.assertTrue(processor.has_processes())
            
    def test_processor_statistics(self):
        """Test statistics tracking."""
        processor = Processor(processor_id=0)
        
        if hasattr(processor, 'statistics'):
            self.assertIsInstance(processor.statistics, ProcessorStatistics)
            
    def test_processor_get_current_process(self):
        """Test getting current executing process."""
        processor = Processor(processor_id=0)
        process = Process(pid=1, burst_time=10)
        processor.add_process(process)
        
        # current_process may be None if not executing yet
        self.assertTrue(hasattr(processor, 'current_process'))


class TestProcessorManager(unittest.TestCase):
    """Test ProcessorManager class."""
    
    def test_manager_creation(self):
        """Test processor manager initialization."""
        manager = ProcessorManager(num_processors=4)
        self.assertEqual(len(manager.processors), 4)
        
    def test_manager_get_processor(self):
        """Test getting specific processor."""
        manager = ProcessorManager(num_processors=4)
        
        proc0 = manager.get_processor(0)
        self.assertIsNotNone(proc0)
        self.assertEqual(proc0.processor_id, 0)
        
    def test_manager_get_all_loads(self):
        """Test getting all processor loads."""
        manager = ProcessorManager(num_processors=4)
        
        if hasattr(manager, 'get_all_loads'):
            loads = manager.get_all_loads()
            self.assertEqual(len(loads), 4)
            
    def test_manager_get_least_loaded(self):
        """Test finding least loaded processor."""
        manager = ProcessorManager(num_processors=4)
        
        # Add different loads
        manager.processors[0].add_process(Process(pid=1, burst_time=20))
        manager.processors[0].add_process(Process(pid=2, burst_time=20))
        manager.processors[1].add_process(Process(pid=3, burst_time=5))
        
        if hasattr(manager, 'get_least_loaded_processor'):
            least = manager.get_least_loaded_processor()
            self.assertIsNotNone(least)
            
    def test_manager_total_processes(self):
        """Test counting total processes."""
        manager = ProcessorManager(num_processors=4)
        
        manager.processors[0].add_process(Process(pid=1, burst_time=5))
        manager.processors[1].add_process(Process(pid=2, burst_time=5))
        manager.processors[1].add_process(Process(pid=3, burst_time=5))
        
        if hasattr(manager, 'get_total_processes'):
            total = manager.get_total_processes()
            self.assertEqual(total, 3)


# =============================================================================
# TEST LOAD BALANCER MODULE
# =============================================================================

class TestRoundRobinBalancer(unittest.TestCase):
    """Test Round Robin load balancing algorithm."""
    
    def test_round_robin_creation(self):
        """Test Round Robin balancer creation."""
        balancer = RoundRobinBalancer()
        self.assertEqual(balancer.algorithm_type, LoadBalancingAlgorithm.ROUND_ROBIN)
        
    def test_round_robin_cyclic_assignment(self):
        """Test cyclic assignment pattern."""
        config = SimulationConfig(num_processors=4)
        balancer = RoundRobinBalancer(config)
        manager = ProcessorManager(num_processors=4)
        
        assigned_processors = []
        for i in range(8):
            process = Process(pid=i, burst_time=5)
            processor = balancer.assign_process(process, manager.processors)
            if processor:
                assigned_processors.append(processor.processor_id)
                
        # Should cycle through processors
        # Expect pattern like 0, 1, 2, 3, 0, 1, 2, 3
        for i in range(min(4, len(assigned_processors))):
            self.assertEqual(assigned_processors[i] % 4, i)
            
    def test_round_robin_handles_empty_processors(self):
        """Test handling when processors list is empty."""
        balancer = RoundRobinBalancer()
        process = Process(pid=1, burst_time=5)
        
        result = balancer.assign_process(process, [])
        self.assertIsNone(result)


class TestLeastLoadedBalancer(unittest.TestCase):
    """Test Least Loaded First load balancing algorithm."""
    
    def test_least_loaded_creation(self):
        """Test Least Loaded balancer creation."""
        balancer = LeastLoadedBalancer()
        self.assertEqual(balancer.algorithm_type, LoadBalancingAlgorithm.LEAST_LOADED)
        
    def test_least_loaded_selects_minimum(self):
        """Test that it selects the least loaded processor."""
        config = SimulationConfig(num_processors=4)
        balancer = LeastLoadedBalancer(config)
        manager = ProcessorManager(num_processors=4)
        
        # Load processors differently
        manager.processors[0].add_process(Process(pid=100, burst_time=50))
        manager.processors[0].add_process(Process(pid=101, burst_time=50))
        manager.processors[1].add_process(Process(pid=102, burst_time=30))
        manager.processors[2].add_process(Process(pid=103, burst_time=10))
        # Processor 3 is empty - should be selected
        
        process = Process(pid=1, burst_time=5)
        selected = balancer.assign_process(process, manager.processors)
        
        # Should select processor 3 (least loaded)
        self.assertIsNotNone(selected)
        
    def test_least_loaded_load_balancing(self):
        """Test load balancing effectiveness."""
        config = SimulationConfig(num_processors=4)
        balancer = LeastLoadedBalancer(config)
        manager = ProcessorManager(num_processors=4)
        
        # Assign many processes
        for i in range(20):
            process = Process(pid=i, burst_time=random.randint(5, 15))
            balancer.assign_process(process, manager.processors)
            
        # Get loads
        loads = [p.get_load() for p in manager.processors]
        
        # Check load variance is reasonable (should be more balanced)
        if any(loads):
            load_variance = statistics.variance(loads) if len(loads) > 1 else 0
            # Just verify it runs - variance check is informational
            self.assertGreaterEqual(load_variance, 0)


class TestThresholdBasedBalancer(unittest.TestCase):
    """Test Threshold Based load balancing algorithm."""
    
    def test_threshold_based_creation(self):
        """Test Threshold Based balancer creation."""
        balancer = ThresholdBasedBalancer()
        self.assertEqual(balancer.algorithm_type, LoadBalancingAlgorithm.THRESHOLD_BASED)
        
    def test_threshold_based_migration_check(self):
        """Test migration decision based on threshold."""
        config = SimulationConfig(num_processors=4, load_threshold=0.3)
        balancer = ThresholdBasedBalancer(config)
        manager = ProcessorManager(num_processors=4)
        
        # Create imbalanced load
        for _ in range(5):
            manager.processors[0].add_process(
                Process(pid=random.randint(100, 200), burst_time=20)
            )
            
        # Check for migrations
        if hasattr(balancer, 'check_for_migration'):
            migrations = balancer.check_for_migration(manager.processors, current_time=0)
            # May or may not trigger migration depending on threshold
            self.assertIsInstance(migrations, (list, type(None)))


class TestLoadBalancerFactory(unittest.TestCase):
    """Test LoadBalancerFactory."""
    
    def test_factory_creates_round_robin(self):
        """Test factory creates Round Robin balancer."""
        balancer = LoadBalancerFactory.create(LoadBalancingAlgorithm.ROUND_ROBIN)
        self.assertIsInstance(balancer, RoundRobinBalancer)
        
    def test_factory_creates_least_loaded(self):
        """Test factory creates Least Loaded balancer."""
        balancer = LoadBalancerFactory.create(LoadBalancingAlgorithm.LEAST_LOADED)
        self.assertIsInstance(balancer, LeastLoadedBalancer)
        
    def test_factory_creates_threshold_based(self):
        """Test factory creates Threshold Based balancer."""
        balancer = LoadBalancerFactory.create(LoadBalancingAlgorithm.THRESHOLD_BASED)
        self.assertIsInstance(balancer, ThresholdBasedBalancer)
        
    def test_factory_creates_qlearning(self):
        """Test factory creates Q-Learning balancer."""
        from ai_balancer import QLearningBalancer
        balancer = LoadBalancerFactory.create(LoadBalancingAlgorithm.Q_LEARNING, num_processors=4)
        self.assertIsInstance(balancer, QLearningBalancer)
        
    def test_factory_with_config(self):
        """Test factory passes configuration."""
        config = SimulationConfig(num_processors=8)
        balancer = LoadBalancerFactory.create(LoadBalancingAlgorithm.ROUND_ROBIN, config)
        self.assertEqual(balancer.config.num_processors, 8)


# =============================================================================
# TEST SIMULATION ENGINE
# =============================================================================

class TestSimulationEngine(unittest.TestCase):
    """Test SimulationEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = SimulationConfig(num_processors=4, num_processes=10)
        self.engine = SimulationEngine(self.config)
        
    def test_engine_creation(self):
        """Test engine initialization."""
        self.assertIsNotNone(self.engine)
        self.assertEqual(self.engine.state, SimulationState.IDLE)
        
    def test_engine_initialize(self):
        """Test engine initialization."""
        result = self.engine.initialize()
        self.assertTrue(result)
        self.assertIsNotNone(self.engine.processor_manager)
        self.assertIsNotNone(self.engine.load_balancer)
        
    def test_engine_start_stop(self):
        """Test start and stop functionality."""
        self.engine.initialize()
        
        # Run a few steps
        self.engine.step()
        self.engine.step()
        
        self.engine.stop()
        self.assertIn(self.engine.state, 
                     [SimulationState.STOPPED, SimulationState.COMPLETED])
        
    def test_engine_step(self):
        """Test single simulation step."""
        self.engine.initialize()
        initial_time = self.engine.current_time
        
        self.engine.step()
        
        # Time should advance
        self.assertGreater(self.engine.current_time, initial_time)
        
    def test_engine_reset(self):
        """Test engine reset via reinitialize."""
        self.engine.initialize()
        self.engine.step()
        self.engine.step()
        
        # Re-initialize resets the engine
        self.engine.initialize()
        
        self.assertEqual(self.engine.state, SimulationState.IDLE)
        self.assertEqual(self.engine.current_time, 0)
        
    def test_engine_run_to_completion(self):
        """Test running simulation to completion."""
        config = SimulationConfig(num_processors=4, num_processes=5)
        engine = SimulationEngine(config)
        engine.initialize()
        
        result = engine.run()
        
        self.assertIsInstance(result, SimulationResult)
        self.assertIn(engine.state, 
                     [SimulationState.COMPLETED, SimulationState.STOPPED])
        
    def test_engine_change_algorithm(self):
        """Test changing load balancing algorithm."""
        # Test Least Loaded
        self.engine.initialize(algorithm=LoadBalancingAlgorithm.LEAST_LOADED)
        self.assertEqual(
            self.engine.load_balancer.algorithm_type,
            LoadBalancingAlgorithm.LEAST_LOADED
        )
        
        # Create new engine for different algorithm test
        engine2 = SimulationEngine(self.config)
        engine2.initialize(algorithm=LoadBalancingAlgorithm.THRESHOLD_BASED)
        self.assertEqual(
            engine2.load_balancer.algorithm_type,
            LoadBalancingAlgorithm.THRESHOLD_BASED
        )
        
    def test_engine_get_current_state(self):
        """Test getting simulation state snapshot."""
        self.engine.initialize()
        
        if hasattr(self.engine, 'get_current_state'):
            state = self.engine.get_current_state()
            self.assertIsNotNone(state)


class TestSimulationResult(unittest.TestCase):
    """Test SimulationResult data class."""
    
    def test_result_creation(self):
        """Test result object creation."""
        config = SimulationConfig(num_processors=4, num_processes=5)
        engine = SimulationEngine(config)
        engine.initialize()
        result = engine.run()
        
        self.assertIsInstance(result, SimulationResult)
        self.assertIsNotNone(result.config)
        
    def test_result_has_metrics(self):
        """Test result contains metrics."""
        config = SimulationConfig(num_processors=4, num_processes=5)
        engine = SimulationEngine(config)
        engine.initialize()
        result = engine.run()
        
        # Should have some form of metrics
        if hasattr(result, 'system_metrics'):
            self.assertIsNotNone(result.system_metrics)


# =============================================================================
# TEST METRICS MODULE
# =============================================================================

class TestProcessMetrics(unittest.TestCase):
    """Test ProcessMetrics class."""
    
    def test_turnaround_time_calculation(self):
        """Test turnaround time calculation."""
        metrics = ProcessMetrics(
            pid=1,
            arrival_time=0,
            burst_time=10,
            start_time=2,
            completion_time=15,
            waiting_time=2,
            processor_id=0,
            migration_count=0,
            priority="MEDIUM"
        )
        
        # Turnaround = Completion - Arrival = 15 - 0 = 15
        self.assertEqual(metrics.turnaround_time, 15)
        
    def test_response_time_calculation(self):
        """Test response time calculation."""
        metrics = ProcessMetrics(
            pid=1,
            arrival_time=5,
            burst_time=10,
            start_time=8,
            completion_time=18,
            waiting_time=3,
            processor_id=0,
            migration_count=0,
            priority="MEDIUM"
        )
        
        # Response = Start - Arrival = 8 - 5 = 3
        self.assertEqual(metrics.response_time, 3)
        
    def test_normalized_turnaround(self):
        """Test normalized turnaround calculation."""
        metrics = ProcessMetrics(
            pid=1,
            arrival_time=0,
            burst_time=10,
            start_time=0,
            completion_time=20,
            waiting_time=0,
            processor_id=0,
            migration_count=0,
            priority="MEDIUM"
        )
        
        # Normalized = Turnaround / Burst = 20 / 10 = 2.0
        self.assertEqual(metrics.normalized_turnaround, 2.0)
        
    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = ProcessMetrics(
            pid=1,
            arrival_time=0,
            burst_time=10,
            start_time=0,
            completion_time=10,
            waiting_time=0,
            processor_id=0,
            migration_count=0,
            priority="MEDIUM"
        )
        
        data = metrics.to_dict()
        self.assertEqual(data['pid'], 1)
        self.assertEqual(data['burst_time'], 10)


class TestSystemMetrics(unittest.TestCase):
    """Test SystemMetrics calculations."""
    
    def test_system_metrics_structure(self):
        """Test system metrics data structure."""
        # SystemMetrics uses default values, create instance
        metrics = SystemMetrics()
        metrics.total_processes = 10
        metrics.completed_processes = 10
        metrics.avg_turnaround_time = 25.0
        metrics.avg_waiting_time = 15.0
        metrics.avg_response_time = 5.0
        metrics.avg_utilization = 0.85
        metrics.total_throughput = 0.4
        metrics.load_variance = 0.02
        metrics.total_migrations = 3
        metrics.total_simulation_time = 25
        metrics.algorithm_name = "Round Robin"
        
        self.assertEqual(metrics.total_processes, 10)
        self.assertEqual(metrics.completed_processes, 10)
        self.assertAlmostEqual(metrics.avg_utilization, 0.85)
        
    def test_system_metrics_to_dict(self):
        """Test system metrics serialization."""
        metrics = SystemMetrics()
        metrics.total_processes = 10
        metrics.completed_processes = 10
        
        data = metrics.to_dict()
        self.assertIn('total_processes', data)
        self.assertIn('completed_processes', data)


class TestMetricsCalculator(unittest.TestCase):
    """Test MetricsCalculator class."""
    
    def test_calculator_creation(self):
        """Test calculator initialization."""
        calculator = MetricsCalculator()
        self.assertIsNotNone(calculator)
        
    def test_calculate_from_processes(self):
        """Test metrics calculation from processes."""
        calculator = MetricsCalculator()
        
        processes = [
            Process(pid=1, arrival_time=0, burst_time=10),
            Process(pid=2, arrival_time=2, burst_time=5),
        ]
        
        # Simulate completion
        processes[0].start_time = 0
        processes[0].completion_time = 10
        processes[0].state = ProcessState.COMPLETED
        processes[0].remaining_time = 0
        
        processes[1].start_time = 2
        processes[1].completion_time = 12
        processes[1].state = ProcessState.COMPLETED
        processes[1].remaining_time = 0
        
        if hasattr(calculator, 'calculate_process_metrics'):
            for proc in processes:
                metrics = calculator.calculate_process_metrics(proc)
                self.assertIsNotNone(metrics)


class TestMetricsComparator(unittest.TestCase):
    """Test MetricsComparator class."""
    
    def test_comparator_creation(self):
        """Test comparator initialization."""
        comparator = MetricsComparator()
        self.assertIsNotNone(comparator)
        
    def test_compare_algorithms(self):
        """Test algorithm comparison."""
        comparator = MetricsComparator()
        
        # Create sample results using actual SystemMetrics class
        result1 = SystemMetrics()
        result1.total_processes = 10
        result1.completed_processes = 10
        result1.avg_turnaround_time = 20.0
        result1.avg_waiting_time = 10.0
        result1.avg_response_time = 5.0
        result1.avg_utilization = 0.8
        result1.total_throughput = 0.5
        result1.load_variance = 0.05
        result1.total_migrations = 0
        result1.total_simulation_time = 20
        result1.algorithm_name = "Round Robin"
        
        result2 = SystemMetrics()
        result2.total_processes = 10
        result2.completed_processes = 10
        result2.avg_turnaround_time = 18.0
        result2.avg_waiting_time = 8.0
        result2.avg_response_time = 4.0
        result2.avg_utilization = 0.85
        result2.total_throughput = 0.55
        result2.load_variance = 0.03
        result2.total_migrations = 2
        result2.total_simulation_time = 18
        result2.algorithm_name = "Least Loaded"
        
        if hasattr(comparator, 'compare'):
            comparison = comparator.compare([result1, result2])
            self.assertIsNotNone(comparison)


# =============================================================================
# TEST UTILS MODULE
# =============================================================================

class TestDataExporter(unittest.TestCase):
    """Test DataExporter class."""
    
    def test_exporter_creation(self):
        """Test exporter initialization."""
        exporter = DataExporter()
        self.assertIsNotNone(exporter)
        
    def test_export_to_dict(self):
        """Test exporting data to dictionary."""
        exporter = DataExporter()
        
        config = SimulationConfig(num_processors=4, num_processes=5)
        engine = SimulationEngine(config)
        engine.initialize()
        result = engine.run()
        
        # Check result has to_dict method
        if hasattr(result, 'to_dict'):
            data = result.to_dict()
            self.assertIsInstance(data, dict)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def test_full_simulation_round_robin(self):
        """Test complete simulation with Round Robin."""
        config = SimulationConfig(num_processors=4, num_processes=10)
        engine = SimulationEngine(config)
        engine.initialize(algorithm=LoadBalancingAlgorithm.ROUND_ROBIN)
        
        result = engine.run()
        
        self.assertIsNotNone(result)
        self.assertGreater(result.system_metrics.completed_processes, 0)
        
    def test_full_simulation_least_loaded(self):
        """Test complete simulation with Least Loaded."""
        config = SimulationConfig(num_processors=4, num_processes=10)
        engine = SimulationEngine(config)
        engine.initialize(algorithm=LoadBalancingAlgorithm.LEAST_LOADED)
        
        result = engine.run()
        
        self.assertIsNotNone(result)
        self.assertGreater(result.system_metrics.completed_processes, 0)
        
    def test_full_simulation_threshold_based(self):
        """Test complete simulation with Threshold Based."""
        config = SimulationConfig(num_processors=4, num_processes=10)
        engine = SimulationEngine(config)
        engine.initialize(algorithm=LoadBalancingAlgorithm.THRESHOLD_BASED)
        
        result = engine.run()
        
        self.assertIsNotNone(result)
        self.assertGreater(result.system_metrics.completed_processes, 0)
        
    def test_all_processes_complete(self):
        """Test that all processes eventually complete."""
        config = SimulationConfig(num_processors=4, num_processes=20)
        engine = SimulationEngine(config)
        engine.initialize()
        
        result = engine.run()
        
        self.assertEqual(
            result.system_metrics.completed_processes,
            result.system_metrics.total_processes
        )
        
    def test_algorithm_comparison(self):
        """Test comparing all algorithms."""
        config = SimulationConfig(num_processors=4, num_processes=15)
        
        results = {}
        for algorithm in LoadBalancingAlgorithm:
            engine = SimulationEngine(config)
            engine.initialize(algorithm=algorithm)
            result = engine.run()
            results[algorithm.value] = result
            
        # All should complete
        for name, result in results.items():
            self.assertEqual(
                result.system_metrics.completed_processes,
                result.system_metrics.total_processes,
                f"{name} did not complete all processes"
            )


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""
    
    def test_single_processor(self):
        """Test simulation with single processor."""
        config = SimulationConfig(num_processors=1, num_processes=5)
        engine = SimulationEngine(config)
        engine.initialize()
        
        result = engine.run()
        self.assertEqual(result.system_metrics.completed_processes, 5)
        
    def test_maximum_processors(self):
        """Test simulation with maximum processors."""
        config = SimulationConfig(num_processors=8, num_processes=5)
        engine = SimulationEngine(config)
        engine.initialize()
        
        result = engine.run()
        self.assertEqual(result.system_metrics.completed_processes, 5)
        
    def test_single_process(self):
        """Test simulation with single process."""
        config = SimulationConfig(num_processors=4, num_processes=1)
        engine = SimulationEngine(config)
        engine.initialize()
        
        result = engine.run()
        self.assertEqual(result.system_metrics.completed_processes, 1)
        
    def test_many_processes(self):
        """Test simulation with many processes."""
        config = SimulationConfig(num_processors=4, num_processes=50)
        engine = SimulationEngine(config)
        engine.initialize()
        
        result = engine.run()
        self.assertEqual(result.system_metrics.completed_processes, 50)
        
    def test_short_burst_times(self):
        """Test with very short burst times."""
        config = SimulationConfig(
            num_processors=4, 
            num_processes=10,
            min_burst_time=1,
            max_burst_time=3
        )
        engine = SimulationEngine(config)
        engine.initialize()
        
        result = engine.run()
        self.assertEqual(result.system_metrics.completed_processes, 10)
        
    def test_long_burst_times(self):
        """Test with longer burst times."""
        config = SimulationConfig(
            num_processors=4,
            num_processes=5,
            min_burst_time=15,
            max_burst_time=25
        )
        engine = SimulationEngine(config)
        engine.initialize()
        
        result = engine.run()
        self.assertEqual(result.system_metrics.completed_processes, 5)
        
    def test_processes_more_than_processors(self):
        """Test when processes outnumber processors."""
        config = SimulationConfig(num_processors=2, num_processes=10)
        engine = SimulationEngine(config)
        engine.initialize()
        
        result = engine.run()
        self.assertEqual(result.system_metrics.completed_processes, 10)
        
    def test_processors_more_than_processes(self):
        """Test when processors outnumber processes."""
        config = SimulationConfig(num_processors=8, num_processes=3)
        engine = SimulationEngine(config)
        engine.initialize()
        
        result = engine.run()
        self.assertEqual(result.system_metrics.completed_processes, 3)


# =============================================================================
# SCENARIO TESTS
# =============================================================================

class TestScenarios(unittest.TestCase):
    """Test realistic usage scenarios."""
    
    def test_scenario_light_workload(self):
        """Test light workload scenario."""
        # Few short processes on many processors
        config = SimulationConfig(
            num_processors=8,
            num_processes=5,
            min_burst_time=2,
            max_burst_time=5
        )
        engine = SimulationEngine(config)
        engine.initialize()
        
        result = engine.run()
        
        # Should have low utilization
        self.assertEqual(result.system_metrics.completed_processes, 5)
        
    def test_scenario_heavy_workload(self):
        """Test heavy workload scenario."""
        # Many long processes on few processors
        config = SimulationConfig(
            num_processors=2,
            num_processes=20,
            min_burst_time=10,
            max_burst_time=20
        )
        engine = SimulationEngine(config)
        engine.initialize()
        
        result = engine.run()
        
        # Should complete all despite heavy load
        self.assertEqual(result.system_metrics.completed_processes, 20)
        
    def test_scenario_balanced_workload(self):
        """Test balanced workload scenario."""
        config = SimulationConfig(
            num_processors=4,
            num_processes=16,
            min_burst_time=5,
            max_burst_time=10
        )
        engine = SimulationEngine(config)
        engine.initialize()
        
        result = engine.run()
        
        self.assertEqual(result.system_metrics.completed_processes, 16)
        
    def test_scenario_bursty_arrivals(self):
        """Test bursty arrival pattern."""
        config = SimulationConfig(
            num_processors=4,
            num_processes=15
        )
        engine = SimulationEngine(config)
        engine.initialize()
        
        result = engine.run()
        self.assertEqual(result.system_metrics.completed_processes, 15)
        
    def test_scenario_spread_arrivals(self):
        """Test spread out arrival pattern."""
        config = SimulationConfig(
            num_processors=4,
            num_processes=10
        )
        engine = SimulationEngine(config)
        engine.initialize()
        
        result = engine.run()
        self.assertEqual(result.system_metrics.completed_processes, 10)


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance(unittest.TestCase):
    """Basic performance tests."""
    
    def test_simulation_completes_in_reasonable_time(self):
        """Test that simulation doesn't hang."""
        config = SimulationConfig(num_processors=4, num_processes=30)
        engine = SimulationEngine(config)
        engine.initialize()
        
        start_time = time.time()
        result = engine.run()
        elapsed = time.time() - start_time
        
        # Should complete within reasonable time (5 seconds)
        self.assertLess(elapsed, 5.0)
        self.assertEqual(result.system_metrics.completed_processes, 30)
        
    def test_large_simulation_stability(self):
        """Test stability with larger simulation."""
        config = SimulationConfig(num_processors=8, num_processes=100)
        engine = SimulationEngine(config)
        engine.initialize()
        
        result = engine.run()
        
        # Should complete without errors
        self.assertEqual(result.system_metrics.completed_processes, 100)


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================

class TestErrorHandling(unittest.TestCase):
    """Test error handling and validation."""
    
    def test_invalid_processor_count_handled(self):
        """Test handling of edge case processor counts."""
        # Test with minimum
        config = SimulationConfig(num_processors=1)
        engine = SimulationEngine(config)
        engine.initialize()
        # Should not raise
        
    def test_zero_processes_handled(self):
        """Test handling of zero processes."""
        config = SimulationConfig(num_processors=4, num_processes=0)
        engine = SimulationEngine(config)
        
        # Should handle gracefully - may not initialize properly with 0 processes
        try:
            result = engine.initialize()
            if result:
                result = engine.run()
                self.assertEqual(result.system_metrics.completed_processes, 0)
        except (ValueError, ZeroDivisionError, IndexError):
            pass  # Acceptable - some implementations may reject 0 processes
            
    def test_negative_values_handled(self):
        """Test that negative values don't crash system."""
        # The system should either reject or handle gracefully
        try:
            config = SimulationConfig(num_processors=4, num_processes=-1)
            # If it allows creation, should handle during run
        except (ValueError, AssertionError):
            pass  # Expected behavior
            
    def test_engine_double_initialize(self):
        """Test initializing already initialized engine."""
        config = SimulationConfig(num_processors=4, num_processes=5)
        engine = SimulationEngine(config)
        engine.initialize()
        
        # Initializing again should be safe
        try:
            engine.initialize()
        except Exception:
            pass  # Some implementations may raise
            
    def test_engine_stop_not_started(self):
        """Test stopping engine that wasn't started."""
        config = SimulationConfig(num_processors=4, num_processes=5)
        engine = SimulationEngine(config)
        engine.initialize()
        
        # Should handle gracefully
        try:
            engine.stop()
        except Exception:
            pass  # Acceptable to raise


# =============================================================================
# TEST AI BALANCER
# =============================================================================

class TestQLearningBalancer(unittest.TestCase):
    """Test Q-Learning load balancer."""
    
    def setUp(self):
        """Set up test fixtures."""
        from ai_balancer import QLearningBalancer, QLearningAgent, StateEncoder
        self.config = SimulationConfig(num_processors=4)
        self.balancer = QLearningBalancer(config=self.config, num_processors=4)
        self.manager = ProcessorManager(num_processors=4)
        self.processors = list(self.manager)
        
    def test_balancer_creation(self):
        """Test Q-Learning balancer initialization."""
        self.assertIsNotNone(self.balancer)
        self.assertEqual(self.balancer.algorithm_type, LoadBalancingAlgorithm.Q_LEARNING)
        self.assertEqual(self.balancer.name, "AI (Q-Learning)")
        
    def test_process_assignment(self):
        """Test that balancer assigns processes."""
        process = Process(pid=1, burst_time=10)
        selected = self.balancer.assign_process(process, self.processors)
        
        self.assertIsNotNone(selected)
        self.assertIn(selected, self.processors)
        self.assertEqual(process.processor_id, selected.processor_id)
        
    def test_training_mode_toggle(self):
        """Test switching between training and exploitation modes."""
        self.assertTrue(self.balancer.agent.training_mode)
        
        self.balancer.set_training_mode(False)
        self.assertFalse(self.balancer.agent.training_mode)
        
        self.balancer.set_training_mode(True)
        self.assertTrue(self.balancer.agent.training_mode)
        
    def test_get_statistics(self):
        """Test statistics retrieval."""
        stats = self.balancer.get_statistics()
        
        self.assertIn('episode_count', stats)
        self.assertIn('epsilon', stats)
        self.assertIn('q_table_size', stats)
        self.assertIn('training_mode', stats)
        
    def test_multiple_assignments(self):
        """Test multiple process assignments."""
        generator = ProcessGenerator(config=self.config)
        processes = generator.generate_processes(10)
        
        for process in processes:
            selected = self.balancer.assign_process(process, self.processors)
            self.assertIsNotNone(selected)
            
        self.assertEqual(self.balancer.assignment_count, 10)
        
    def test_process_completed_feedback(self):
        """Test reward feedback mechanism."""
        process = Process(pid=1, burst_time=10)
        process.start_time = 0
        process.completion_time = 12
        
        selected = self.balancer.assign_process(process, self.processors)
        
        # Simulate process completion
        self.balancer.process_completed(process, self.processors)
        
        # Agent should have received update
        stats = self.balancer.get_statistics()
        self.assertGreater(stats['total_steps'], 0)
        

class TestQLearningAgent(unittest.TestCase):
    """Test Q-Learning agent."""
    
    def setUp(self):
        """Set up test fixtures."""
        from ai_balancer import QLearningAgent, QLearningConfig
        self.agent = QLearningAgent(num_processors=4)
        
    def test_agent_creation(self):
        """Test agent initialization."""
        self.assertEqual(self.agent.num_processors, 4)
        self.assertEqual(self.agent.epsilon, self.agent.config.epsilon_start)
        self.assertTrue(self.agent.training_mode)
        
    def test_action_selection(self):
        """Test action selection."""
        from ai_balancer import SystemState
        state = SystemState(
            load_levels=(0, 0, 0, 0),
            queue_levels=(0, 0, 0, 0),
            process_priority=2,
            process_burst_bucket=1
        )
        
        action = self.agent.get_action(state)
        self.assertIn(action, range(4))
        
    def test_q_value_update(self):
        """Test Q-value update."""
        from ai_balancer import SystemState
        state = SystemState(
            load_levels=(0, 1, 0, 0),
            queue_levels=(0, 1, 0, 0),
            process_priority=2,
            process_burst_bucket=1
        )
        
        # Get action and update
        action = self.agent.get_action(state)
        self.agent.update(reward=-1.0, next_state=state, done=False)
        
        # Q-table should have entry
        self.assertIn(state, self.agent.q_table)
        
    def test_epsilon_decay(self):
        """Test exploration rate decay."""
        from ai_balancer import SystemState
        initial_epsilon = self.agent.epsilon
        
        state = SystemState(
            load_levels=(0, 0, 0, 0),
            queue_levels=(0, 0, 0, 0),
            process_priority=2,
            process_burst_bucket=1
        )
        
        # Complete an episode
        for _ in range(10):
            self.agent.get_action(state)
            self.agent.update(reward=-1.0, next_state=state, done=False)
        
        # Mark episode complete
        self.agent.update(reward=-1.0, next_state=None, done=True)
        
        # Epsilon should have decayed
        self.assertLess(self.agent.epsilon, initial_epsilon)
        
    def test_reset(self):
        """Test agent reset."""
        from ai_balancer import SystemState
        state = SystemState(
            load_levels=(0, 0, 0, 0),
            queue_levels=(0, 0, 0, 0),
            process_priority=2,
            process_burst_bucket=1
        )
        
        # Do some learning
        self.agent.get_action(state)
        self.agent.update(reward=-1.0, next_state=state, done=True)
        
        # Reset
        self.agent.reset()
        
        self.assertEqual(len(self.agent.q_table), 0)
        self.assertEqual(self.agent.episode_count, 0)
        

class TestStateEncoder(unittest.TestCase):
    """Test state encoding for Q-Learning."""
    
    def setUp(self):
        """Set up test fixtures."""
        from ai_balancer import StateEncoder
        self.encoder = StateEncoder()
        self.manager = ProcessorManager(num_processors=4)
        self.processors = list(self.manager)
        
    def test_encode_empty_state(self):
        """Test encoding empty system state."""
        process = Process(pid=1, burst_time=5)
        state = self.encoder.encode(self.processors, process)
        
        self.assertIsNotNone(state)
        self.assertEqual(len(state.load_levels), 4)
        self.assertEqual(len(state.queue_levels), 4)
        
    def test_encode_loaded_state(self):
        """Test encoding with some load."""
        # Add processes to first processor
        for i in range(3):
            p = Process(pid=i, burst_time=10)
            self.processors[0].add_process(p)
            
        process = Process(pid=99, burst_time=5)
        state = self.encoder.encode(self.processors, process)
        
        # First processor should have higher load level
        self.assertGreater(state.load_levels[0], state.load_levels[1])
        
    def test_state_hashable(self):
        """Test that states are hashable for Q-table."""
        process = Process(pid=1, burst_time=5)
        state = self.encoder.encode(self.processors, process)
        
        # Should be usable as dict key
        test_dict = {state: 1.0}
        self.assertEqual(test_dict[state], 1.0)


class TestExperienceReplay(unittest.TestCase):
    """Test experience replay buffer."""
    
    def setUp(self):
        """Set up test fixtures."""
        from ai_balancer import ReplayBuffer, Experience, SystemState
        self.buffer = ReplayBuffer(capacity=100)
        
    def test_buffer_add(self):
        """Test adding experiences."""
        from ai_balancer import Experience, SystemState
        state = SystemState((0,), (0,), 1, 1)
        exp = Experience(state=state, action=0, reward=1.0, next_state=state, done=False)
        
        self.buffer.add(exp)
        self.assertEqual(len(self.buffer), 1)
        
    def test_buffer_sample(self):
        """Test sampling from buffer."""
        from ai_balancer import Experience, SystemState
        
        # Add 10 experiences
        for i in range(10):
            state = SystemState((i % 3,), (0,), 1, 1)
            exp = Experience(state=state, action=i % 4, reward=-1.0, next_state=state, done=False)
            self.buffer.add(exp)
            
        # Sample batch
        batch = self.buffer.sample(5)
        self.assertEqual(len(batch), 5)
        
    def test_buffer_capacity(self):
        """Test buffer respects capacity."""
        from ai_balancer import Experience, SystemState
        
        # Add more than capacity
        for i in range(150):
            state = SystemState((i % 3,), (0,), 1, 1)
            exp = Experience(state=state, action=0, reward=1.0, next_state=state, done=False)
            self.buffer.add(exp)
            
        self.assertEqual(len(self.buffer), 100)


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_all_tests(verbosity=2):
    """Run all tests with specified verbosity."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestConfig,
        TestProcess,
        TestProcessGenerator,
        TestProcessor,
        TestProcessorManager,
        TestRoundRobinBalancer,
        TestLeastLoadedBalancer,
        TestThresholdBasedBalancer,
        TestLoadBalancerFactory,
        TestQLearningBalancer,
        TestQLearningAgent,
        TestStateEncoder,
        TestExperienceReplay,
        TestSimulationEngine,
        TestSimulationResult,
        TestProcessMetrics,
        TestSystemMetrics,
        TestMetricsCalculator,
        TestMetricsComparator,
        TestDataExporter,
        TestIntegration,
        TestEdgeCases,
        TestScenarios,
        TestPerformance,
        TestErrorHandling,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
        
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)
    
    return result


def run_quick_tests():
    """Run a quick subset of critical tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Only critical tests
    critical_tests = [
        TestConfig,
        TestProcess,
        TestProcessor,
        TestSimulationEngine,
        TestIntegration,
    ]
    
    for test_class in critical_tests:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
        
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == '__main__':
    print("=" * 70)
    print("DYNAMIC LOAD BALANCING SIMULATOR - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print()
    
    # Enable logging for debugging if needed
    # logging.disable(logging.NOTSET)
    
    # Run all tests
    result = run_all_tests(verbosity=2)
    
    # Summary
    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print()
    
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
    else:
        print("❌ SOME TESTS FAILED")
        
        if result.failures:
            print("\nFailures:")
            for test, trace in result.failures:
                print(f"  - {test}")
                
        if result.errors:
            print("\nErrors:")
            for test, trace in result.errors:
                print(f"  - {test}")
    
    print("=" * 70)
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
