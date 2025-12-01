#!/usr/bin/env python3
"""
Dynamic Load Balancing in Multiprocessor Systems - Main Entry Point

This is the main entry point for the Dynamic Load Balancing Simulator.
It initializes all components, sets up the GUI, and starts the application.

The simulation demonstrates key Operating System concepts:
- Process Management and Scheduling
- Multiprocessor Systems
- Load Balancing Algorithms
- Performance Metrics and Analysis

Usage:
    python main.py              # Run with GUI
    python main.py --cli        # Run in CLI mode (for testing)
    python main.py --test       # Run module tests

Author: Student
Date: December 2024
"""

import sys
import os
import argparse
import logging
from typing import Optional

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import project modules
from config import (
    SimulationConfig,
    GUIConfig,
    LoggingConfig,
    LoadBalancingAlgorithm,
    VERSION,
    APP_NAME,
    DEFAULT_SIMULATION_CONFIG,
    DEFAULT_GUI_CONFIG,
    DEFAULT_LOGGING_CONFIG
)
from process import Process, ProcessGenerator
from processor import Processor, ProcessorManager
from utils import (
    SimulationLogger,
    setup_logging,
    DataExporter,
    calculate_mean,
    calculate_std_dev,
    calculate_load_balance_index,
    calculate_jains_fairness_index
)


def print_banner():
    """Print application banner."""
    banner = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║       DYNAMIC LOAD BALANCING IN MULTIPROCESSOR SYSTEMS               ║
║                        Simulator v{VERSION}                             ║
║                                                                      ║
║   An educational simulation demonstrating OS load balancing concepts ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def run_cli_simulation():
    """
    Run a command-line simulation for testing purposes.
    
    This provides a quick way to test the core simulation logic
    without the GUI. Uses the new SimulationEngine for proper execution.
    """
    print_banner()
    print("\n[CLI Mode - Testing Core Simulation]\n")
    
    # Import simulation engine
    from simulation import SimulationEngine, BatchSimulator
    from load_balancer import LoadBalancerFactory
    
    # Initialize configuration
    config = SimulationConfig(
        num_processors=4,
        num_processes=10,
        time_quantum=4
    )
    
    logger = setup_logging()
    logger.info("Starting CLI simulation")
    
    print(f"Configuration:")
    print(f"  Processors: {config.num_processors}")
    print(f"  Processes: {config.num_processes}")
    print(f"  Time Quantum: {config.time_quantum}")
    print(f"  Algorithm: {config.default_algorithm.value}")
    
    # Create and run simulation engine
    print(f"\n[Running Simulation with {config.default_algorithm.value}]")
    print("-" * 60)
    
    engine = SimulationEngine(config)
    engine.initialize(algorithm=config.default_algorithm)
    
    # Show generated processes
    print(f"\nGenerated Processes:")
    for p in engine.all_processes:
        print(f"  P{p.pid}: arrival={p.arrival_time}, burst={p.burst_time}, "
              f"priority={p.priority.name}")
    
    # Run simulation
    print(f"\n[Simulation Progress]")
    result = engine.run()
    
    # Display results
    print(f"\n[Simulation Complete]")
    print("-" * 60)
    print(f"Total Time: {result.total_time} time units")
    print(f"Execution Duration: {result.execution_duration:.3f} seconds")
    print(f"Completed: {result.system_metrics.completed_processes}/{result.system_metrics.total_processes}")
    
    # Process metrics
    print(f"\nProcess Metrics:")
    print(f"  Average Turnaround Time: {result.system_metrics.avg_turnaround_time:.2f}")
    print(f"  Average Waiting Time: {result.system_metrics.avg_waiting_time:.2f}")
    print(f"  Average Response Time: {result.system_metrics.avg_response_time:.2f}")
    
    # Processor metrics
    print(f"\nProcessor Metrics:")
    for pm in result.processor_metrics:
        util = pm.get_utilization(result.total_time) * 100
        print(f"  Processor {pm.processor_id}: "
              f"Utilization={util:.1f}%, "
              f"Completed={pm.processes_completed}")
    
    print(f"\nOverall Metrics:")
    print(f"  Average Utilization: {result.system_metrics.avg_utilization*100:.1f}%")
    print(f"  Load Balance Index: {result.system_metrics.load_balance_index:.4f}")
    print(f"  Jain's Fairness Index: {result.system_metrics.jains_fairness_index:.4f}")
    print(f"  Total Migrations: {result.system_metrics.total_migrations}")
    
    # Run comparison across all algorithms
    print(f"\n[Algorithm Comparison]")
    print("-" * 60)
    
    batch = BatchSimulator(config)
    comparison_results = batch.run_comparison()
    
    print("\nResults by Algorithm:")
    for algo_name, algo_result in comparison_results.items():
        m = algo_result.system_metrics
        print(f"\n  {algo_name}:")
        print(f"    Time: {algo_result.total_time}")
        print(f"    Avg Turnaround: {m.avg_turnaround_time:.2f}")
        print(f"    Avg Waiting: {m.avg_waiting_time:.2f}")
        print(f"    Utilization: {m.avg_utilization*100:.1f}%")
        print(f"    Migrations: {m.total_migrations}")
    
    print(f"\n  Best for turnaround: {batch.get_best_algorithm('avg_turnaround_time')}")
    print(f"  Best for fairness: {batch.get_best_algorithm('jains_fairness_index')}")
    
    logger.info("CLI simulation completed")
    print("\n[CLI Simulation Complete]")


def run_module_tests():
    """Run tests for all modules."""
    print_banner()
    print("\n[Running Module Tests]\n")
    
    test_results = []
    
    # Test config module
    print("Testing config.py...")
    try:
        from config import SimulationConfig, GUIConfig
        config = SimulationConfig()
        config.validate()
        gui_config = GUIConfig()
        test_results.append(("config.py", "PASS"))
        print("  ✓ config.py: PASS")
    except Exception as e:
        test_results.append(("config.py", f"FAIL: {e}"))
        print(f"  ✗ config.py: FAIL - {e}")
    
    # Test process module
    print("Testing process.py...")
    try:
        from process import Process, ProcessGenerator, ProcessPriority
        p = Process(pid=1, arrival_time=0, burst_time=10)
        assert p.remaining_time == 10
        p.set_ready()
        assert p.is_ready()
        
        gen = ProcessGenerator()
        processes = gen.generate_processes(5)
        assert len(processes) == 5
        
        test_results.append(("process.py", "PASS"))
        print("  ✓ process.py: PASS")
    except Exception as e:
        test_results.append(("process.py", f"FAIL: {e}"))
        print(f"  ✗ process.py: FAIL - {e}")
    
    # Test processor module
    print("Testing processor.py...")
    try:
        from processor import Processor, ProcessorManager
        proc = Processor(processor_id=0)
        assert proc.is_idle()
        
        manager = ProcessorManager(num_processors=4)
        assert len(manager) == 4
        
        test_results.append(("processor.py", "PASS"))
        print("  ✓ processor.py: PASS")
    except Exception as e:
        test_results.append(("processor.py", f"FAIL: {e}"))
        print(f"  ✗ processor.py: FAIL - {e}")
    
    # Test utils module
    print("Testing utils.py...")
    try:
        from utils import (
            calculate_mean, 
            calculate_variance,
            calculate_load_balance_index,
            validate_positive_int
        )
        assert calculate_mean([1, 2, 3, 4, 5]) == 3.0
        assert validate_positive_int(5, "test") == 5
        
        test_results.append(("utils.py", "PASS"))
        print("  ✓ utils.py: PASS")
    except Exception as e:
        test_results.append(("utils.py", f"FAIL: {e}"))
        print(f"  ✗ utils.py: FAIL - {e}")
    
    # Test metrics module
    print("Testing metrics.py...")
    try:
        from metrics import (
            ProcessMetrics,
            ProcessorMetrics,
            SystemMetrics,
            MetricsCalculator,
            MetricsComparator
        )
        calc = MetricsCalculator()
        assert calc is not None
        comparator = MetricsComparator()
        assert comparator is not None
        
        test_results.append(("metrics.py", "PASS"))
        print("  ✓ metrics.py: PASS")
    except Exception as e:
        test_results.append(("metrics.py", f"FAIL: {e}"))
        print(f"  ✗ metrics.py: FAIL - {e}")
    
    # Test load_balancer module
    print("Testing load_balancer.py...")
    try:
        from load_balancer import (
            LoadBalancerFactory,
            RoundRobinBalancer,
            LeastLoadedBalancer,
            ThresholdBasedBalancer
        )
        from config import LoadBalancingAlgorithm
        
        rr = LoadBalancerFactory.create(LoadBalancingAlgorithm.ROUND_ROBIN)
        assert rr is not None
        ll = LoadBalancerFactory.create(LoadBalancingAlgorithm.LEAST_LOADED)
        assert ll is not None
        tb = LoadBalancerFactory.create(LoadBalancingAlgorithm.THRESHOLD_BASED)
        assert tb is not None
        
        test_results.append(("load_balancer.py", "PASS"))
        print("  ✓ load_balancer.py: PASS")
    except Exception as e:
        test_results.append(("load_balancer.py", f"FAIL: {e}"))
        print(f"  ✗ load_balancer.py: FAIL - {e}")
    
    # Test simulation module
    print("Testing simulation.py...")
    try:
        from simulation import (
            SimulationEngine,
            SimulationState,
            BatchSimulator
        )
        engine = SimulationEngine()
        assert engine is not None
        assert engine.state == SimulationState.IDLE
        
        test_results.append(("simulation.py", "PASS"))
        print("  ✓ simulation.py: PASS")
    except Exception as e:
        test_results.append(("simulation.py", f"FAIL: {e}"))
        print(f"  ✗ simulation.py: FAIL - {e}")
    
    # Test GUI module
    print("Testing gui.py...")
    try:
        from gui import (
            LoadBalancerGUI,
            ColorScheme,
            LoadBar,
            ProcessorWidget,
            MetricCard
        )
        # Test color scheme
        assert ColorScheme.get_load_color(0.3) == ColorScheme.LOAD_LOW
        assert ColorScheme.get_load_color(0.5) == ColorScheme.LOAD_MEDIUM
        assert ColorScheme.get_load_color(0.8) == ColorScheme.LOAD_HIGH
        
        # Test processor color assignment
        color = ColorScheme.get_processor_color(0)
        assert color is not None
        
        test_results.append(("gui.py", "PASS"))
        print("  ✓ gui.py: PASS")
    except Exception as e:
        test_results.append(("gui.py", f"FAIL: {e}"))
        print(f"  ✗ gui.py: FAIL - {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in test_results if result == "PASS")
    total = len(test_results)
    
    for module, result in test_results:
        status = "✓" if result == "PASS" else "✗"
        print(f"  {status} {module}: {result}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All module tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1


def run_gui():
    """
    Run the graphical user interface.
    
    This imports and starts the GUI module.
    """
    print_banner()
    print("\n[Starting GUI Application]\n")
    
    try:
        # Import GUI module (will be created in Phase 4)
        from gui import LoadBalancerGUI
        
        # Create and run the application
        app = LoadBalancerGUI()
        app.run()
        
    except ImportError as e:
        print(f"Error: GUI module not found. {e}")
        print("\nThe GUI module (gui.py) will be implemented in Phase 4.")
        print("For now, you can run the CLI simulation with: python main.py --cli")
        print("Or run module tests with: python main.py --test")
        return 1
    except Exception as e:
        print(f"Error starting GUI: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def main():
    """
    Main entry point for the application.
    
    Parses command-line arguments and runs the appropriate mode.
    """
    parser = argparse.ArgumentParser(
        description=f"{APP_NAME} v{VERSION}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py              Run the GUI application
  python main.py --cli        Run CLI simulation for testing
  python main.py --test       Run module tests
  python main.py --version    Show version information
        """
    )
    
    parser.add_argument(
        '--cli',
        action='store_true',
        help='Run in command-line mode (no GUI)'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run module tests'
    )
    
    parser.add_argument(
        '--version', '-v',
        action='version',
        version=f'{APP_NAME} v{VERSION}'
    )
    
    parser.add_argument(
        '--processors', '-p',
        type=int,
        default=4,
        help='Number of processors (default: 4)'
    )
    
    parser.add_argument(
        '--processes', '-n',
        type=int,
        default=20,
        help='Number of processes (default: 20)'
    )
    
    parser.add_argument(
        '--algorithm', '-a',
        choices=['rr', 'round_robin', 'll', 'least_loaded', 'tb', 'threshold'],
        default='round_robin',
        help='Load balancing algorithm (default: round_robin)'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run appropriate mode
    if args.test:
        return run_module_tests()
    elif args.cli:
        run_cli_simulation()
        return 0
    else:
        return run_gui()


if __name__ == "__main__":
    sys.exit(main())
