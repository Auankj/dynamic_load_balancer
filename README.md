# Dynamic Load Balancing in Multiprocessor Systems

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/Tests-91%20Passing-success.svg)](test_suite.py)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/Coverage-Comprehensive-brightgreen.svg)](test_suite.py)

A production-grade educational simulation demonstrating dynamic load balancing algorithms in multiprocessor systems. This project visualizes how operating systems distribute workloads across multiple CPUs to optimize performance and resource utilization.

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Load Balancing Algorithms](#-load-balancing-algorithms)
- [Project Structure](#-project-structure)
- [Performance Metrics](#-performance-metrics)
- [Testing](#-testing)
- [API Reference](#-api-reference)
- [Configuration](#-configuration)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Overview

In modern computing systems, multiple processors (CPUs/cores) work together to execute tasks. **Load balancing** is the process of distributing workloads across these processors to:

- **Maximize throughput** - Complete more work in less time
- **Minimize response time** - Users get faster responses
- **Optimize resource utilization** - All processors stay busy
- **Prevent bottlenecks** - No single processor gets overwhelmed

This simulator allows you to visualize and compare different load balancing strategies in real-time with a comprehensive GUI, detailed metrics, and export capabilities.

## ✨ Features

### Core Simulation Engine
- **Multi-Processor Simulation** - Configure 2-16 virtual processors with customizable speed
- **Process Generation** - Create processes with random or custom attributes (burst time, priority, memory)
- **Discrete Event Simulation** - Accurate time-stepped simulation with configurable speed
- **Multiple Algorithms** - Compare Round Robin, Least Loaded, and Threshold-Based strategies
- **Process Migration** - Dynamic process movement between processors for optimal balance

### Rich Visualization
- **Real-Time Load Bars** - Color-coded processor load visualization (green→yellow→red)
- **Process Queue Display** - Live view of pending and executing processes
- **Performance Dashboard** - Live metrics cards with key statistics
- **Embedded Charts** - Matplotlib-powered graphs for:
  - Processor utilization over time
  - Load balance distribution
  - Process completion timeline
  - Algorithm comparison

### Comprehensive Analytics
- **Process Metrics** - Turnaround time, waiting time, response time per process
- **Processor Metrics** - CPU utilization, queue length, throughput per processor
- **System Metrics** - Load variance, Jain's fairness index, total migrations
- **Data Export** - Export results to JSON and CSV formats

### Robust Architecture
- **Validation Framework** - Input validation and sanitization
- **Error Handling** - Graceful error recovery with detailed logging
- **Thread-Safe** - Non-blocking GUI with background simulation
- **91 Unit Tests** - Comprehensive test coverage
### Data Management
- **Simulation Logging** - Detailed event logging with severity levels
- **Results History** - Track multiple simulation runs
- **Comparison Reports** - Side-by-side algorithm analysis

## 🏗️ Architecture

The project follows a modular architecture with clear separation of concerns:

```
┌──────────────────────────────────────────────────────────────────┐
│                          GUI Layer                                │
│                         (gui.py)                                  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────────┐ │
│  │ Load Bars   │ │  Metrics    │ │   Charts    │ │  Controls  │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └────────────┘ │
└─────────────────────────────────┬────────────────────────────────┘
                                  │
┌─────────────────────────────────▼────────────────────────────────┐
│                     Simulation Layer                              │
│                    (simulation.py)                                │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              SimulationEngine                                │ │
│  │   - Time management  - Event processing  - State control    │ │
│  └─────────────────────────────────────────────────────────────┘ │
└───────┬───────────────────┬───────────────────┬──────────────────┘
        │                   │                   │
┌───────▼───────┐   ┌───────▼───────┐   ┌───────▼───────┐
│ Load Balancer │   │  Processor    │   │   Metrics     │
│ (load_bal.py) │   │ (processor.py)│   │  (metrics.py) │
│               │   │               │   │               │
│ - RoundRobin  │   │ - Execution   │   │ - Process     │
│ - LeastLoaded │   │ - Queue Mgmt  │   │ - Processor   │
│ - Threshold   │   │ - Migration   │   │ - System      │
└───────┬───────┘   └───────┬───────┘   └───────────────┘
        │                   │
┌───────▼───────────────────▼───────┐
│          Core Layer               │
│  (config.py, process.py, utils.py)│
│                                   │
│  - Configuration management       │
│  - Process data structures        │
│  - Logging and export utilities   │
│  - Validation (validators.py)     │
└───────────────────────────────────┘
```

### Design Patterns Used

| Pattern | Implementation | Purpose |
|---------|----------------|---------|
| **Strategy** | LoadBalancer ABC with concrete implementations | Swappable algorithms |
| **Factory** | LoadBalancerFactory | Algorithm instantiation |
| **Observer** | GUI callbacks on simulation events | Real-time updates |
| **Singleton** | SimulationLogger | Centralized logging |
| **State** | ProcessState, SimulationState enums | Clear state management |

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Tkinter (usually included with Python)
- Matplotlib for embedded charts

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/dynamic_load_balancer.git
cd dynamic_load_balancer

# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py

# Or run tests first to verify installation
python test_suite.py
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| matplotlib | ≥3.5.0 | Chart rendering |
| tkinter | (built-in) | GUI framework |

## 📖 Usage

### GUI Mode (Default)
```bash
python main.py
```

The graphical interface provides:
1. **Configuration Panel** - Set processors (2-16), processes (1-100), algorithm
2. **Processor Visualization** - Real-time load bars with color coding
3. **Metrics Dashboard** - Live statistics and performance indicators
4. **Chart View** - Embedded Matplotlib charts for analysis
5. **Control Panel** - Start/Stop/Pause/Reset/Step controls
6. **Export Options** - Save results to JSON/CSV

### CLI Mode
```bash
python main.py --cli
```

Runs a quick simulation in the terminal for testing and automation.

### Run Tests
```bash
# Run all 91 tests
python test_suite.py

# Or with verbose output
python -m pytest test_suite.py -v

# Run specific test class
python -m pytest test_suite.py::TestLoadBalancers -v
```

### Command Line Options
```bash
python main.py --help

Options:
  --cli              Run in command-line mode (no GUI)
  --test             Run module self-tests
  --processors, -p   Number of processors (default: 4)
  --processes, -n    Number of processes (default: 20)
  --algorithm, -a    Load balancing algorithm (round_robin|least_loaded|threshold)
  --version, -v      Show version information
  --debug            Enable debug logging
```

### Example Sessions

```bash
# Quick simulation with 8 processors using threshold algorithm
python main.py --cli -p 8 -n 50 -a threshold

# GUI with custom configuration
python main.py -p 6 -n 30 -a least_loaded

# Run validation tests
python test_suite.py
```

## ⚖️ Load Balancing Algorithms

### 1. Round Robin (`round_robin`)
**How it works:** Distributes processes to processors in a cyclic manner (P0→P1→P2→P3→P0→...).

```python
# Simplified logic
def assign(self, process, processors):
    processor = processors[self.current_index]
    self.current_index = (self.current_index + 1) % len(processors)
    return processor
```

| Pros | Cons |
|------|------|
| Simple and predictable | Ignores actual load |
| Zero monitoring overhead | Can create imbalance |
| Equal distribution by count | Poor for varied workloads |

**Best for:** Homogeneous workloads with similar process sizes

**Time Complexity:** O(1) assignment

---

### 2. Least Loaded First (`least_loaded`)
**How it works:** Assigns each new process to the processor with the lowest current load.

```python
# Simplified logic
def assign(self, process, processors):
    return min(processors, key=lambda p: p.current_load)
```

| Pros | Cons |
|------|------|
| Optimal load distribution | O(n) per assignment |
| Adapts to current state | Requires load monitoring |
| Efficient for varied work | Slightly higher overhead |

**Best for:** Variable workloads with different burst times

**Time Complexity:** O(n) assignment where n = number of processors

---

### 3. Threshold-Based (`threshold`)
**How it works:** Monitors processor loads and migrates processes when load difference exceeds threshold.

```python
# Simplified logic
def check_balance(self, processors):
    loads = [p.current_load for p in processors]
    if max(loads) - min(loads) > self.threshold:
        self.migrate_process(overloaded, underloaded)
```

| Pros | Cons |
|------|------|
| Dynamic rebalancing | Migration has cost |
| Handles changing loads | Needs threshold tuning |
| Prevents severe imbalance | More complex logic |

**Best for:** Dynamic workloads where load changes over time

**Time Complexity:** O(n) monitoring + O(1) migration decision

### Algorithm Comparison

| Metric | Round Robin | Least Loaded | Threshold |
|--------|-------------|--------------|-----------|
| **Assignment Speed** | ⭐⭐⭐ Fastest | ⭐⭐ Medium | ⭐⭐ Medium |
| **Load Balance** | ⭐ Poor | ⭐⭐⭐ Good | ⭐⭐⭐ Best |
| **Adaptability** | ⭐ None | ⭐⭐ Reactive | ⭐⭐⭐ Proactive |
| **Overhead** | ⭐⭐⭐ Minimal | ⭐⭐ Low | ⭐ Higher |
| **Best Scenario** | Uniform tasks | Mixed tasks | Dynamic loads |

## 📁 Project Structure

```
dynamic_load_balancer/
├── main.py              # Application entry point with CLI parsing
├── config.py            # Configuration classes, enums, and constants
├── process.py           # Process dataclass and ProcessGenerator
├── processor.py         # Processor class and ProcessorManager
├── load_balancer.py     # Load balancing algorithms (Strategy pattern)
├── simulation.py        # SimulationEngine and SimulationResult
├── metrics.py           # ProcessMetrics, ProcessorMetrics, SystemMetrics
├── gui.py               # Full Tkinter GUI with Matplotlib integration
├── utils.py             # SimulationLogger, DataExporter utilities
├── validators.py        # Input validation and error handling
├── test_suite.py        # Comprehensive test suite (91 tests)
├── requirements.txt     # Python dependencies
├── README.md            # This documentation
└── project.xml          # Project specification
```

### Module Details

| Module | Lines | Classes | Description |
|--------|-------|---------|-------------|
| `config.py` | ~420 | 5 | SimulationConfig, GUIConfig, ProcessState, ProcessPriority, LoadBalancerType |
| `process.py` | ~750 | 2 | Process dataclass with lifecycle, ProcessGenerator for creating workloads |
| `processor.py` | ~975 | 2 | Processor execution logic, ProcessorManager for multi-processor coordination |
| `load_balancer.py` | ~810 | 5 | LoadBalancer ABC, RoundRobin, LeastLoaded, ThresholdBased, Factory |
| `simulation.py` | ~760 | 3 | SimulationEngine, SimulationState enum, SimulationResult |
| `metrics.py` | ~870 | 4 | ProcessMetrics, ProcessorMetrics, SystemMetrics, MetricsCalculator |
| `gui.py` | ~1430 | 5 | LoadBalancerGUI, LoadBar, ProcessorWidget, MetricCard, ChartFrame |
| `utils.py` | ~700 | 2 | SimulationLogger (singleton), DataExporter (JSON/CSV) |
| `validators.py` | ~500 | 6 | ValidationError, ValidationResult, Config/Process/SimulationValidators |
| `test_suite.py` | ~800 | 18 | Comprehensive unit and integration tests |

## 📊 Performance Metrics

### Process Metrics (per process)

| Metric | Formula | Description | Unit |
|--------|---------|-------------|------|
| **Arrival Time** | Given | When process enters system | time units |
| **Start Time** | First execution | When process first runs | time units |
| **Completion Time** | Last execution | When process finishes | time units |
| **Turnaround Time** | `completion - arrival` | Total time in system | time units |
| **Waiting Time** | `start - arrival` | Time spent in queue | time units |
| **Response Time** | `first_run - arrival` | Time to first response | time units |

### Processor Metrics (per processor)

| Metric | Formula | Description | Range |
|--------|---------|-------------|-------|
| **CPU Utilization** | `busy_time / total_time × 100` | Percentage busy | 0-100% |
| **Queue Length** | `len(waiting_queue)` | Current waiting processes | 0-∞ |
| **Current Load** | `queue_length + remaining_work` | Combined workload | 0-∞ |
| **Throughput** | `completed / total_time` | Processes per time unit | 0-∞ |
| **Idle Time** | `total_time - busy_time` | Time with no work | time units |

### System Metrics (aggregate)

| Metric | Formula | Description | Range |
|--------|---------|-------------|-------|
| **Average Turnaround** | `Σ(turnaround) / n` | Mean turnaround time | 0-∞ |
| **Average Waiting** | `Σ(waiting) / n` | Mean waiting time | 0-∞ |
| **Load Variance** | `σ²(loads)` | Spread of processor loads | 0-∞ |
| **Load Balance Index** | `1 - (max-min)/max` | Balance quality | 0-1 |
| **Jain's Fairness** | `(Σx)² / (n × Σx²)` | Statistical fairness | 0-1 |
| **Total Migrations** | Count | Process movements | 0-∞ |
| **Throughput** | `completed / time` | System throughput | 0-∞ |

### Jain's Fairness Index

A widely-used metric for evaluating fairness in resource allocation:

$$J(x_1, x_2, ..., x_n) = \frac{(\sum_{i=1}^{n} x_i)^2}{n \cdot \sum_{i=1}^{n} x_i^2}$$

Where $x_i$ is the allocation (load) for processor $i$.

- **J = 1.0**: Perfect fairness (all processors equally loaded)
- **J = 1/n**: Worst case (all load on one processor)

## 🧪 Testing

### Test Suite Overview

The project includes a comprehensive test suite with **91 tests** covering:

```bash
# Run all tests
python test_suite.py

# Expected output
Ran 91 tests in X.XXXs
OK
```

### Test Categories

| Test Class | Tests | Coverage |
|------------|-------|----------|
| `TestConfig` | 6 | Configuration creation and defaults |
| `TestProcess` | 7 | Process lifecycle and state transitions |
| `TestProcessGenerator` | 5 | Random and custom process generation |
| `TestProcessor` | 8 | Process execution and queue management |
| `TestProcessorManager` | 6 | Multi-processor coordination |
| `TestRoundRobinBalancer` | 5 | Round robin algorithm correctness |
| `TestLeastLoadedBalancer` | 5 | Least loaded algorithm correctness |
| `TestThresholdBasedBalancer` | 6 | Threshold algorithm and migration |
| `TestLoadBalancerFactory` | 4 | Factory pattern and instantiation |
| `TestSimulationEngine` | 8 | Engine initialization and execution |
| `TestProcessMetrics` | 4 | Individual process metrics |
| `TestSystemMetrics` | 4 | Aggregate system metrics |
| `TestMetricsCalculator` | 5 | Metric calculations |
| `TestIntegration` | 6 | End-to-end workflow tests |
| `TestEdgeCases` | 5 | Boundary conditions |
| `TestScenarios` | 4 | Real-world simulation scenarios |
| `TestPerformance` | 4 | Performance and stress tests |
| `TestErrorHandling` | 4 | Error handling and validation |

### Running Specific Tests

```bash
# Run single test class
python -m pytest test_suite.py::TestLoadBalancers -v

# Run single test method
python -m pytest test_suite.py::TestSimulationEngine::test_initialization -v

# Run with coverage
python -m pytest test_suite.py --cov=. --cov-report=html
```

## 📚 API Reference

### Core Classes

#### `Process` (process.py)
```python
from process import Process, ProcessGenerator

# Create a process
process = Process(
    pid=1,
    burst_time=10,
    arrival_time=0,
    priority=ProcessPriority.NORMAL,
    memory_required=256
)

# Generate random processes
generator = ProcessGenerator(config)
processes = generator.generate_processes(count=20)
```

#### `Processor` (processor.py)
```python
from processor import Processor, ProcessorManager

# Create processor
processor = Processor(processor_id=0, speed=1.0)
processor.add_process(process)
processor.execute_step()

# Create manager for multiple processors
manager = ProcessorManager(num_processors=4)
manager.get_least_loaded_processor()
```

#### `LoadBalancer` (load_balancer.py)
```python
from load_balancer import LoadBalancerFactory, LoadBalancerType

# Create balancer using factory
balancer = LoadBalancerFactory.create(
    LoadBalancerType.LEAST_LOADED,
    config
)

# Assign process to best processor
target = balancer.assign_process(process, processors)
```

#### `SimulationEngine` (simulation.py)
```python
from simulation import SimulationEngine

# Create and run simulation
engine = SimulationEngine(config)
engine.initialize(algorithm=LoadBalancerType.ROUND_ROBIN)

# Add processes
for process in processes:
    engine.add_process(process)

# Run simulation
while not engine.is_complete():
    engine.step()

# Get results
result = engine.get_results()
```

#### `MetricsCalculator` (metrics.py)
```python
from metrics import MetricsCalculator

# Calculate metrics
calculator = MetricsCalculator()
process_metrics = calculator.calculate_process_metrics(process)
system_metrics = calculator.calculate_system_metrics(processes, processors)
```

### Configuration

#### `SimulationConfig` (config.py)
```python
from config import SimulationConfig

config = SimulationConfig(
    num_processors=4,
    min_burst_time=1,
    max_burst_time=20,
    min_arrival_time=0,
    max_arrival_time=50,
    time_quantum=4,
    migration_threshold=0.3,
    simulation_speed=1.0
)
```

### Validation

#### `Validators` (validators.py)
```python
from validators import ConfigValidator, ProcessValidator

# Validate configuration
result = ConfigValidator.validate_config(config)
if not result.is_valid:
    print(result.errors)

# Validate process
result = ProcessValidator.validate_process(process)
```

## ⚙️ Configuration

### SimulationConfig Options

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `num_processors` | int | 4 | 2-16 | Number of processors |
| `min_burst_time` | int | 1 | 1-100 | Minimum process burst time |
| `max_burst_time` | int | 20 | 1-1000 | Maximum process burst time |
| `min_arrival_time` | int | 0 | 0-∞ | Earliest process arrival |
| `max_arrival_time` | int | 50 | 0-∞ | Latest process arrival |
| `time_quantum` | int | 4 | 1-100 | Round robin time slice |
| `migration_threshold` | float | 0.3 | 0.0-1.0 | Load diff for migration |
| `simulation_speed` | float | 1.0 | 0.1-10.0 | GUI update speed |

### GUIConfig Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `window_width` | int | 1400 | Window width in pixels |
| `window_height` | int | 900 | Window height in pixels |
| `update_interval` | int | 100 | GUI refresh rate (ms) |
| `chart_update_interval` | int | 500 | Chart refresh rate (ms) |
| `color_scheme` | dict | {...} | Color definitions |

### Environment Variables

```bash
# Enable debug logging
export LOAD_BALANCER_DEBUG=1

# Set default processor count
export LOAD_BALANCER_PROCESSORS=8

# Set default algorithm
export LOAD_BALANCER_ALGORITHM=threshold
```

## 🤝 Contributing

### Development Setup

```bash
# Clone and setup
git clone https://github.com/yourusername/dynamic_load_balancer.git
cd dynamic_load_balancer
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests before making changes
python test_suite.py

# Make your changes...

# Run tests after changes
python test_suite.py
```

### Code Standards

- **Style**: Follow PEP 8 guidelines
- **Types**: Use type hints for all functions
- **Docs**: Include docstrings for all public classes/methods
- **Tests**: Add tests for new functionality (maintain 91+ passing tests)

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for your changes
4. Ensure all tests pass (`python test_suite.py`)
5. Commit your changes with conventional commit format
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Commit Message Format
```
type(scope): Brief description

Types: feat, fix, docs, refactor, test, perf

Examples:
- feat(balancer): Add weighted round robin algorithm
- fix(gui): Resolve chart rendering on resize
- test(metrics): Add edge case tests for fairness index
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Operating Systems concepts from Silberschatz, Galvin, and Gagne
- Python documentation and community
- Tkinter and Matplotlib libraries

---

**Made with ❤️ for learning Operating Systems concepts**

**Version:** 1.0.0 | **Tests:** 91 Passing | **Python:** 3.8+