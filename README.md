# 🚀 Dynamic Load Balancing in Multiprocessor Systems

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-133%20Passing-28A745?style=for-the-badge&logo=pytest&logoColor=white)](test_suite.py)
[![Algorithms](https://img.shields.io/badge/Algorithms-13-blue?style=for-the-badge&logo=buffer&logoColor=white)](scheduling_algorithms.py)
[![License](https://img.shields.io/badge/License-MIT-FFC107?style=for-the-badge)](LICENSE)

**A production-grade simulator for dynamic load balancing algorithms with AI-powered optimization**

[Quick Start](#-quick-start) • [Features](#-features) • [Algorithms](#-load-balancing-algorithms) • [Documentation](#-api-reference) • [Contributing](#-contributing)

</div>

---

## 🎬 Quick Start

```bash
# Clone the repository
git clone https://github.com/Auankj/dynamic_load_balancer.git
cd dynamic_load_balancer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the GUI
python main.py
```

<details>
<summary>🖼️ <b>Screenshot Preview</b></summary>

The GUI features:
- Real-time processor load visualization with color-coded bars
- Interactive Gantt chart showing process execution timeline
- Live metrics dashboard with performance statistics
- Algorithm comparison with side-by-side analysis

</details>

---

## 🎯 Overview

**Load balancing** is a critical operating system technique that distributes workloads across multiple processors to maximize efficiency. This simulator provides:

| Goal | Description |
|------|-------------|
| 🚀 **Maximize Throughput** | Complete more work in less time |
| ⚡ **Minimize Response Time** | Users get faster responses |
| ⚖️ **Optimize Utilization** | All processors stay busy |
| 🛡️ **Prevent Bottlenecks** | No single processor gets overwhelmed |

---

## ✨ Features

### 🎮 Core Simulation Engine
| Feature | Description |
|---------|-------------|
| **Multi-Processor** | Configure 2-16 virtual processors with customizable speed |
| **Process Types** | CPU-bound, I/O-bound, Real-time, Batch, Interactive |
| **Workload Patterns** | Uniform, Bursty, Poisson, Diurnal, Spike, Wave |
| **13 Algorithms** | Load Balancing + Classic CPU Scheduling |
| **AI-Powered** | Deep reinforcement learning with PyTorch (GPU accelerated) |
| **Process Migration** | Dynamic load rebalancing across processors |

### 🤖 AI Load Balancing
| Feature | Description |
|---------|-------------|
| **Q-Learning** | Discrete state-space reinforcement learning |
| **Deep Q-Network (DQN)** | Neural network with experience replay |
| **Double DQN** | Reduced overestimation bias |
| **Prioritized Replay** | Focus on important experiences |
| **Model Persistence** | Save/load trained models automatically |

### 📊 Advanced Simulation
| Feature | Description |
|---------|-------------|
| **Process Types** | CPU_BOUND, IO_BOUND, MIXED, REAL_TIME, BATCH, INTERACTIVE |
| **Workload Patterns** | UNIFORM, BURSTY, POISSON, DIURNAL, SPIKE, GRADUAL_RAMP, WAVE |
| **Advanced Processors** | Multi-level feedback queue, cache simulation, power states |
| **Scenario System** | Predefined and custom simulation scenarios |
| **SLA Tracking** | Service Level Agreement metrics and violations |

### 🎨 Rich Visualization

| Chart Type | Description |
|------------|-------------|
| 📊 **Load Bars** | Real-time processor load with traffic-light colors (🟢→🟡→🔴) |
| 📈 **Gantt Chart** | Interactive timeline showing process execution across processors |
| 📉 **Performance Graphs** | Live metrics with trend lines and historical data |
| 🔄 **Queue Visualization** | Ready queue depth per processor |
| ⚖️ **Algorithm Comparison** | Side-by-side performance analysis with bar/line charts |
| 🎯 **Fairness Index** | Jain's fairness visualization across processors |
| ⏱️ **Response Time Distribution** | Histogram of process response times |
| 🔥 **Heatmap** | Processor utilization over time |

### 📈 Comprehensive Analytics
- **Process Metrics** — Turnaround time, waiting time, response time
- **Processor Metrics** — CPU utilization, queue length, throughput
- **System Metrics** — Load variance, Jain's fairness index, migrations
- **Data Export** — JSON and CSV export for external analysis

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                           GUI Layer                                │
│                          (gui.py)                                  │
│   ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────────┐  │
│   │ Load Bars  │ │  Metrics   │ │   Charts   │ │   Controls     │  │
│   └────────────┘ └────────────┘ └────────────┘ └────────────────┘  │
└────────────────────────────────┬───────────────────────────────────┘
                                 │
┌────────────────────────────────▼───────────────────────────────────┐
│                       Simulation Layer                             │
│           (simulation.py / enhanced_simulation.py)                 │
│   ┌────────────────────────────────────────────────────────────┐   │
│   │                    SimulationEngine                        │   │
│   │    Time Management • Event Processing • State Control      │   │
│   └────────────────────────────────────────────────────────────┘   │
└──────┬──────────────────┬───────────────────┬──────────────────────┘
       │                  │                   │
┌──────▼──────┐   ┌───────▼───────┐   ┌───────▼───────┐
│Load Balancer│   │   Processor   │   │    Metrics    │
│             │   │               │   │               │
│• RoundRobin │   │• Execution    │   │• Process      │
│• LeastLoaded│   │• Queue Mgmt   │   │• Processor    │
│• Threshold  │   │• Migration    │   │• System       │
│• Q-Learning │   │• Power States │   │• SLA Tracking │
│• DQN        │   │• Cache Sim    │   │               │
└──────┬──────┘   └───────┬───────┘   └───────────────┘
       │                  │
┌──────▼──────────────────▼──────┐
│          Core Layer            │
│   config.py • process.py       │
│   utils.py • validators.py     │
│   advanced_simulation.py       │
│   integration.py               │
└────────────────────────────────┘
```

### Design Patterns

| Pattern | Implementation | Purpose |
|---------|----------------|---------|
| **Strategy** | LoadBalancer ABC | Swappable algorithms |
| **Factory** | LoadBalancerFactory | Algorithm instantiation |
| **Observer** | GUI callbacks | Real-time updates |
| **Builder** | ScenarioBuilder | Custom scenario creation |
| **Singleton** | SimulationLogger | Centralized logging |

---

## ⚖️ Load Balancing Algorithms

### Quick Comparison

| Algorithm | Speed | Balance | Adaptability | Best For |
|-----------|:-----:|:-------:|:------------:|----------|
| **Round Robin** | ⭐⭐⭐ | ⭐ | ⭐ | Uniform workloads |
| **Least Loaded** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | Variable workloads |
| **Threshold** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Dynamic environments |
| **Q-Learning** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Pattern-rich workloads |
| **DQN** | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Complex continuous states |

### 1. Round Robin
> Distributes processes cyclically: P0→P1→P2→P3→P0...

```python
def assign(self, process, processors):
    target = processors[self.current_index]
    self.current_index = (self.current_index + 1) % len(processors)
    return target
```

✅ Simple, predictable, zero overhead  
❌ Ignores actual load, can create imbalance

---

### 2. Least Loaded First
> Assigns to the processor with the lowest current load

```python
def assign(self, process, processors):
    return min(processors, key=lambda p: p.current_load)
```

✅ Optimal distribution, adapts to state  
❌ O(n) per assignment, monitoring overhead

---

### 3. Threshold-Based
> Migrates processes when load difference exceeds threshold

```python
def check_balance(self, processors):
    loads = [p.current_load for p in processors]
    if max(loads) - min(loads) > self.threshold:
        self.migrate_process(overloaded, underloaded)
```

✅ Dynamic rebalancing, prevents severe imbalance  
❌ Migration has cost, needs threshold tuning

---

### 4. Q-Learning (AI)
> Learns optimal assignments through reinforcement learning

```python
def assign(self, process, processors):
    state = self.encode_state(processors, process)
    if self.training and random() < self.epsilon:
        action = random_choice(len(processors))  # Explore
    else:
        action = argmax(self.Q[state])           # Exploit
    return processors[action]
```

✅ Learns optimal strategy, improves over time  
❌ Needs training, initial random behavior

---

### 5. Deep Q-Network (DQN)
> Neural network approximates Q-function for continuous states

```python
class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, action_dim)
```

✅ Handles continuous states, excellent generalization  
❌ Requires PyTorch, more computationally intensive

---

### 🎓 AI Training Guide

| Mode | Exploration (ε) | Purpose | When to Use |
|------|-----------------|---------|-------------|
| **Train** | 100% → 5% | Learn optimal strategies | First runs, new patterns |
| **Exploit** | Fixed 1% | Use learned knowledge | After training complete |

**Recommended Training:**
- Q-Learning: 500-2000+ process assignments
- DQN: 1000-5000+ process assignments

---

## 📚 Classic CPU Scheduling Algorithms

The simulator also includes **8 classic CPU scheduling algorithms** that every OS student should know!

### Algorithm Comparison Table

| Algorithm | Type | Preemptive | Optimal For | Starvation |
|-----------|:----:|:----------:|-------------|:----------:|
| **FCFS** | 📋 FIFO | ❌ | Simplicity | ❌ |
| **SJF** | ⏱️ Burst | ❌ | Avg Wait Time | ⚠️ |
| **SRTF** | ⏱️ Burst | ✅ | Response Time | ⚠️ |
| **Priority** | 🎯 Priority | ❌ | Critical Tasks | ⚠️ |
| **Priority (P)** | 🎯 Priority | ✅ | Urgent Tasks | ⚠️ |
| **Multilevel Queue** | 📊 Class | ✅ | Mixed Workloads | ⚠️ |
| **MLFQ** | 🧠 Adaptive | ✅ | General Purpose | ❌ |
| **EDF** | ⏰ Deadline | ✅ | Real-Time | ❌ |

---

### 1️⃣ FCFS – First Come First Served
> The OG of schedulers. Whoever comes first gets the CPU first.

```python
def select_next(self):
    return sorted(self.queue, key=lambda p: p.arrival_time)[0]
```

✅ Simple, no starvation, minimal overhead  
❌ Convoy effect — one big process blocks everyone

---

### 2️⃣ SJF – Shortest Job First
> Picks the process with the shortest burst time. Productivity king! 👑

```python
def select_next(self):
    return min(self.queue, key=lambda p: p.burst_time)
```

✅ Optimal average waiting time (provably!)  
❌ Long jobs may starve, needs burst time prediction

---

### 3️⃣ SRTF – Shortest Remaining Time First
> The chaotic younger sibling of SJF. Preemptive version!

```python
def should_preempt(self, new_process):
    return new_process.remaining_time < self.current.remaining_time
```

✅ Even better response time than SJF  
❌ High overhead, long jobs get ghosted constantly 👻

---

### 4️⃣ Priority Scheduling
> CPU goes to the highest priority process. VIP treatment! 🎖️

```python
def select_next(self):
    # Uses aging to prevent starvation
    for p in self.queue:
        p.effective_priority = p.priority - p.wait_time // 10
    return min(self.queue, key=lambda p: p.effective_priority)
```

✅ Critical processes get attention, flexible  
❌ Can starve low priority (solved with aging!)

---

### 5️⃣ Round Robin (RR)
> Every process gets a time slice (quantum). Fair & democratic! 🗳️

```python
def tick(self):
    self.quantum_remaining -= 1
    if self.quantum_remaining <= 0:
        self.queue.append(self.current)  # Back of queue
        self.current = self.queue.popleft()
```

✅ Fair, good response time, no starvation  
❌ Quantum too small = too many switches, too big = becomes FCFS

---

### 6️⃣ Multilevel Queue Scheduling
> Think of it like a school with different sections! 🏫

```
┌─────────────────────────────────────┐
│ Queue 0: System Processes (RR, q=8) │ ← Highest Priority
├─────────────────────────────────────┤
│ Queue 1: Interactive (RR, q=4)      │
├─────────────────────────────────────┤
│ Queue 2: Batch Jobs (FCFS)          │
├─────────────────────────────────────┤
│ Queue 3: Idle Processes             │ ← Lowest Priority
└─────────────────────────────────────┘
```

✅ Good for categorized workloads, optimizes each queue  
❌ Inflexible — processes stuck in their queue

---

### 7️⃣ MLFQ – Multilevel Feedback Queue
> The genius, adaptive version. Processes can MOVE between queues! 🧠

**The Rules:**
1. 🆕 New processes start at top queue (highest priority)
2. ⬇️ Use full quantum? Get demoted to lower queue
3. ⬆️ Give up CPU early (I/O)? Stay or get promoted
4. 🔄 Periodic priority boost prevents starvation

```python
def after_quantum(self, process):
    if process.used_full_quantum:
        self.demote(process)  # CPU hog detected!
    else:
        self.promote(process)  # Nice I/O-bound process
```

✅ Adapts automatically, no prior knowledge needed  
❌ Complex to tune, can be gamed

---

### 8️⃣ EDF – Earliest Deadline First
> For real-time systems. Nearest deadline = gets the CPU! ⏰

```python
def select_next(self):
    return min(self.queue, key=lambda p: p.deadline)
```

✅ Optimal for single processor real-time (can achieve 100% utilization!)  
❌ Deadline miss cascade — one miss can cause many

---
## 📁 Project Structure

```
dynamic_load_balancer/
├── 🎯 Core Modules
│   ├── main.py                 # Application entry point
│   ├── config.py               # Configuration and constants
│   ├── process.py              # Process model and generator
│   ├── processor.py            # Processor execution logic
│   └── simulation.py           # Standard simulation engine
│
├── 🤖 AI & Scheduling Modules
│   ├── load_balancer.py        # Algorithm implementations
│   ├── ai_balancer.py          # Q-Learning balancer
│   ├── dqn_balancer.py         # Deep Q-Network balancer
│   └── scheduling_algorithms.py # Classic CPU schedulers (FCFS, SJF, etc.)
│
├── 🚀 Advanced Simulation
│   ├── advanced_simulation.py  # Enhanced process/processor models
│   ├── enhanced_simulation.py  # Production-grade engine
│   └── integration.py          # Scenario management
│
├── 📊 Support Modules
│   ├── metrics.py              # Performance metrics
│   ├── gui.py                  # Tkinter GUI
│   ├── utils.py                # Logging and export
│   └── validators.py           # Input validation
│
├── 🧪 Testing
│   └── test_suite.py           # 133 comprehensive tests
│
└── 📄 Documentation
    ├── README.md               # This file
    └── requirements.txt        # Dependencies
```

### Module Overview

| Module | Purpose | Key Classes |
|--------|---------|-------------|
| `config.py` | Configuration | SimulationConfig, LoadBalancingAlgorithm |
| `process.py` | Process model | Process, ProcessGenerator |
| `processor.py` | Execution | Processor, ProcessorManager |
| `load_balancer.py` | Algorithms | RoundRobin, LeastLoaded, Threshold |
| `ai_balancer.py` | Q-Learning | QLearningAgent, StateEncoder |
| `dqn_balancer.py` | Deep RL | DQNAgent, DQNetwork, PrioritizedReplay |
| `scheduling_algorithms.py` | CPU Scheduling | FCFS, SJF, SRTF, Priority, MLFQ, EDF |
| `advanced_simulation.py` | Advanced models | AdvancedProcess, AdvancedProcessor |
| `enhanced_simulation.py` | Production engine | EnhancedSimulationEngine |
| `integration.py` | Scenarios | ScenarioBuilder, PerformanceAnalyzer |

---

## 🎮 Predefined Scenarios

| Scenario | Processors | Processes | Pattern | Description |
|----------|:----------:|:---------:|---------|-------------|
| **Basic** | 4 | 20 | Uniform | Standard simulation |
| **CPU Intensive** | 8 | 30 | Uniform | Long-running computation |
| **I/O Intensive** | 4 | 40 | Bursty | Frequent blocking |
| **Mixed Workload** | 6 | 50 | Diurnal | Real-world simulation |
| **Bursty Traffic** | 4 | 60 | Spike | Sudden load spikes |
| **Real-Time** | 8 | 25 | Uniform | Strict deadlines |
| **Stress Test** | 4 | 100 | Spike | Maximum load testing |

---

## 📊 Performance Metrics

### Key Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Turnaround Time** | `completion - arrival` | Total time in system |
| **Waiting Time** | `start - arrival` | Time in queue |
| **Response Time** | `first_run - arrival` | Time to first execution |
| **CPU Utilization** | `busy_time / total_time` | Processor efficiency |
| **Throughput** | `completed / time` | Processes per time unit |
| **Jain's Fairness** | `(Σx)² / (n × Σx²)` | Load distribution fairness |

### Jain's Fairness Index

$$J(x_1, x_2, ..., x_n) = \frac{(\sum_{i=1}^{n} x_i)^2}{n \cdot \sum_{i=1}^{n} x_i^2}$$

- **J = 1.0**: Perfect fairness (equal load)
- **J = 1/n**: Worst case (all load on one processor)

---

## 🧪 Testing

```bash
# Run all 125 tests
python -m pytest test_suite.py -v

# Run specific test class
python -m pytest test_suite.py::TestDQNBalancer -v

# Run with coverage
python -m pytest test_suite.py --cov=. --cov-report=html
```

### Test Categories

| Category | Tests | Coverage |
|----------|:-----:|----------|
| Configuration | 6 | Config creation, defaults |
| Process Model | 12 | Lifecycle, state transitions |
| Processor | 14 | Execution, queue management |
| Load Balancers | 20 | All algorithm correctness |
| Q-Learning | 15 | Agent training, inference |
| DQN | 20 | Neural network, replay buffer |
| Simulation | 12 | Engine initialization, execution |
| Metrics | 13 | Calculations, edge cases |
| Integration | 6 | End-to-end workflows |
| Edge Cases | 7 | Boundary conditions |

---

## 📚 API Reference

### Quick Examples

```python
# Create and run simulation
from simulation import SimulationEngine
from config import SimulationConfig, LoadBalancingAlgorithm

config = SimulationConfig(num_processors=4, num_processes=20)
engine = SimulationEngine(config)
engine.initialize(algorithm=LoadBalancingAlgorithm.DQN)

while not engine.is_complete():
    engine.step()

result = engine.get_result()
print(f"Avg Turnaround: {result.system_metrics.avg_turnaround_time:.2f}")
```

```python
# Use scenario builder
from integration import ScenarioBuilder, IntegratedSimulationManager
from advanced_simulation import WorkloadPattern, ProcessType

scenario = (ScenarioBuilder("Custom Test")
    .with_processors(8)
    .with_processes(50)
    .with_workload(WorkloadPattern.BURSTY)
    .with_algorithm(LoadBalancingAlgorithm.DQN)
    .build())

manager = IntegratedSimulationManager(use_enhanced=True)
manager.load_scenario(scenario)
manager.initialize()
manager.start()
```

---

## ⚙️ Configuration

### SimulationConfig Options

| Parameter | Default | Range | Description |
|-----------|:-------:|-------|-------------|
| `num_processors` | 4 | 2-16 | Number of processors |
| `num_processes` | 20 | 1-100 | Processes to generate |
| `time_quantum` | 4 | 1-20 | Round robin time slice |
| `min_burst_time` | 1 | 1-100 | Minimum burst time |
| `max_burst_time` | 20 | 1-1000 | Maximum burst time |
| `migration_threshold` | 0.3 | 0.0-1.0 | Load diff for migration |

---

## 💻 Platform Notes

### macOS
```bash
source venv/bin/activate
python main.py
```

### Windows (PowerShell)
```powershell
.\venv\Scripts\Activate.ps1
python main.py
```

### Windows (Command Prompt)
```bat
venv\Scripts\activate.bat
python main.py
```

---

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Write** tests for your changes
4. **Ensure** all 125 tests pass: `python -m pytest test_suite.py`
5. **Commit** with conventional format: `feat(scope): description`
6. **Push** to your branch: `git push origin feature/amazing-feature`
7. **Open** a Pull Request

### Commit Format
```
type(scope): Brief description

Types: feat, fix, docs, refactor, test, perf

Examples:
- feat(balancer): Add weighted round robin
- fix(gui): Resolve chart rendering issue
- test(dqn): Add edge case tests
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Made with ❤️ for learning Operating Systems concepts**

[![GitHub](https://img.shields.io/badge/GitHub-Auankj-181717?style=flat-square&logo=github)](https://github.com/Auankj/dynamic_load_balancer)

**v2.0.0** • **125 Tests Passing** • **Python 3.8+** • **PyTorch 2.0+**

</div>
