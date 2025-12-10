# 🚀 Dynamic Load Balancing in Multiprocessor Systems

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-125%20Passing-28A745?style=for-the-badge&logo=pytest&logoColor=white)](test_suite.py)
[![License](https://img.shields.io/badge/License-MIT-FFC107?style=for-the-badge)](LICENSE)

**A production-grade simulator for dynamic load balancing algorithms with AI-powered optimization**

[Quick Start](#-quick-start) • [Features](#-features) • [CPU Scheduling](#-cpu-scheduling-algorithms--the-complete-guide) • [Load Balancing](#-load-balancing-algorithms) • [API](#-api-reference) • [Contributing](#-contributing)

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
| **5 Algorithms** | Round Robin, Least Loaded, Threshold, Q-Learning, DQN |
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
- **Real-Time Load Bars** — Color-coded processor visualization (green→yellow→red)
- **Gantt Chart** — Interactive process execution timeline with tooltips
- **Performance Dashboard** — Live metrics with trend indicators
- **Algorithm Comparison** — Side-by-side analysis with charts

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

## 📖 CPU Scheduling Algorithms — The Complete Guide

> *"The CPU is like a popular club — everyone wants in, but only one can party at a time."*

Understanding CPU scheduling is fundamental to OS design. Here's every algorithm you need to know, explained properly:

---

### 1️⃣ FCFS — First Come First Served

> **The OG of schedulers.** Whoever arrives first, gets the CPU first.

```
Queue:  [P1: 24ms] → [P2: 3ms] → [P3: 3ms]
        ═══════════════════════════════════
        |      P1 (24ms)      | P2 | P3 |
        0                     24   27   30
```

| Property | Value |
|----------|-------|
| **Type** | Non-preemptive |
| **Complexity** | O(n) |
| **Starvation** | No |

**Pros:**
- ✅ Dead simple to implement
- ✅ No starvation — every process eventually runs
- ✅ Zero overhead — no context switching mid-process

**Cons:**
- ❌ **Convoy Effect** — One fat process blocks everyone behind it
- ❌ Poor average waiting time
- ❌ Not suitable for interactive systems

**When to use:** Batch systems where simplicity > performance

---

### 2️⃣ SJF — Shortest Job First

> **The productivity king.** Always picks the process with the shortest burst time.

```
Queue:  P1(6ms), P2(8ms), P3(7ms), P4(3ms)

Execution Order: P4 → P1 → P3 → P2
        ═════════════════════════════════
        | P4 |   P1   |   P3   |    P2   |
        0    3        9       16        24
```

| Property | Value |
|----------|-------|
| **Type** | Non-preemptive |
| **Complexity** | O(n log n) |
| **Starvation** | Yes ⚠️ |

**Pros:**
- ✅ **Optimal average waiting time** — mathematically proven!
- ✅ Great for batch processing
- ✅ Maximizes throughput

**Cons:**
- ❌ **How do we know burst time?** — OS has to predict/estimate
- ❌ Long jobs can **starve forever**
- ❌ Not fair for longer processes

**When to use:** When burst times are known or predictable

---

### 3️⃣ SRTF — Shortest Remaining Time First

> **The chaotic younger sibling of SJF.** Preemptive — if a shorter job arrives, *boom*, context switch!

```
Time 0: P1(7ms) arrives, starts running
Time 2: P2(4ms) arrives → P2 is shorter! Preempt P1!
Time 4: P3(1ms) arrives → Even shorter! Preempt P2!

        ═══════════════════════════════════════════
        | P1 |   P2   | P3 |  P2  |     P1      |
        0    2        4    5      7            12
```

| Property | Value |
|----------|-------|
| **Type** | Preemptive |
| **Complexity** | O(n log n) |
| **Starvation** | Yes ⚠️ |

**Pros:**
- ✅ **Best average waiting time** — even better than SJF
- ✅ Responds immediately to short jobs
- ✅ Great for time-sharing systems

**Cons:**
- ❌ Long processes get **constantly ghosted**
- ❌ High context switch overhead
- ❌ Still needs to predict burst times

**When to use:** Interactive systems where responsiveness matters

---

### 4️⃣ Round Robin (RR) — The Crowd Favorite

> **The democratic scheduler.** Everyone gets equal CPU time slices (quantum). Fair, balanced, *Gen Z approved* ✌️

```
Time Quantum = 4ms
Processes: P1(10ms), P2(5ms), P3(8ms)

        ═════════════════════════════════════════════════════
        | P1 | P2 | P3 | P1 | P2 | P3 | P1 | P3 |
        0    4    8   12   16   17   21   23   25
              4ms each (except remainders)
```

| Property | Value |
|----------|-------|
| **Type** | Preemptive |
| **Complexity** | O(1) per decision |
| **Starvation** | No |

**Quantum Sweet Spot:**

| Quantum | Effect |
|---------|--------|
| Too small (1-2ms) | Context switch storm 🌪️ — more switching than computing |
| Too large (100ms+) | Becomes FCFS in disguise |
| Just right (10-100ms) | Balanced responsiveness and efficiency |

**Pros:**
- ✅ **Fair** — no process waits forever
- ✅ Great for time-sharing systems
- ✅ Predictable response time
- ✅ No starvation

**Cons:**
- ❌ More context switches = more overhead
- ❌ Quantum tuning is critical
- ❌ Doesn't consider process priority

**When to use:** Interactive/time-sharing systems, OS like Unix/Linux

---

### 5️⃣ Priority Scheduling

> **VIP access.** CPU goes to the highest priority process. Because some processes are just *more important*.

```
Priority: 1 = Highest, 4 = Lowest

Processes: P1(pri=3), P2(pri=1), P3(pri=4), P4(pri=2)

Execution Order: P2 → P4 → P1 → P3
        ═══════════════════════════════════
        |  P2  |  P4  |  P1  |  P3  |
        (highest)          (lowest)
```

| Property | Value |
|----------|-------|
| **Type** | Preemptive or Non-preemptive |
| **Complexity** | O(n) or O(log n) with heap |
| **Starvation** | Yes ⚠️ |

**Two Flavors:**

| Mode | Behavior |
|------|----------|
| **Preemptive** | Higher priority arrives? Interrupt current! |
| **Non-preemptive** | Wait politely until current finishes |

**The Starvation Problem:**
Low priority processes might wait **forever** if high priority keeps coming.

**Solution — Aging:**
```python
# Increase priority over time
process.priority += time_waiting * AGING_FACTOR
```

**Pros:**
- ✅ Important tasks get priority
- ✅ Flexible for different workloads
- ✅ Works well with real-time constraints

**Cons:**
- ❌ **Starvation** without aging
- ❌ Priority inversion problem
- ❌ Who decides priority? 🤔

**When to use:** Real-time systems, systems with clear task importance

---

### 6️⃣ Multilevel Queue Scheduling

> **Think of it like airport security lanes.** Different queues for different classes — no queue jumping allowed!

```
┌─────────────────────────────────────────────────┐
│  Queue 1: System Processes    [RR, q=8]    ←── Highest Priority
├─────────────────────────────────────────────────┤
│  Queue 2: Interactive/Foreground  [RR, q=16]
├─────────────────────────────────────────────────┤
│  Queue 3: Background/Batch    [FCFS]       ←── Lowest Priority
└─────────────────────────────────────────────────┘
```

| Property | Value |
|----------|-------|
| **Type** | Mixed (per queue) |
| **Flexibility** | Low — fixed queues |
| **Starvation** | Yes ⚠️ |

**Queue Examples:**

| Queue | Processes | Typical Scheduler |
|-------|-----------|-------------------|
| System | Kernel, drivers | Priority/FCFS |
| Interactive | User apps, UI | Round Robin |
| Batch | Backups, compiling | FCFS |

**Pros:**
- ✅ Different policies for different needs
- ✅ System processes always prioritized
- ✅ Efficient for categorized workloads

**Cons:**
- ❌ **No queue jumping** — you're stuck where you are
- ❌ Low priority queues can starve
- ❌ Rigid classification

**When to use:** Systems with clearly separable process classes

---

### 7️⃣ MLFQ — Multilevel Feedback Queue

> **The genius, adaptive version.** Processes can MOVE between queues based on behavior. Short jobs rise, CPU hogs fall.

```
                    New Process Enters
                           ↓
┌─────────────────────────────────────────────────┐
│  Queue 0: Highest Priority   [RR, q=8]         │ ← Start here
│           P1, P2                                │
├─────────────────────────────────────────────────┤
│  Queue 1: Medium Priority    [RR, q=16]        │
│           P3                                    │ ← Demoted if uses full quantum
├─────────────────────────────────────────────────┤
│  Queue 2: Lowest Priority    [FCFS]            │
│           P4, P5                                │ ← CPU hogs end up here
└─────────────────────────────────────────────────┘
                    ↑
            Periodic boost (aging)
```

| Property | Value |
|----------|-------|
| **Type** | Preemptive |
| **Adaptability** | Very High ⭐ |
| **Starvation** | No (with boost) |

**The Rules:**

| Rule | Description |
|------|-------------|
| **Rule 1** | Higher priority queue runs first |
| **Rule 2** | Same priority = Round Robin |
| **Rule 3** | New jobs start at top queue |
| **Rule 4** | Use full quantum? Move DOWN |
| **Rule 5** | Give up CPU early (I/O)? Stay or move UP |
| **Rule 6** | Periodic boost — everyone goes back to top |

**The Brilliance:**
- **Short interactive jobs** → Stay at top, fast response
- **Long CPU-bound jobs** → Sink to bottom, still finish eventually
- **Gaming prevention** → Track total CPU usage, not just last quantum

**Pros:**
- ✅ **Adapts to process behavior** automatically
- ✅ Interactive jobs get great response time
- ✅ No starvation (with periodic boost)
- ✅ Approximates SJF without knowing burst time!

**Cons:**
- ❌ Complex to implement correctly
- ❌ Many parameters to tune (quantums, queues, boost frequency)
- ❌ Vulnerable to gaming (smart processes can exploit rules)

**When to use:** General-purpose OS (Linux, macOS, Windows use MLFQ variants!)

---

### 8️⃣ EDF — Earliest Deadline First

> **For when timing is EVERYTHING.** The process with the nearest deadline gets the CPU. No exceptions.

```
Time: 0
P1: Deadline=10, Burst=3
P2: Deadline=5,  Burst=2
P3: Deadline=8,  Burst=4

Execution: P2(d=5) → P3(d=8) → P1(d=10)
        ═══════════════════════════════════
        | P2 |    P3    |  P1  |
        0    2          6      9
        ✓d=5  ✓d=8       ✓d=10
```

| Property | Value |
|----------|-------|
| **Type** | Preemptive |
| **Optimal for** | Real-time systems |
| **Guarantee** | 100% utilization possible |

**Real-Time Classification:**

| Type | Deadline Miss | Example |
|------|---------------|---------|
| **Hard Real-Time** | Catastrophic failure | Pacemaker, ABS brakes |
| **Soft Real-Time** | Degraded quality | Video streaming, gaming |

**EDF Guarantee:**
> If total CPU utilization ≤ 100%, EDF will meet ALL deadlines!

$$U = \sum_{i=1}^{n} \frac{C_i}{T_i} \leq 1$$

**Pros:**
- ✅ **Optimal** — if deadlines can be met, EDF will meet them
- ✅ Maximizes CPU utilization in real-time systems
- ✅ Dynamic priority = adapts to changing deadlines

**Cons:**
- ❌ **Domino effect** — if overloaded, everything fails
- ❌ Higher overhead than fixed-priority
- ❌ Harder to analyze worst-case behavior

**When to use:** Real-time operating systems (RTOS), embedded systems

---

### 📊 The Ultimate Scheduling Comparison

| Algorithm | Preemptive | Starvation | Overhead | Best For |
|-----------|:----------:|:----------:|:--------:|----------|
| **FCFS** | ❌ | ❌ | Very Low | Batch systems |
| **SJF** | ❌ | ⚠️ Yes | Low | Known burst times |
| **SRTF** | ✅ | ⚠️ Yes | Medium | Interactive systems |
| **Round Robin** | ✅ | ❌ | Medium | Time-sharing |
| **Priority** | Both | ⚠️ Yes | Low-Medium | Real-time, mixed |
| **MLQ** | Mixed | ⚠️ Yes | Low | Categorized workloads |
| **MLFQ** | ✅ | ❌ | High | General-purpose OS |
| **EDF** | ✅ | ❌ | Medium | Real-time systems |

### 🎯 Quick Decision Tree

```
Need real-time guarantees?
├── Yes → EDF or Priority (Hard RT)
└── No → General purpose?
    ├── Yes → MLFQ (most modern OS use this!)
    └── No → What's your priority?
        ├── Simplicity → FCFS or RR
        ├── Efficiency → SJF/SRTF (if burst known)
        └── Fairness → Round Robin
```

---

## ⚖️ Load Balancing Algorithms

> Our simulator implements these algorithms for **multi-processor** systems:

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
├── 🤖 AI Modules
│   ├── load_balancer.py        # Algorithm implementations
│   ├── ai_balancer.py          # Q-Learning balancer
│   └── dqn_balancer.py         # Deep Q-Network balancer
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
│   └── test_suite.py           # 125 comprehensive tests
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
