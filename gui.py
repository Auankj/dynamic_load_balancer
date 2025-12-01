"""
GUI Module for Dynamic Load Balancing Simulator

This module provides a comprehensive graphical user interface using Tkinter
for visualizing and controlling the load balancing simulation.

Components:
1. Control Panel - Configuration and simulation controls
2. Processor Visualization - Real-time load bars and queue display
3. Gantt Chart - Process execution timeline
4. Metrics Dashboard - Performance statistics and comparison
5. Process Table - Detailed process information

OS Concepts Demonstrated:
- Process state visualization (Ready, Running, Completed)
- CPU utilization monitoring
- Load balancing decision visualization
- Migration tracking and display

Author: Student
Date: December 2024
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import List, Dict, Any, Optional, Callable
import threading
import queue
import time
from dataclasses import dataclass
from enum import Enum
import json

# Import matplotlib for charts
import matplotlib
matplotlib.use('TkAgg')  # Use Tkinter backend
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

# Import project modules
from config import (
    SimulationConfig,
    GUIConfig,
    LoadBalancingAlgorithm,
    ProcessState,
    ProcessPriority,
    VERSION,
    APP_NAME,
    DEFAULT_SIMULATION_CONFIG,
    DEFAULT_GUI_CONFIG
)
from process import Process, ProcessGenerator
from processor import Processor, ProcessorManager
from load_balancer import LoadBalancerFactory
from simulation import SimulationEngine, SimulationState, BatchSimulator
from metrics import MetricsCalculator, MetricsComparator, SystemMetrics


# =============================================================================
# COLOR SCHEMES AND CONSTANTS
# =============================================================================

class ColorScheme:
    """
    Color scheme for the GUI.
    
    Using consistent colors improves user experience and makes
    the visualization easier to understand.
    """
    # Main colors
    BACKGROUND = "#f0f0f0"
    PANEL_BG = "#ffffff"
    TEXT = "#333333"
    TEXT_LIGHT = "#666666"
    
    # Load level colors
    LOAD_LOW = "#4CAF50"      # Green - low load (0-40%)
    LOAD_MEDIUM = "#FFC107"   # Yellow - medium load (40-70%)
    LOAD_HIGH = "#F44336"     # Red - high load (70-100%)
    
    # Process state colors
    STATE_NEW = "#9E9E9E"       # Gray
    STATE_READY = "#2196F3"     # Blue
    STATE_RUNNING = "#4CAF50"   # Green
    STATE_WAITING = "#FF9800"   # Orange
    STATE_COMPLETED = "#8BC34A" # Light Green
    STATE_MIGRATING = "#9C27B0" # Purple
    
    # Priority colors
    PRIORITY_HIGH = "#F44336"   # Red
    PRIORITY_MEDIUM = "#FF9800" # Orange
    PRIORITY_LOW = "#4CAF50"    # Green
    
    # Processor colors (for Gantt chart)
    PROCESSOR_COLORS = [
        "#2196F3",  # Blue
        "#4CAF50",  # Green
        "#FF9800",  # Orange
        "#9C27B0",  # Purple
        "#00BCD4",  # Cyan
        "#E91E63",  # Pink
        "#CDDC39",  # Lime
        "#795548",  # Brown
    ]
    
    # Button colors
    BUTTON_START = "#4CAF50"
    BUTTON_STOP = "#F44336"
    BUTTON_PAUSE = "#FF9800"
    BUTTON_RESET = "#2196F3"
    
    @staticmethod
    def get_load_color(load_percentage: float) -> str:
        """Get color based on load percentage."""
        if load_percentage < 0.4:
            return ColorScheme.LOAD_LOW
        elif load_percentage < 0.7:
            return ColorScheme.LOAD_MEDIUM
        else:
            return ColorScheme.LOAD_HIGH
    
    @staticmethod
    def get_state_color(state: ProcessState) -> str:
        """Get color for process state."""
        colors = {
            ProcessState.NEW: ColorScheme.STATE_NEW,
            ProcessState.READY: ColorScheme.STATE_READY,
            ProcessState.RUNNING: ColorScheme.STATE_RUNNING,
            ProcessState.WAITING: ColorScheme.STATE_WAITING,
            ProcessState.COMPLETED: ColorScheme.STATE_COMPLETED,
            ProcessState.MIGRATING: ColorScheme.STATE_MIGRATING,
        }
        return colors.get(state, ColorScheme.STATE_NEW)
    
    @staticmethod
    def get_processor_color(processor_id: int) -> str:
        """Get color for a processor."""
        return ColorScheme.PROCESSOR_COLORS[processor_id % len(ColorScheme.PROCESSOR_COLORS)]


# =============================================================================
# CUSTOM WIDGETS
# =============================================================================

class LoadBar(tk.Canvas):
    """
    Custom widget displaying processor load as a colored bar.
    
    Visual representation of CPU utilization:
    - Green: Low load (0-40%)
    - Yellow: Medium load (40-70%)
    - Red: High load (70-100%)
    """
    
    def __init__(self, parent, width=200, height=30, **kwargs):
        super().__init__(parent, width=width, height=height, 
                         bg=ColorScheme.PANEL_BG, highlightthickness=1,
                         highlightbackground="#cccccc", **kwargs)
        self.bar_width = width - 4
        self.bar_height = height - 4
        self._load = 0.0
        self._draw_bar()
    
    def _draw_bar(self):
        """Draw the load bar."""
        self.delete("all")
        
        # Background
        self.create_rectangle(2, 2, self.bar_width + 2, self.bar_height + 2,
                             fill="#e0e0e0", outline="")
        
        # Load bar
        if self._load > 0:
            fill_width = int(self.bar_width * min(1.0, self._load))
            color = ColorScheme.get_load_color(self._load)
            self.create_rectangle(2, 2, fill_width + 2, self.bar_height + 2,
                                 fill=color, outline="")
        
        # Percentage text
        text = f"{self._load * 100:.1f}%"
        self.create_text(self.bar_width // 2 + 2, self.bar_height // 2 + 2,
                        text=text, fill=ColorScheme.TEXT, font=("Arial", 10, "bold"))
    
    def set_load(self, load: float):
        """Set the load value (0.0 to 1.0)."""
        self._load = max(0.0, min(1.0, load))
        self._draw_bar()


class ProcessorWidget(ttk.Frame):
    """
    Widget displaying a single processor's status.
    
    Shows:
    - Processor ID and name
    - Current load bar
    - Queue size
    - Current running process
    """
    
    def __init__(self, parent, processor_id: int, **kwargs):
        super().__init__(parent, **kwargs)
        self.processor_id = processor_id
        self._create_widgets()
    
    def _create_widgets(self):
        """Create the processor display widgets."""
        # Header with processor ID
        header_frame = ttk.Frame(self)
        header_frame.pack(fill=tk.X, padx=5, pady=2)
        
        color = ColorScheme.get_processor_color(self.processor_id)
        self.id_label = tk.Label(header_frame, text=f"P{self.processor_id}",
                                  font=("Arial", 12, "bold"), fg=color)
        self.id_label.pack(side=tk.LEFT)
        
        self.status_label = ttk.Label(header_frame, text="Idle",
                                       font=("Arial", 10))
        self.status_label.pack(side=tk.RIGHT)
        
        # Load bar
        self.load_bar = LoadBar(self, width=180, height=25)
        self.load_bar.pack(padx=5, pady=2)
        
        # Info row
        info_frame = ttk.Frame(self)
        info_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.queue_label = ttk.Label(info_frame, text="Queue: 0",
                                      font=("Arial", 9))
        self.queue_label.pack(side=tk.LEFT)
        
        self.current_label = ttk.Label(info_frame, text="Running: -",
                                        font=("Arial", 9))
        self.current_label.pack(side=tk.RIGHT)
    
    def update_display(self, load: float, queue_size: int, 
                       current_process: Optional[int], utilization: float):
        """Update the processor display."""
        self.load_bar.set_load(utilization)
        self.queue_label.config(text=f"Queue: {queue_size}")
        
        if current_process is not None:
            self.current_label.config(text=f"Running: P{current_process}")
            self.status_label.config(text="Active")
        else:
            self.current_label.config(text="Running: -")
            self.status_label.config(text="Idle")


class MetricCard(ttk.Frame):
    """
    Card widget displaying a single metric with label and value.
    """
    
    def __init__(self, parent, label: str, value: str = "0", **kwargs):
        super().__init__(parent, **kwargs)
        self._create_widgets(label, value)
    
    def _create_widgets(self, label: str, value: str):
        """Create the metric card widgets."""
        self.configure(relief="groove", borderwidth=1, padding=5)
        
        self.label = ttk.Label(self, text=label, font=("Arial", 9))
        self.label.pack(anchor=tk.W)
        
        self.value_label = ttk.Label(self, text=value, 
                                      font=("Arial", 14, "bold"))
        self.value_label.pack(anchor=tk.W)
    
    def set_value(self, value: str):
        """Update the displayed value."""
        self.value_label.config(text=value)


# =============================================================================
# MAIN GUI CLASS
# =============================================================================

class LoadBalancerGUI:
    """
    Main GUI application for the Dynamic Load Balancing Simulator.
    
    This class manages:
    - Main window and layout
    - Control panel for configuration
    - Real-time visualization of processors
    - Gantt chart for process execution
    - Metrics dashboard
    - Thread-safe simulation updates
    
    Thread Safety:
    The GUI runs on the main thread while simulation runs in a background
    thread. Updates are coordinated using a queue and the after() method.
    """
    
    def __init__(self):
        """Initialize the GUI application."""
        # Create main window
        self.root = tk.Tk()
        self.root.title(f"{APP_NAME} v{VERSION}")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 700)
        
        # Configuration
        self.config = DEFAULT_SIMULATION_CONFIG
        self.gui_config = DEFAULT_GUI_CONFIG
        
        # Simulation components
        self.engine: Optional[SimulationEngine] = None
        self.simulation_thread: Optional[threading.Thread] = None
        
        # Update queue for thread-safe GUI updates
        self.update_queue = queue.Queue()
        
        # State tracking
        self.is_running = False
        self.is_paused = False
        
        # Processor widgets
        self.processor_widgets: List[ProcessorWidget] = []
        
        # Metric cards
        self.metric_cards: Dict[str, MetricCard] = {}
        
        # Gantt chart data
        self.gantt_data: List[Dict] = []
        
        # Create GUI components
        self._create_styles()
        self._create_menu()
        self._create_main_layout()
        
        # Bind events
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Start update loop
        self._process_updates()
    
    def _create_styles(self):
        """Configure ttk styles for the application."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure frame styles
        style.configure("Panel.TFrame", background=ColorScheme.PANEL_BG)
        style.configure("Header.TLabel", font=("Arial", 14, "bold"))
        style.configure("SubHeader.TLabel", font=("Arial", 11, "bold"))
        
        # Button styles
        style.configure("Start.TButton", foreground=ColorScheme.BUTTON_START)
        style.configure("Stop.TButton", foreground=ColorScheme.BUTTON_STOP)
        style.configure("Pause.TButton", foreground=ColorScheme.BUTTON_PAUSE)
        style.configure("Reset.TButton", foreground=ColorScheme.BUTTON_RESET)
    
    def _create_menu(self):
        """Create the application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Export Results...", command=self._export_results)
        file_menu.add_command(label="Save Configuration...", command=self._save_config)
        file_menu.add_command(label="Load Configuration...", command=self._load_config)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_close)
        
        # Simulation menu
        sim_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Simulation", menu=sim_menu)
        sim_menu.add_command(label="Start", command=self._start_simulation)
        sim_menu.add_command(label="Pause/Resume", command=self._toggle_pause)
        sim_menu.add_command(label="Stop", command=self._stop_simulation)
        sim_menu.add_command(label="Reset", command=self._reset_simulation)
        sim_menu.add_separator()
        sim_menu.add_command(label="Compare Algorithms", command=self._compare_algorithms)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
        help_menu.add_command(label="Algorithm Info", command=self._show_algorithm_info)
    
    def _create_main_layout(self):
        """Create the main window layout."""
        # Main container
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Top section: Control Panel
        self._create_control_panel(main_frame)
        
        # Middle section: Visualization (Processors + Gantt Chart)
        middle_frame = ttk.Frame(main_frame)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Left: Processors
        self._create_processor_panel(middle_frame)
        
        # Right: Gantt Chart and Process Table
        right_frame = ttk.Frame(middle_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        self._create_gantt_chart(right_frame)
        self._create_process_table(right_frame)
        
        # Bottom section: Metrics Dashboard
        self._create_metrics_panel(main_frame)
    
    def _create_control_panel(self, parent):
        """Create the control panel with configuration options."""
        panel = ttk.LabelFrame(parent, text="Control Panel", padding=10)
        panel.pack(fill=tk.X)
        
        # Configuration row
        config_frame = ttk.Frame(panel)
        config_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Number of Processors
        ttk.Label(config_frame, text="Processors:").pack(side=tk.LEFT, padx=(0, 5))
        self.processors_var = tk.IntVar(value=self.config.num_processors)
        processors_spin = ttk.Spinbox(config_frame, from_=2, to=8, width=5,
                                       textvariable=self.processors_var)
        processors_spin.pack(side=tk.LEFT, padx=(0, 20))
        
        # Number of Processes
        ttk.Label(config_frame, text="Processes:").pack(side=tk.LEFT, padx=(0, 5))
        self.processes_var = tk.IntVar(value=self.config.num_processes)
        processes_spin = ttk.Spinbox(config_frame, from_=5, to=100, width=5,
                                      textvariable=self.processes_var)
        processes_spin.pack(side=tk.LEFT, padx=(0, 20))
        
        # Time Quantum
        ttk.Label(config_frame, text="Time Quantum:").pack(side=tk.LEFT, padx=(0, 5))
        self.quantum_var = tk.IntVar(value=self.config.time_quantum)
        quantum_spin = ttk.Spinbox(config_frame, from_=1, to=20, width=5,
                                    textvariable=self.quantum_var)
        quantum_spin.pack(side=tk.LEFT, padx=(0, 20))
        
        # Algorithm Selection
        ttk.Label(config_frame, text="Algorithm:").pack(side=tk.LEFT, padx=(0, 5))
        self.algorithm_var = tk.StringVar(value=self.config.default_algorithm.value)
        algorithm_combo = ttk.Combobox(config_frame, textvariable=self.algorithm_var,
                                        values=[a.value for a in LoadBalancingAlgorithm],
                                        state="readonly", width=20)
        algorithm_combo.pack(side=tk.LEFT, padx=(0, 20))
        
        # Speed slider
        ttk.Label(config_frame, text="Speed:").pack(side=tk.LEFT, padx=(0, 5))
        self.speed_var = tk.IntVar(value=50)
        speed_scale = ttk.Scale(config_frame, from_=1, to=100, orient=tk.HORIZONTAL,
                                 variable=self.speed_var, length=100)
        speed_scale.pack(side=tk.LEFT, padx=(0, 5))
        self.speed_label = ttk.Label(config_frame, text="50%")
        self.speed_label.pack(side=tk.LEFT)
        speed_scale.bind("<Motion>", self._update_speed_label)
        
        # Control buttons row
        button_frame = ttk.Frame(panel)
        button_frame.pack(fill=tk.X)
        
        self.start_btn = ttk.Button(button_frame, text="▶ Start", 
                                     command=self._start_simulation, width=12)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.pause_btn = ttk.Button(button_frame, text="⏸ Pause",
                                     command=self._toggle_pause, width=12,
                                     state=tk.DISABLED)
        self.pause_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="⏹ Stop",
                                    command=self._stop_simulation, width=12,
                                    state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_btn = ttk.Button(button_frame, text="↺ Reset",
                                     command=self._reset_simulation, width=12)
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Separator(button_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, 
                                                               fill=tk.Y, padx=10)
        
        self.compare_btn = ttk.Button(button_frame, text="📊 Compare Algorithms",
                                       command=self._compare_algorithms, width=18)
        self.compare_btn.pack(side=tk.LEFT, padx=5)
        
        # Status bar
        status_frame = ttk.Frame(panel)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_label = ttk.Label(status_frame, text="Status: Ready",
                                       font=("Arial", 10))
        self.status_label.pack(side=tk.LEFT)
        
        self.time_label = ttk.Label(status_frame, text="Time: 0",
                                     font=("Arial", 10))
        self.time_label.pack(side=tk.RIGHT)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var,
                                             maximum=100, length=200)
        self.progress_bar.pack(side=tk.RIGHT, padx=10)
    
    def _create_processor_panel(self, parent):
        """Create the processor visualization panel."""
        panel = ttk.LabelFrame(parent, text="Processors", padding=10)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        # Container for processor widgets
        self.processor_container = ttk.Frame(panel)
        self.processor_container.pack(fill=tk.BOTH, expand=True)
        
        # Create initial processor widgets
        self._create_processor_widgets()
    
    def _create_processor_widgets(self):
        """Create processor display widgets based on configuration."""
        # Clear existing widgets
        for widget in self.processor_widgets:
            widget.destroy()
        self.processor_widgets.clear()
        
        # Create new widgets
        num_processors = self.processors_var.get()
        for i in range(num_processors):
            widget = ProcessorWidget(self.processor_container, processor_id=i)
            widget.pack(fill=tk.X, pady=5)
            self.processor_widgets.append(widget)
    
    def _create_gantt_chart(self, parent):
        """Create the Gantt chart visualization."""
        panel = ttk.LabelFrame(parent, text="Process Execution Timeline (Gantt Chart)", 
                               padding=10)
        panel.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.gantt_fig = Figure(figsize=(8, 3), dpi=100)
        self.gantt_ax = self.gantt_fig.add_subplot(111)
        self.gantt_ax.set_xlabel("Time")
        self.gantt_ax.set_ylabel("Processor")
        self.gantt_fig.tight_layout()
        
        # Embed in Tkinter
        self.gantt_canvas = FigureCanvasTkAgg(self.gantt_fig, master=panel)
        self.gantt_canvas.draw()
        self.gantt_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _create_process_table(self, parent):
        """Create the process table display."""
        panel = ttk.LabelFrame(parent, text="Process Details", padding=10)
        panel.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Create treeview with scrollbar
        tree_frame = ttk.Frame(panel)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ("PID", "Arrival", "Burst", "Remaining", "Priority", 
                   "State", "Processor", "Wait", "Turnaround")
        
        self.process_tree = ttk.Treeview(tree_frame, columns=columns, 
                                          show="headings", height=8)
        
        # Configure columns
        col_widths = {"PID": 50, "Arrival": 60, "Burst": 60, "Remaining": 70,
                      "Priority": 70, "State": 80, "Processor": 70, 
                      "Wait": 50, "Turnaround": 80}
        
        for col in columns:
            self.process_tree.heading(col, text=col)
            self.process_tree.column(col, width=col_widths.get(col, 70), 
                                      anchor=tk.CENTER)
        
        # Scrollbars
        vsb = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, 
                            command=self.process_tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL,
                            command=self.process_tree.xview)
        self.process_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Grid layout
        self.process_tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        tree_frame.grid_rowconfigure(0, weight=1)
        tree_frame.grid_columnconfigure(0, weight=1)
    
    def _create_metrics_panel(self, parent):
        """Create the metrics dashboard."""
        panel = ttk.LabelFrame(parent, text="Performance Metrics", padding=10)
        panel.pack(fill=tk.X, pady=(0, 0))
        
        # Metrics grid
        metrics_frame = ttk.Frame(panel)
        metrics_frame.pack(fill=tk.X)
        
        # Row 1: Process metrics
        row1 = ttk.Frame(metrics_frame)
        row1.pack(fill=tk.X, pady=5)
        
        metrics_row1 = [
            ("Completed", "completed", "0/0"),
            ("Avg Turnaround", "turnaround", "0.00"),
            ("Avg Waiting", "waiting", "0.00"),
            ("Avg Response", "response", "0.00"),
        ]
        
        for label, key, default in metrics_row1:
            card = MetricCard(row1, label, default)
            card.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            self.metric_cards[key] = card
        
        # Row 2: System metrics
        row2 = ttk.Frame(metrics_frame)
        row2.pack(fill=tk.X, pady=5)
        
        metrics_row2 = [
            ("Avg Utilization", "utilization", "0.0%"),
            ("Load Balance Index", "lbi", "0.0000"),
            ("Jain's Fairness", "fairness", "0.0000"),
            ("Migrations", "migrations", "0"),
        ]
        
        for label, key, default in metrics_row2:
            card = MetricCard(row2, label, default)
            card.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
            self.metric_cards[key] = card
        
        # Comparison chart button area
        chart_frame = ttk.Frame(panel)
        chart_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(chart_frame, text="Algorithm: ").pack(side=tk.LEFT)
        self.current_algo_label = ttk.Label(chart_frame, text="-", 
                                             font=("Arial", 10, "bold"))
        self.current_algo_label.pack(side=tk.LEFT)
    
    def _update_speed_label(self, event=None):
        """Update the speed label when slider changes."""
        self.speed_label.config(text=f"{self.speed_var.get()}%")
    
    # =========================================================================
    # SIMULATION CONTROL METHODS
    # =========================================================================
    
    def _start_simulation(self):
        """Start the simulation."""
        if self.is_running:
            return
        
        # Update configuration from GUI
        self._update_config()
        
        # Recreate processor widgets if count changed
        if len(self.processor_widgets) != self.config.num_processors:
            self._create_processor_widgets()
        
        # Create and initialize simulation engine
        self.engine = SimulationEngine(self.config, self.gui_config)
        
        # Get selected algorithm
        algo_name = self.algorithm_var.get()
        algorithm = next(a for a in LoadBalancingAlgorithm if a.value == algo_name)
        
        if not self.engine.initialize(algorithm=algorithm):
            messagebox.showerror("Error", "Failed to initialize simulation")
            return
        
        # Clear previous data
        self.gantt_data.clear()
        self._clear_process_table()
        self._reset_metrics()
        
        # Populate process table
        self._populate_process_table()
        
        # Update UI state
        self.is_running = True
        self.is_paused = False
        self._update_button_states()
        self.status_label.config(text="Status: Running")
        self.current_algo_label.config(text=algorithm.value)
        
        # Start simulation in background thread
        self.simulation_thread = threading.Thread(target=self._run_simulation, 
                                                   daemon=True)
        self.simulation_thread.start()
    
    def _run_simulation(self):
        """Run the simulation (called in background thread)."""
        while self.is_running and not self.engine.is_complete():
            if not self.is_paused:
                # Execute one step (time increments at end of step)
                self.engine.step()
                
                # Record what was running during the step that just completed
                # current_time is now T+1, so the slot that ran was T to T+1
                self._record_gantt_step()
                
                # Queue update for GUI
                self.update_queue.put(("update", self.engine.get_current_state()))
                
                # Calculate delay based on speed setting
                speed = self.speed_var.get()
                delay = (101 - speed) / 100.0 * 0.2  # 0 to 0.2 seconds
                time.sleep(delay)
            else:
                time.sleep(0.1)  # Check pause state periodically
        
        # Simulation complete
        if self.engine.is_complete():
            result = self.engine.get_result()
            self.update_queue.put(("complete", result))
    
    def _toggle_pause(self):
        """Toggle pause state."""
        if not self.is_running:
            return
        
        self.is_paused = not self.is_paused
        
        if self.is_paused:
            self.pause_btn.config(text="▶ Resume")
            self.status_label.config(text="Status: Paused")
        else:
            self.pause_btn.config(text="⏸ Pause")
            self.status_label.config(text="Status: Running")
    
    def _stop_simulation(self):
        """Stop the simulation."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.engine:
            self.engine.stop()
        
        self._update_button_states()
        self.status_label.config(text="Status: Stopped")
    
    def _reset_simulation(self):
        """Reset the simulation."""
        self._stop_simulation()
        
        # Clear visualizations
        self._clear_gantt_chart()
        self._clear_process_table()
        self._reset_metrics()
        self._reset_processor_displays()
        
        self.status_label.config(text="Status: Ready")
        self.time_label.config(text="Time: 0")
        self.progress_var.set(0)
        self.current_algo_label.config(text="-")
    
    def _update_config(self):
        """Update configuration from GUI inputs."""
        self.config = SimulationConfig(
            num_processors=self.processors_var.get(),
            num_processes=self.processes_var.get(),
            time_quantum=self.quantum_var.get()
        )
    
    def _update_button_states(self):
        """Update button enabled states based on simulation state."""
        if self.is_running:
            self.start_btn.config(state=tk.DISABLED)
            self.pause_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.NORMAL)
            self.reset_btn.config(state=tk.DISABLED)
            self.compare_btn.config(state=tk.DISABLED)
        else:
            self.start_btn.config(state=tk.NORMAL)
            self.pause_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.DISABLED)
            self.reset_btn.config(state=tk.NORMAL)
            self.compare_btn.config(state=tk.NORMAL)
    
    # =========================================================================
    # UPDATE METHODS (Thread-safe via queue)
    # =========================================================================
    
    def _process_updates(self):
        """Process updates from the simulation thread (runs on main thread)."""
        try:
            while True:
                msg_type, data = self.update_queue.get_nowait()
                
                if msg_type == "update":
                    self._handle_state_update(data)
                elif msg_type == "complete":
                    self._handle_completion(data)
                    
        except queue.Empty:
            pass
        
        # Schedule next check
        self.root.after(50, self._process_updates)
    
    def _handle_state_update(self, state: Dict[str, Any]):
        """Handle simulation state update."""
        # Update time display
        self.time_label.config(text=f"Time: {state['time']}")
        
        # Update progress
        total = state['total_processes']
        completed = state['completed']
        progress = (completed / total * 100) if total > 0 else 0
        self.progress_var.set(progress)
        
        # Update processor displays
        for i, proc_data in enumerate(state['processors']):
            if i < len(self.processor_widgets):
                self.processor_widgets[i].update_display(
                    load=proc_data['load'],
                    queue_size=proc_data['queue_size'],
                    current_process=proc_data['current_process'],
                    utilization=proc_data['utilization']
                )
        
        # Update process table
        self._update_process_table()
        
        # Update Gantt chart more frequently (every 2 time units or when there's data)
        if state['time'] % 2 == 0 or state['time'] <= 5:
            self._update_gantt_chart()
        
        # Update basic metrics
        self.metric_cards['completed'].set_value(f"{completed}/{total}")
        self.metric_cards['migrations'].set_value(str(state['migrations']))
    
    def _handle_completion(self, result):
        """Handle simulation completion."""
        self.is_running = False
        self._update_button_states()
        self.status_label.config(text="Status: Completed")
        
        # Update final metrics
        metrics = result.system_metrics
        self.metric_cards['completed'].set_value(
            f"{metrics.completed_processes}/{metrics.total_processes}")
        self.metric_cards['turnaround'].set_value(f"{metrics.avg_turnaround_time:.2f}")
        self.metric_cards['waiting'].set_value(f"{metrics.avg_waiting_time:.2f}")
        self.metric_cards['response'].set_value(f"{metrics.avg_response_time:.2f}")
        self.metric_cards['utilization'].set_value(f"{metrics.avg_utilization*100:.1f}%")
        self.metric_cards['lbi'].set_value(f"{metrics.load_balance_index:.4f}")
        self.metric_cards['fairness'].set_value(f"{metrics.jains_fairness_index:.4f}")
        self.metric_cards['migrations'].set_value(str(metrics.total_migrations))
        
        # Final gantt chart update
        self._update_gantt_chart()
        
        # Final process table update
        self._update_process_table()
        
        messagebox.showinfo("Simulation Complete", 
                           f"Simulation completed in {result.total_time} time units.\n"
                           f"Completed: {metrics.completed_processes}/{metrics.total_processes} processes\n"
                           f"Average Turnaround: {metrics.avg_turnaround_time:.2f}\n"
                           f"Average Utilization: {metrics.avg_utilization*100:.1f}%")
    
    # =========================================================================
    # VISUALIZATION METHODS
    # =========================================================================
    
    def _record_gantt_step(self):
        """
        Record processor activity for Gantt chart.
        
        Called AFTER step() completes. Uses the processor's execution_history
        which tracks actual execution segments.
        """
        if not self.engine or not self.engine.processor_manager:
            return
        
        current_time = self.engine.current_time
        if current_time < 1:
            return
        
        slot_start = current_time - 1
        slot_end = current_time
            
        for proc in self.engine.processor_manager:
            # Method 1: Check current process (still running)
            if proc.current_process:
                self.gantt_data.append({
                    'processor': proc.processor_id,
                    'process': proc.current_process.pid,
                    'start': slot_start,
                    'end': slot_end
                })
            # Method 2: Check execution history for recently completed processes
            elif hasattr(proc, 'execution_history') and proc.execution_history:
                # Check if there was execution in this time slot
                for entry in reversed(proc.execution_history):
                    if entry['end'] == current_time:
                        self.gantt_data.append({
                            'processor': proc.processor_id,
                            'process': entry['pid'],
                            'start': slot_start,
                            'end': slot_end
                        })
                        break
    
    def _record_gantt_step_before(self):
        """Record current state for Gantt chart BEFORE step executes (deprecated)."""
        pass
    
    def _update_gantt_chart(self):
        """Update the Gantt chart visualization."""
        self.gantt_ax.clear()
        
        if not self.gantt_data:
            self.gantt_ax.set_xlabel("Time")
            self.gantt_ax.set_ylabel("Processor")
            self.gantt_canvas.draw()
            return
        
        # Consolidate consecutive entries for same process
        consolidated = []
        for entry in self.gantt_data:
            if consolidated and consolidated[-1]['processor'] == entry['processor'] and \
               consolidated[-1]['process'] == entry['process'] and \
               consolidated[-1]['end'] == entry['start']:
                consolidated[-1]['end'] = entry['end']
            else:
                consolidated.append(entry.copy())
        
        # Draw rectangles
        num_processors = self.processors_var.get()
        process_colors = {}
        color_index = 0
        
        for entry in consolidated:
            pid = entry['process']
            if pid not in process_colors:
                process_colors[pid] = ColorScheme.PROCESSOR_COLORS[color_index % len(ColorScheme.PROCESSOR_COLORS)]
                color_index += 1
            
            rect = Rectangle(
                (entry['start'], entry['processor'] - 0.4),
                entry['end'] - entry['start'],
                0.8,
                facecolor=process_colors[pid],
                edgecolor='black',
                linewidth=0.5
            )
            self.gantt_ax.add_patch(rect)
            
            # Add process label if wide enough
            width = entry['end'] - entry['start']
            if width >= 2:
                self.gantt_ax.text(
                    entry['start'] + width/2,
                    entry['processor'],
                    f"P{pid}",
                    ha='center', va='center',
                    fontsize=8, fontweight='bold'
                )
        
        # Configure axes
        max_time = max(e['end'] for e in consolidated) if consolidated else 10
        self.gantt_ax.set_xlim(0, max_time + 1)
        self.gantt_ax.set_ylim(-0.5, num_processors - 0.5)
        self.gantt_ax.set_yticks(range(num_processors))
        self.gantt_ax.set_yticklabels([f"P{i}" for i in range(num_processors)])
        self.gantt_ax.set_xlabel("Time")
        self.gantt_ax.set_ylabel("Processor")
        self.gantt_ax.grid(True, axis='x', alpha=0.3)
        
        self.gantt_fig.tight_layout()
        self.gantt_canvas.draw()
    
    def _clear_gantt_chart(self):
        """Clear the Gantt chart."""
        self.gantt_data.clear()
        self.gantt_ax.clear()
        self.gantt_ax.set_xlabel("Time")
        self.gantt_ax.set_ylabel("Processor")
        self.gantt_canvas.draw()
    
    def _populate_process_table(self):
        """Populate the process table with initial data."""
        self._clear_process_table()
        
        if not self.engine:
            return
        
        for p in self.engine.all_processes:
            self.process_tree.insert("", tk.END, iid=str(p.pid), values=(
                p.pid,
                p.arrival_time,
                p.burst_time,
                p.remaining_time,
                p.priority.name,
                p.state.name,
                p.processor_id if p.processor_id is not None else "-",
                p.waiting_time,
                "-"
            ))
    
    def _update_process_table(self):
        """Update process table with current data."""
        if not self.engine:
            return
        
        for p in self.engine.all_processes:
            turnaround = p.get_turnaround_time()
            turnaround_str = str(turnaround) if turnaround is not None else "-"
            
            try:
                self.process_tree.item(str(p.pid), values=(
                    p.pid,
                    p.arrival_time,
                    p.burst_time,
                    p.remaining_time,
                    p.priority.name,
                    p.state.name,
                    p.processor_id if p.processor_id is not None else "-",
                    p.waiting_time,
                    turnaround_str
                ))
            except tk.TclError:
                pass  # Item may not exist
    
    def _clear_process_table(self):
        """Clear all items from the process table."""
        for item in self.process_tree.get_children():
            self.process_tree.delete(item)
    
    def _reset_metrics(self):
        """Reset all metric displays."""
        self.metric_cards['completed'].set_value("0/0")
        self.metric_cards['turnaround'].set_value("0.00")
        self.metric_cards['waiting'].set_value("0.00")
        self.metric_cards['response'].set_value("0.00")
        self.metric_cards['utilization'].set_value("0.0%")
        self.metric_cards['lbi'].set_value("0.0000")
        self.metric_cards['fairness'].set_value("0.0000")
        self.metric_cards['migrations'].set_value("0")
    
    def _reset_processor_displays(self):
        """Reset all processor displays."""
        for widget in self.processor_widgets:
            widget.update_display(load=0, queue_size=0, 
                                  current_process=None, utilization=0)
    
    # =========================================================================
    # COMPARISON AND ANALYSIS METHODS
    # =========================================================================
    
    def _compare_algorithms(self):
        """Run comparison across all algorithms and show results."""
        # Update config
        self._update_config()
        
        # Show progress dialog
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Comparing Algorithms")
        progress_window.geometry("400x150")
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        ttk.Label(progress_window, text="Running simulations for each algorithm...",
                  font=("Arial", 12)).pack(pady=20)
        
        progress_bar = ttk.Progressbar(progress_window, length=300, mode='determinate')
        progress_bar.pack(pady=10)
        
        status_label = ttk.Label(progress_window, text="")
        status_label.pack(pady=10)
        
        self.root.update()
        
        # Run batch comparison
        batch = BatchSimulator(self.config)
        algorithms = list(LoadBalancingAlgorithm)
        results = {}
        
        for i, algo in enumerate(algorithms):
            status_label.config(text=f"Running {algo.value}...")
            progress_bar['value'] = (i / len(algorithms)) * 100
            self.root.update()
            
            # Create and run simulation for this algorithm
            engine = SimulationEngine(self.config)
            engine.initialize(algorithm=algo)
            result = engine.run()
            results[algo] = result
            batch.results[algo] = result
            batch.comparator.add_result(algo, result.system_metrics)
        
        progress_bar['value'] = 100
        progress_window.destroy()
        
        # Show comparison results
        self._show_comparison_results(results, batch)
    
    def _show_comparison_results(self, results: Dict, batch: BatchSimulator):
        """Display comparison results in a new window."""
        window = tk.Toplevel(self.root)
        window.title("Algorithm Comparison Results")
        window.geometry("1000x700")
        
        # Create notebook for tabs
        notebook = ttk.Notebook(window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Summary Table
        table_frame = ttk.Frame(notebook)
        notebook.add(table_frame, text="Summary")
        
        columns = ("Algorithm", "Time", "Avg Turnaround", "Avg Waiting",
                   "Utilization", "LBI", "Fairness", "Migrations")
        
        tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=5)
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=110, anchor=tk.CENTER)
        
        for algo, result in results.items():
            m = result.system_metrics
            tree.insert("", tk.END, values=(
                algo.value,
                result.total_time,
                f"{m.avg_turnaround_time:.2f}",
                f"{m.avg_waiting_time:.2f}",
                f"{m.avg_utilization*100:.1f}%",
                f"{m.load_balance_index:.4f}",
                f"{m.jains_fairness_index:.4f}",
                m.total_migrations
            ))
        
        tree.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Best algorithm summary
        best_frame = ttk.LabelFrame(table_frame, text="Best Algorithm By Metric", padding=10)
        best_frame.pack(fill=tk.X, pady=10)
        
        best_turnaround = batch.get_best_algorithm('avg_turnaround_time')
        best_waiting = batch.get_best_algorithm('avg_waiting_time')
        best_fairness = batch.get_best_algorithm('jains_fairness_index')
        
        ttk.Label(best_frame, text=f"Lowest Turnaround Time: {best_turnaround}",
                  font=("Arial", 10)).pack(anchor=tk.W)
        ttk.Label(best_frame, text=f"Lowest Waiting Time: {best_waiting}",
                  font=("Arial", 10)).pack(anchor=tk.W)
        ttk.Label(best_frame, text=f"Best Fairness: {best_fairness}",
                  font=("Arial", 10)).pack(anchor=tk.W)
        
        # Tab 2: Comparison Charts
        chart_frame = ttk.Frame(notebook)
        notebook.add(chart_frame, text="Charts")
        
        fig = Figure(figsize=(10, 6), dpi=100)
        
        # Turnaround Time comparison
        ax1 = fig.add_subplot(221)
        algo_names = [a.value for a in results.keys()]
        turnaround_values = [r.system_metrics.avg_turnaround_time for r in results.values()]
        bars = ax1.bar(algo_names, turnaround_values, color=['#2196F3', '#4CAF50', '#FF9800'])
        ax1.set_ylabel("Time")
        ax1.set_title("Average Turnaround Time")
        ax1.tick_params(axis='x', rotation=15)
        
        # Utilization comparison
        ax2 = fig.add_subplot(222)
        util_values = [r.system_metrics.avg_utilization * 100 for r in results.values()]
        bars = ax2.bar(algo_names, util_values, color=['#2196F3', '#4CAF50', '#FF9800'])
        ax2.set_ylabel("Percentage")
        ax2.set_title("Average CPU Utilization")
        ax2.tick_params(axis='x', rotation=15)
        
        # Fairness Index comparison
        ax3 = fig.add_subplot(223)
        fairness_values = [r.system_metrics.jains_fairness_index for r in results.values()]
        bars = ax3.bar(algo_names, fairness_values, color=['#2196F3', '#4CAF50', '#FF9800'])
        ax3.set_ylabel("Index (0-1)")
        ax3.set_title("Jain's Fairness Index")
        ax3.tick_params(axis='x', rotation=15)
        
        # Migrations comparison
        ax4 = fig.add_subplot(224)
        migration_values = [r.system_metrics.total_migrations for r in results.values()]
        bars = ax4.bar(algo_names, migration_values, color=['#2196F3', '#4CAF50', '#FF9800'])
        ax4.set_ylabel("Count")
        ax4.set_title("Process Migrations")
        ax4.tick_params(axis='x', rotation=15)
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(canvas, chart_frame)
        toolbar.update()
        
        # Export button
        export_frame = ttk.Frame(window)
        export_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def export_comparison():
            filepath = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if filepath:
                data = {
                    'comparison_date': time.strftime("%Y-%m-%d %H:%M:%S"),
                    'config': self.config.to_dict(),
                    'results': {
                        algo.value: result.to_dict()
                        for algo, result in results.items()
                    }
                }
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)
                messagebox.showinfo("Export Complete", f"Results exported to {filepath}")
        
        ttk.Button(export_frame, text="Export Results", 
                   command=export_comparison).pack(side=tk.RIGHT)
    
    # =========================================================================
    # FILE OPERATIONS
    # =========================================================================
    
    def _export_results(self):
        """Export current simulation results."""
        if not self.engine or not self.engine.metrics_calculator:
            messagebox.showwarning("No Data", "No simulation results to export.")
            return
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                self.engine.metrics_calculator.export_to_json(filepath)
                messagebox.showinfo("Export Complete", f"Results exported to {filepath}")
            except Exception as e:
                messagebox.showerror("Export Error", str(e))
    
    def _save_config(self):
        """Save current configuration to file."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filepath:
            self._update_config()
            config_data = self.config.to_dict()
            config_data['algorithm'] = self.algorithm_var.get()
            
            with open(filepath, 'w') as f:
                json.dump(config_data, f, indent=2)
            messagebox.showinfo("Save Complete", f"Configuration saved to {filepath}")
    
    def _load_config(self):
        """Load configuration from file."""
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                with open(filepath, 'r') as f:
                    config_data = json.load(f)
                
                # Update GUI fields
                self.processors_var.set(config_data.get('num_processors', 4))
                self.processes_var.set(config_data.get('num_processes', 20))
                self.quantum_var.set(config_data.get('time_quantum', 4))
                
                if 'algorithm' in config_data:
                    self.algorithm_var.set(config_data['algorithm'])
                
                messagebox.showinfo("Load Complete", "Configuration loaded successfully")
            except Exception as e:
                messagebox.showerror("Load Error", str(e))
    
    # =========================================================================
    # HELP AND INFO
    # =========================================================================
    
    def _show_about(self):
        """Show about dialog."""
        about_text = f"""
{APP_NAME}
Version {VERSION}

An educational simulation demonstrating dynamic 
load balancing algorithms in multiprocessor systems.

Key Features:
• Multiple load balancing algorithms
• Real-time visualization
• Gantt chart process timeline
• Performance metrics and comparison

This project demonstrates Operating System concepts:
• Process Management
• CPU Scheduling
• Load Balancing
• Resource Utilization

Author: Student
Date: December 2024
        """
        messagebox.showinfo("About", about_text)
    
    def _show_algorithm_info(self):
        """Show information about load balancing algorithms."""
        info_window = tk.Toplevel(self.root)
        info_window.title("Load Balancing Algorithms")
        info_window.geometry("600x500")
        
        text = tk.Text(info_window, wrap=tk.WORD, padx=20, pady=20)
        text.pack(fill=tk.BOTH, expand=True)
        
        info_text = """
LOAD BALANCING ALGORITHMS
=========================

1. ROUND ROBIN
--------------
Distribution: Assigns processes to processors in cyclic order (P0→P1→P2→P3→P0→...)

Advantages:
• Simple and fair
• Equal distribution by count
• Low computational overhead
• Deterministic behavior

Disadvantages:
• Ignores actual processor load
• Can cause imbalance with varied process sizes
• No dynamic adaptation

Best for: Homogeneous workloads with similar process sizes


2. LEAST LOADED FIRST
---------------------
Distribution: Assigns each new process to the processor with minimum current load.

Advantages:
• Better load distribution
• Adapts to current system state
• Efficient for varied workloads
• Considers actual work remaining

Disadvantages:
• Slightly higher overhead (requires load monitoring)
• No process migration after initial assignment
• May cause "herd behavior"

Best for: Variable workloads with different burst times


3. THRESHOLD-BASED
------------------
Distribution: Uses least loaded for initial assignment, then migrates processes 
when load difference exceeds a threshold.

Advantages:
• Dynamic rebalancing
• Handles changing workloads
• Prevents severe imbalances
• Combines best of other approaches

Disadvantages:
• Migration overhead (context switch cost)
• Requires careful threshold tuning
• More complex implementation
• May cause oscillation if thresholds wrong

Best for: Dynamic workloads where load changes over time


METRICS EXPLAINED
=================

• Turnaround Time: Total time from process arrival to completion
• Waiting Time: Time spent waiting in ready queue
• CPU Utilization: Percentage of time processor is busy
• Load Balance Index: Measure of load distribution (1.0 = perfect)
• Jain's Fairness Index: Statistical fairness measure (1.0 = perfectly fair)
        """
        
        text.insert(tk.END, info_text)
        text.config(state=tk.DISABLED)
    
    def _on_close(self):
        """Handle window close event."""
        if self.is_running:
            if messagebox.askyesno("Confirm Exit", 
                                   "Simulation is running. Stop and exit?"):
                self._stop_simulation()
            else:
                return
        
        self.root.destroy()
    
    def run(self):
        """Start the GUI application."""
        self.root.mainloop()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    app = LoadBalancerGUI()
    app.run()
