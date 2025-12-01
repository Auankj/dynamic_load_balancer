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

Features:
- Modern dark/light theme with glassmorphism effects
- Animated load bars with gradient fills
- Card-based responsive layout
- Professional color scheme
- Status LED indicators

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
from matplotlib.patches import Rectangle, FancyBboxPatch
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
# MODERN COLOR SCHEMES AND CONSTANTS
# =============================================================================

class ModernColors:
    """
    Modern color scheme with dark mode support.
    Uses a professional color palette with gradients and accents.
    """
    # Background colors
    BG_DARK = "#0f0f1a"
    BG_CARD = "#1a1a2e"
    BG_CARD_HOVER = "#252542"
    BG_INPUT = "#16213e"
    
    # Accent colors
    PRIMARY = "#4361ee"
    PRIMARY_LIGHT = "#4cc9f0"
    SECONDARY = "#7209b7"
    ACCENT = "#f72585"
    
    # Text colors
    TEXT_PRIMARY = "#ffffff"
    TEXT_SECONDARY = "#a0a0b0"
    TEXT_MUTED = "#6c6c7c"
    
    # Status colors
    SUCCESS = "#06d6a0"
    WARNING = "#ffd166"
    DANGER = "#ef476f"
    INFO = "#118ab2"
    
    # Load level colors (gradient-ready)
    LOAD_LOW = "#06d6a0"       # Emerald green
    LOAD_MEDIUM = "#ffd166"    # Amber
    LOAD_HIGH = "#ef476f"      # Coral red
    LOAD_CRITICAL = "#d90429"  # Deep red
    
    # Process state colors
    STATE_NEW = "#6c757d"
    STATE_READY = "#4361ee"
    STATE_RUNNING = "#06d6a0"
    STATE_WAITING = "#ffd166"
    STATE_COMPLETED = "#20c997"
    STATE_MIGRATING = "#7209b7"
    
    # Processor visualization colors
    PROCESSOR_COLORS = [
        "#4361ee",  # Royal Blue
        "#06d6a0",  # Emerald
        "#f72585",  # Pink
        "#7209b7",  # Purple
        "#4cc9f0",  # Cyan
        "#ffd166",  # Amber
        "#ff6b6b",  # Coral
        "#48cae4",  # Sky Blue
    ]
    
    # Gradient pairs for bars
    GRADIENT_SUCCESS = ("#06d6a0", "#20c997")
    GRADIENT_WARNING = ("#ffd166", "#ffbe0b")
    GRADIENT_DANGER = ("#ef476f", "#d90429")
    GRADIENT_INFO = ("#4361ee", "#4cc9f0")
    
    @staticmethod
    def get_load_color(load_percentage: float) -> str:
        """Get color based on load percentage."""
        if load_percentage < 0.4:
            return ModernColors.LOAD_LOW
        elif load_percentage < 0.7:
            return ModernColors.LOAD_MEDIUM
        elif load_percentage < 0.9:
            return ModernColors.LOAD_HIGH
        else:
            return ModernColors.LOAD_CRITICAL
    
    @staticmethod
    def get_load_gradient(load_percentage: float) -> tuple:
        """Get gradient colors based on load percentage."""
        if load_percentage < 0.4:
            return ModernColors.GRADIENT_SUCCESS
        elif load_percentage < 0.7:
            return ModernColors.GRADIENT_WARNING
        else:
            return ModernColors.GRADIENT_DANGER
    
    @staticmethod
    def get_state_color(state: ProcessState) -> str:
        """Get color for process state."""
        colors = {
            ProcessState.NEW: ModernColors.STATE_NEW,
            ProcessState.READY: ModernColors.STATE_READY,
            ProcessState.RUNNING: ModernColors.STATE_RUNNING,
            ProcessState.WAITING: ModernColors.STATE_WAITING,
            ProcessState.COMPLETED: ModernColors.STATE_COMPLETED,
            ProcessState.MIGRATING: ModernColors.STATE_MIGRATING,
        }
        return colors.get(state, ModernColors.STATE_NEW)
    
    @staticmethod
    def get_processor_color(processor_id: int) -> str:
        """Get color for a processor."""
        return ModernColors.PROCESSOR_COLORS[processor_id % len(ModernColors.PROCESSOR_COLORS)]


# Alias for backward compatibility
ColorScheme = ModernColors


# =============================================================================
# MODERN CUSTOM WIDGETS
# =============================================================================

class ModernLoadBar(tk.Canvas):
    """
    Modern animated load bar with gradient fill and rounded corners.
    
    Features:
    - Smooth gradient fills based on load level
    - Rounded corners for modern look
    - Animated transitions
    - Glow effect on high load
    """
    
    def __init__(self, parent, width=220, height=32, **kwargs):
        super().__init__(parent, width=width, height=height,
                         bg=ModernColors.BG_CARD, highlightthickness=0, **kwargs)
        self.bar_width = width - 8
        self.bar_height = height - 8
        self._load = 0.0
        self._target_load = 0.0
        self._animating = False
        self._draw_bar()
    
    def _draw_bar(self):
        """Draw the modern load bar with gradient."""
        self.delete("all")
        
        # Draw background with rounded corners
        self._draw_rounded_rect(4, 4, self.bar_width + 4, self.bar_height + 4,
                               radius=8, fill=ModernColors.BG_INPUT, outline="")
        
        # Draw load bar with gradient effect
        if self._load > 0:
            fill_width = int(self.bar_width * min(1.0, self._load))
            if fill_width > 2:
                color = ModernColors.get_load_color(self._load)
                
                # Create gradient effect by drawing multiple rectangles
                steps = min(fill_width, 20)
                for i in range(steps):
                    alpha = 0.3 + (0.7 * (i / steps))
                    x_start = 4 + (fill_width * i // steps)
                    x_end = 4 + (fill_width * (i + 1) // steps)
                    self.create_rectangle(x_start, 6, x_end, self.bar_height + 2,
                                        fill=color, outline="")
                
                # Add highlight at top
                self.create_rectangle(6, 6, fill_width + 2, 10,
                                    fill=self._lighten_color(color, 0.3), outline="")
        
        # Percentage text with shadow
        text = f"{self._load * 100:.0f}%"
        # Shadow
        self.create_text(self.bar_width // 2 + 5, self.bar_height // 2 + 5,
                        text=text, fill="#000000", font=("Segoe UI", 11, "bold"))
        # Main text
        self.create_text(self.bar_width // 2 + 4, self.bar_height // 2 + 4,
                        text=text, fill=ModernColors.TEXT_PRIMARY, 
                        font=("Segoe UI", 11, "bold"))
    
    def _draw_rounded_rect(self, x1, y1, x2, y2, radius=10, **kwargs):
        """Draw a rounded rectangle."""
        points = [
            x1 + radius, y1,
            x2 - radius, y1,
            x2, y1,
            x2, y1 + radius,
            x2, y2 - radius,
            x2, y2,
            x2 - radius, y2,
            x1 + radius, y2,
            x1, y2,
            x1, y2 - radius,
            x1, y1 + radius,
            x1, y1,
        ]
        return self.create_polygon(points, smooth=True, **kwargs)
    
    def _lighten_color(self, color: str, factor: float) -> str:
        """Lighten a hex color."""
        color = color.lstrip('#')
        r, g, b = int(color[:2], 16), int(color[2:4], 16), int(color[4:], 16)
        r = min(255, int(r + (255 - r) * factor))
        g = min(255, int(g + (255 - g) * factor))
        b = min(255, int(b + (255 - b) * factor))
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def set_load(self, load: float, animate: bool = False):
        """Set the load value (0.0 to 1.0)."""
        self._load = max(0.0, min(1.0, load))
        self._draw_bar()


class ModernProcessorCard(tk.Frame):
    """
    Modern card widget displaying a single processor's status.
    
    Features:
    - Elevated card design with subtle shadow effect
    - LED status indicator
    - Modern typography
    - Animated load visualization
    """
    
    def __init__(self, parent, processor_id: int, **kwargs):
        super().__init__(parent, bg=ModernColors.BG_CARD, **kwargs)
        self.processor_id = processor_id
        self.configure(highlightbackground=ModernColors.BG_CARD_HOVER,
                      highlightthickness=1)
        self._create_widgets()
    
    def _create_widgets(self):
        """Create the modern processor display widgets."""
        # Main container with padding
        container = tk.Frame(self, bg=ModernColors.BG_CARD, padx=12, pady=10)
        container.pack(fill=tk.BOTH, expand=True)
        
        # Header row with processor ID and status LED
        header_frame = tk.Frame(container, bg=ModernColors.BG_CARD)
        header_frame.pack(fill=tk.X, pady=(0, 8))
        
        # Processor icon/badge
        color = ModernColors.get_processor_color(self.processor_id)
        self.badge = tk.Label(header_frame, text=f"CPU {self.processor_id}",
                              font=("Segoe UI", 12, "bold"), fg=color,
                              bg=ModernColors.BG_CARD)
        self.badge.pack(side=tk.LEFT)
        
        # Status LED indicator
        self.led_canvas = tk.Canvas(header_frame, width=12, height=12,
                                     bg=ModernColors.BG_CARD, highlightthickness=0)
        self.led_canvas.pack(side=tk.RIGHT, padx=(0, 5))
        self._draw_led(ModernColors.TEXT_MUTED)
        
        # Status text
        self.status_label = tk.Label(header_frame, text="Idle",
                                      font=("Segoe UI", 10),
                                      fg=ModernColors.TEXT_SECONDARY,
                                      bg=ModernColors.BG_CARD)
        self.status_label.pack(side=tk.RIGHT)
        
        # Modern load bar
        self.load_bar = ModernLoadBar(container, width=200, height=28)
        self.load_bar.pack(pady=(0, 8))
        
        # Stats row with icons
        stats_frame = tk.Frame(container, bg=ModernColors.BG_CARD)
        stats_frame.pack(fill=tk.X)
        
        # Queue indicator
        queue_frame = tk.Frame(stats_frame, bg=ModernColors.BG_CARD)
        queue_frame.pack(side=tk.LEFT)
        
        tk.Label(queue_frame, text="📋", font=("Segoe UI", 10),
                bg=ModernColors.BG_CARD).pack(side=tk.LEFT)
        self.queue_label = tk.Label(queue_frame, text="0",
                                     font=("Segoe UI", 10, "bold"),
                                     fg=ModernColors.TEXT_PRIMARY,
                                     bg=ModernColors.BG_CARD)
        self.queue_label.pack(side=tk.LEFT, padx=(2, 0))
        
        # Current process indicator
        current_frame = tk.Frame(stats_frame, bg=ModernColors.BG_CARD)
        current_frame.pack(side=tk.RIGHT)
        
        tk.Label(current_frame, text="▶", font=("Segoe UI", 10),
                fg=ModernColors.SUCCESS, bg=ModernColors.BG_CARD).pack(side=tk.LEFT)
        self.current_label = tk.Label(current_frame, text="—",
                                       font=("Segoe UI", 10, "bold"),
                                       fg=ModernColors.TEXT_PRIMARY,
                                       bg=ModernColors.BG_CARD)
        self.current_label.pack(side=tk.LEFT, padx=(2, 0))
    
    def _draw_led(self, color: str, glow: bool = False):
        """Draw LED status indicator."""
        self.led_canvas.delete("all")
        # Glow effect for active state
        if glow:
            self.led_canvas.create_oval(1, 1, 11, 11, fill=color, outline="")
        # Main LED
        self.led_canvas.create_oval(2, 2, 10, 10, fill=color, outline=color)
        # Highlight
        self.led_canvas.create_oval(3, 3, 6, 6, fill=self._lighten(color), outline="")
    
    def _lighten(self, color: str) -> str:
        """Lighten a color for highlight effect."""
        color = color.lstrip('#')
        r, g, b = int(color[:2], 16), int(color[2:4], 16), int(color[4:], 16)
        r = min(255, r + 60)
        g = min(255, g + 60)
        b = min(255, b + 60)
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def update_display(self, load: float, queue_size: int,
                       current_process: Optional[int], utilization: float):
        """Update the processor display with animation."""
        self.load_bar.set_load(utilization)
        self.queue_label.config(text=str(queue_size))
        
        if current_process is not None:
            self.current_label.config(text=f"P{current_process}")
            self.status_label.config(text="Active", fg=ModernColors.SUCCESS)
            self._draw_led(ModernColors.SUCCESS, glow=True)
        else:
            self.current_label.config(text="—")
            self.status_label.config(text="Idle", fg=ModernColors.TEXT_MUTED)
            self._draw_led(ModernColors.TEXT_MUTED)


class ModernMetricCard(tk.Frame):
    """
    Modern metric card with icon and styled value display.
    """
    
    def __init__(self, parent, label: str, value: str = "0", 
                 icon: str = "📊", accent_color: str = None, **kwargs):
        super().__init__(parent, bg=ModernColors.BG_CARD, **kwargs)
        self.accent = accent_color or ModernColors.PRIMARY
        self.configure(highlightbackground=ModernColors.BG_CARD_HOVER,
                      highlightthickness=1)
        self._create_widgets(label, value, icon)
    
    def _create_widgets(self, label: str, value: str, icon: str):
        """Create the modern metric card widgets."""
        container = tk.Frame(self, bg=ModernColors.BG_CARD, padx=15, pady=12)
        container.pack(fill=tk.BOTH, expand=True)
        
        # Top row with icon and label
        header = tk.Frame(container, bg=ModernColors.BG_CARD)
        header.pack(fill=tk.X)
        
        tk.Label(header, text=icon, font=("Segoe UI", 14),
                bg=ModernColors.BG_CARD).pack(side=tk.LEFT)
        
        self.label = tk.Label(header, text=label, font=("Segoe UI", 10),
                              fg=ModernColors.TEXT_SECONDARY,
                              bg=ModernColors.BG_CARD)
        self.label.pack(side=tk.LEFT, padx=(8, 0))
        
        # Value with accent color
        self.value_label = tk.Label(container, text=value,
                                     font=("Segoe UI", 20, "bold"),
                                     fg=self.accent,
                                     bg=ModernColors.BG_CARD)
        self.value_label.pack(anchor=tk.W, pady=(8, 0))
    
    def set_value(self, value: str):
        """Update the displayed value."""
        self.value_label.config(text=value)


class ModernButton(tk.Button):
    """
    Modern styled button with hover effects.
    """
    
    def __init__(self, parent, text: str, command=None, 
                 style: str = "primary", icon: str = "", **kwargs):
        self.style = style
        self.colors = self._get_style_colors()
        
        super().__init__(
            parent,
            text=f"{icon} {text}".strip(),
            command=command,
            font=("Segoe UI", 10, "bold"),
            fg=self.colors['fg'],
            bg=self.colors['bg'],
            activeforeground=self.colors['fg'],
            activebackground=self.colors['hover'],
            relief=tk.FLAT,
            cursor="hand2",
            padx=16,
            pady=8,
            **kwargs
        )
        
        # Bind hover events
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
    
    def _get_style_colors(self) -> dict:
        """Get colors based on button style."""
        styles = {
            'primary': {'bg': ModernColors.PRIMARY, 'fg': '#ffffff', 
                       'hover': '#3651d4'},
            'success': {'bg': ModernColors.SUCCESS, 'fg': '#ffffff', 
                       'hover': '#05c090'},
            'danger': {'bg': ModernColors.DANGER, 'fg': '#ffffff', 
                      'hover': '#dc3d5f'},
            'warning': {'bg': ModernColors.WARNING, 'fg': '#1a1a2e', 
                       'hover': '#eec44e'},
            'secondary': {'bg': ModernColors.BG_CARD_HOVER, 
                         'fg': ModernColors.TEXT_PRIMARY, 
                         'hover': '#303050'},
        }
        return styles.get(self.style, styles['primary'])
    
    def _on_enter(self, event):
        """Handle mouse enter."""
        self.config(bg=self.colors['hover'])
    
    def _on_leave(self, event):
        """Handle mouse leave."""
        self.config(bg=self.colors['bg'])


# Legacy widget aliases for compatibility
class LoadBar(ModernLoadBar):
    """Backward compatible alias for ModernLoadBar."""
    pass


class ProcessorWidget(ModernProcessorCard):
    """Backward compatible alias for ModernProcessorCard."""
    pass


class MetricCard(ModernMetricCard):
    """Backward compatible alias for ModernMetricCard."""
    def __init__(self, parent, label: str, value: str = "0", **kwargs):
        super().__init__(parent, label, value, icon="", **kwargs)


# =============================================================================
# MAIN GUI CLASS - MODERN DESIGN
# =============================================================================

class LoadBalancerGUI:
    """
    Modern GUI application for the Dynamic Load Balancing Simulator.
    
    Features:
    - Dark theme with modern aesthetics
    - Card-based layout with elevation effects
    - Smooth animations and transitions
    - Professional color scheme
    - Responsive design
    
    Thread Safety:
    The GUI runs on the main thread while simulation runs in a background
    thread. Updates are coordinated using a queue and the after() method.
    """
    
    def __init__(self):
        """Initialize the modern GUI application."""
        # Create main window
        self.root = tk.Tk()
        self.root.title(f"⚡ {APP_NAME} v{VERSION}")
        self.root.geometry("1500x950")
        self.root.minsize(1300, 800)
        
        # Set dark theme background
        self.root.configure(bg=ModernColors.BG_DARK)
        
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
        self.processor_widgets: List[ModernProcessorCard] = []
        
        # Metric cards
        self.metric_cards: Dict[str, ModernMetricCard] = {}
        
        # Gantt chart data
        self.gantt_data: List[Dict] = []
        
        # Create GUI components
        self._configure_styles()
        self._create_menu()
        self._create_main_layout()
        
        # Bind events
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Start update loop
        self._process_updates()
    
    def _configure_styles(self):
        """Configure modern ttk styles."""
        style = ttk.Style()
        
        # Try to use a modern theme
        try:
            style.theme_use('clam')
        except:
            pass
        
        # Configure colors for dark theme
        style.configure(".", 
                       background=ModernColors.BG_DARK,
                       foreground=ModernColors.TEXT_PRIMARY,
                       fieldbackground=ModernColors.BG_INPUT)
        
        style.configure("TFrame", background=ModernColors.BG_DARK)
        style.configure("Card.TFrame", background=ModernColors.BG_CARD)
        
        style.configure("TLabel",
                       background=ModernColors.BG_DARK,
                       foreground=ModernColors.TEXT_PRIMARY,
                       font=("Segoe UI", 10))
        
        style.configure("Header.TLabel",
                       font=("Segoe UI", 16, "bold"),
                       foreground=ModernColors.TEXT_PRIMARY)
        
        style.configure("SubHeader.TLabel",
                       font=("Segoe UI", 12, "bold"),
                       foreground=ModernColors.TEXT_SECONDARY)
        
        style.configure("TLabelframe",
                       background=ModernColors.BG_CARD,
                       foreground=ModernColors.TEXT_PRIMARY)
        style.configure("TLabelframe.Label",
                       background=ModernColors.BG_CARD,
                       foreground=ModernColors.PRIMARY,
                       font=("Segoe UI", 11, "bold"))
        
        # Entry/Spinbox styles
        style.configure("TEntry",
                       fieldbackground=ModernColors.BG_INPUT,
                       foreground=ModernColors.TEXT_PRIMARY)
        
        style.configure("TSpinbox",
                       fieldbackground=ModernColors.BG_INPUT,
                       foreground=ModernColors.TEXT_PRIMARY,
                       arrowcolor=ModernColors.TEXT_PRIMARY)
        
        # Combobox style
        style.configure("TCombobox",
                       fieldbackground=ModernColors.BG_INPUT,
                       foreground=ModernColors.TEXT_PRIMARY,
                       arrowcolor=ModernColors.TEXT_PRIMARY)
        style.map("TCombobox",
                 fieldbackground=[('readonly', ModernColors.BG_INPUT)],
                 selectbackground=[('readonly', ModernColors.PRIMARY)])
        
        # Progress bar style
        style.configure("TProgressbar",
                       background=ModernColors.PRIMARY,
                       troughcolor=ModernColors.BG_INPUT,
                       bordercolor=ModernColors.BG_INPUT,
                       lightcolor=ModernColors.PRIMARY,
                       darkcolor=ModernColors.PRIMARY)
        
        # Treeview style
        style.configure("Treeview",
                       background=ModernColors.BG_CARD,
                       foreground=ModernColors.TEXT_PRIMARY,
                       fieldbackground=ModernColors.BG_CARD,
                       font=("Segoe UI", 9))
        style.configure("Treeview.Heading",
                       background=ModernColors.BG_CARD_HOVER,
                       foreground=ModernColors.TEXT_PRIMARY,
                       font=("Segoe UI", 10, "bold"))
        style.map("Treeview",
                 background=[('selected', ModernColors.PRIMARY)],
                 foreground=[('selected', '#ffffff')])
        
        # Scale style
        style.configure("TScale",
                       background=ModernColors.BG_DARK,
                       troughcolor=ModernColors.BG_INPUT)
    
    def _create_menu(self):
        """Create the modern application menu bar."""
        menubar = tk.Menu(self.root, bg=ModernColors.BG_CARD, 
                         fg=ModernColors.TEXT_PRIMARY,
                         activebackground=ModernColors.PRIMARY,
                         activeforeground='#ffffff')
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0, bg=ModernColors.BG_CARD,
                           fg=ModernColors.TEXT_PRIMARY,
                           activebackground=ModernColors.PRIMARY)
        menubar.add_cascade(label="📁 File", menu=file_menu)
        file_menu.add_command(label="📤 Export Results...", command=self._export_results)
        file_menu.add_command(label="💾 Save Configuration...", command=self._save_config)
        file_menu.add_command(label="📂 Load Configuration...", command=self._load_config)
        file_menu.add_separator()
        file_menu.add_command(label="🚪 Exit", command=self._on_close)
        
        # Simulation menu
        sim_menu = tk.Menu(menubar, tearoff=0, bg=ModernColors.BG_CARD,
                          fg=ModernColors.TEXT_PRIMARY,
                          activebackground=ModernColors.PRIMARY)
        menubar.add_cascade(label="⚙️ Simulation", menu=sim_menu)
        sim_menu.add_command(label="▶️ Start", command=self._start_simulation)
        sim_menu.add_command(label="⏸️ Pause/Resume", command=self._toggle_pause)
        sim_menu.add_command(label="⏹️ Stop", command=self._stop_simulation)
        sim_menu.add_command(label="🔄 Reset", command=self._reset_simulation)
        sim_menu.add_separator()
        sim_menu.add_command(label="📊 Compare Algorithms", command=self._compare_algorithms)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0, bg=ModernColors.BG_CARD,
                           fg=ModernColors.TEXT_PRIMARY,
                           activebackground=ModernColors.PRIMARY)
        menubar.add_cascade(label="❓ Help", menu=help_menu)
        help_menu.add_command(label="ℹ️ About", command=self._show_about)
        help_menu.add_command(label="📖 Algorithm Info", command=self._show_algorithm_info)
    
    def _create_main_layout(self):
        """Create the modern main window layout."""
        # Main container with padding
        main_frame = tk.Frame(self.root, bg=ModernColors.BG_DARK, padx=15, pady=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header with title
        self._create_header(main_frame)
        
        # Top section: Control Panel
        self._create_control_panel(main_frame)
        
        # Middle section: Visualization (Processors + Charts)
        middle_frame = tk.Frame(main_frame, bg=ModernColors.BG_DARK)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=15)
        
        # Left: Processors Panel
        self._create_processor_panel(middle_frame)
        
        # Right: Charts and Process Table
        right_frame = tk.Frame(middle_frame, bg=ModernColors.BG_DARK)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(15, 0))
        
        self._create_gantt_chart(right_frame)
        self._create_process_table(right_frame)
        
        # Bottom section: Metrics Dashboard
        self._create_metrics_panel(main_frame)
    
    def _create_header(self, parent):
        """Create the header section with title and status."""
        header_frame = tk.Frame(parent, bg=ModernColors.BG_DARK)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Title with icon
        title_frame = tk.Frame(header_frame, bg=ModernColors.BG_DARK)
        title_frame.pack(side=tk.LEFT)
        
        tk.Label(title_frame, text="⚡", font=("Segoe UI", 24),
                bg=ModernColors.BG_DARK, fg=ModernColors.PRIMARY).pack(side=tk.LEFT)
        
        tk.Label(title_frame, text=APP_NAME,
                font=("Segoe UI", 20, "bold"),
                bg=ModernColors.BG_DARK,
                fg=ModernColors.TEXT_PRIMARY).pack(side=tk.LEFT, padx=(10, 0))
        
        tk.Label(title_frame, text=f"v{VERSION}",
                font=("Segoe UI", 12),
                bg=ModernColors.BG_DARK,
                fg=ModernColors.TEXT_MUTED).pack(side=tk.LEFT, padx=(10, 0), pady=(8, 0))
    
    def _create_control_panel(self, parent):
        """Create the modern control panel with configuration options."""
        # Card container
        panel = tk.Frame(parent, bg=ModernColors.BG_CARD,
                        highlightbackground=ModernColors.BG_CARD_HOVER,
                        highlightthickness=1)
        panel.pack(fill=tk.X)
        
        inner = tk.Frame(panel, bg=ModernColors.BG_CARD, padx=20, pady=15)
        inner.pack(fill=tk.X)
        
        # Section title
        title_row = tk.Frame(inner, bg=ModernColors.BG_CARD)
        title_row.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(title_row, text="🎮 Control Panel",
                font=("Segoe UI", 14, "bold"),
                bg=ModernColors.BG_CARD,
                fg=ModernColors.PRIMARY).pack(side=tk.LEFT)
        
        # Configuration row
        config_frame = tk.Frame(inner, bg=ModernColors.BG_CARD)
        config_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Processors input
        self._create_config_input(config_frame, "🖥️ Processors", "processors",
                                  2, 16, self.config.num_processors)
        
        # Processes input
        self._create_config_input(config_frame, "📦 Processes", "processes",
                                  5, 100, self.config.num_processes)
        
        # Time Quantum input
        self._create_config_input(config_frame, "⏱️ Time Quantum", "quantum",
                                  1, 20, self.config.time_quantum)
        
        # Algorithm selection
        algo_frame = tk.Frame(config_frame, bg=ModernColors.BG_CARD)
        algo_frame.pack(side=tk.LEFT, padx=(0, 30))
        
        tk.Label(algo_frame, text="🔀 Algorithm",
                font=("Segoe UI", 10),
                bg=ModernColors.BG_CARD,
                fg=ModernColors.TEXT_SECONDARY).pack(anchor=tk.W)
        
        self.algorithm_var = tk.StringVar(value=self.config.default_algorithm.value)
        algorithm_combo = ttk.Combobox(algo_frame, textvariable=self.algorithm_var,
                                        values=[a.value for a in LoadBalancingAlgorithm],
                                        state="readonly", width=18,
                                        font=("Segoe UI", 10))
        algorithm_combo.pack(pady=(5, 0))
        
        # Speed control
        speed_frame = tk.Frame(config_frame, bg=ModernColors.BG_CARD)
        speed_frame.pack(side=tk.LEFT, padx=(0, 30))
        
        tk.Label(speed_frame, text="⚡ Speed",
                font=("Segoe UI", 10),
                bg=ModernColors.BG_CARD,
                fg=ModernColors.TEXT_SECONDARY).pack(anchor=tk.W)
        
        speed_inner = tk.Frame(speed_frame, bg=ModernColors.BG_CARD)
        speed_inner.pack(pady=(5, 0))
        
        self.speed_var = tk.IntVar(value=50)
        speed_scale = ttk.Scale(speed_inner, from_=1, to=100, orient=tk.HORIZONTAL,
                                 variable=self.speed_var, length=120)
        speed_scale.pack(side=tk.LEFT)
        
        self.speed_label = tk.Label(speed_inner, text="50%",
                                    font=("Segoe UI", 10, "bold"),
                                    bg=ModernColors.BG_CARD,
                                    fg=ModernColors.PRIMARY,
                                    width=5)
        self.speed_label.pack(side=tk.LEFT, padx=(5, 0))
        speed_scale.bind("<Motion>", self._update_speed_label)
        speed_scale.bind("<ButtonRelease-1>", self._update_speed_label)
        
        # Control buttons row
        button_frame = tk.Frame(inner, bg=ModernColors.BG_CARD)
        button_frame.pack(fill=tk.X)
        
        # Action buttons
        self.start_btn = ModernButton(button_frame, "Start", self._start_simulation,
                                       style="success", icon="▶")
        self.start_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.pause_btn = ModernButton(button_frame, "Pause", self._toggle_pause,
                                       style="warning", icon="⏸")
        self.pause_btn.pack(side=tk.LEFT, padx=(0, 10))
        self.pause_btn.config(state=tk.DISABLED)
        
        self.stop_btn = ModernButton(button_frame, "Stop", self._stop_simulation,
                                      style="danger", icon="⏹")
        self.stop_btn.pack(side=tk.LEFT, padx=(0, 10))
        self.stop_btn.config(state=tk.DISABLED)
        
        self.reset_btn = ModernButton(button_frame, "Reset", self._reset_simulation,
                                       style="secondary", icon="↺")
        self.reset_btn.pack(side=tk.LEFT, padx=(0, 20))
        
        # Separator
        sep = tk.Frame(button_frame, bg=ModernColors.TEXT_MUTED, width=2, height=35)
        sep.pack(side=tk.LEFT, padx=(0, 20))
        
        self.compare_btn = ModernButton(button_frame, "Compare Algorithms",
                                         self._compare_algorithms,
                                         style="primary", icon="📊")
        self.compare_btn.pack(side=tk.LEFT)
        
        # Status section
        status_frame = tk.Frame(inner, bg=ModernColors.BG_CARD)
        status_frame.pack(fill=tk.X, pady=(15, 0))
        
        # Status indicator
        status_left = tk.Frame(status_frame, bg=ModernColors.BG_CARD)
        status_left.pack(side=tk.LEFT)
        
        self.status_led = tk.Canvas(status_left, width=12, height=12,
                                     bg=ModernColors.BG_CARD, highlightthickness=0)
        self.status_led.pack(side=tk.LEFT)
        self._draw_status_led(ModernColors.TEXT_MUTED)
        
        self.status_label = tk.Label(status_left, text="Ready",
                                      font=("Segoe UI", 11, "bold"),
                                      bg=ModernColors.BG_CARD,
                                      fg=ModernColors.TEXT_SECONDARY)
        self.status_label.pack(side=tk.LEFT, padx=(8, 0))
        
        # Time and progress
        status_right = tk.Frame(status_frame, bg=ModernColors.BG_CARD)
        status_right.pack(side=tk.RIGHT)
        
        self.time_label = tk.Label(status_right, text="⏱️ Time: 0",
                                    font=("Segoe UI", 11),
                                    bg=ModernColors.BG_CARD,
                                    fg=ModernColors.TEXT_PRIMARY)
        self.time_label.pack(side=tk.RIGHT, padx=(20, 0))
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(status_right, variable=self.progress_var,
                                             maximum=100, length=250, mode='determinate')
        self.progress_bar.pack(side=tk.RIGHT)
        
        tk.Label(status_right, text="Progress:",
                font=("Segoe UI", 10),
                bg=ModernColors.BG_CARD,
                fg=ModernColors.TEXT_SECONDARY).pack(side=tk.RIGHT, padx=(0, 10))
    
    def _create_config_input(self, parent, label: str, var_name: str,
                            min_val: int, max_val: int, default: int):
        """Create a modern config input field."""
        frame = tk.Frame(parent, bg=ModernColors.BG_CARD)
        frame.pack(side=tk.LEFT, padx=(0, 30))
        
        tk.Label(frame, text=label,
                font=("Segoe UI", 10),
                bg=ModernColors.BG_CARD,
                fg=ModernColors.TEXT_SECONDARY).pack(anchor=tk.W)
        
        var = tk.IntVar(value=default)
        setattr(self, f"{var_name}_var", var)
        
        spinbox = ttk.Spinbox(frame, from_=min_val, to=max_val, width=8,
                              textvariable=var, font=("Segoe UI", 10))
        spinbox.pack(pady=(5, 0))
    
    def _draw_status_led(self, color: str):
        """Draw status LED indicator."""
        self.status_led.delete("all")
        self.status_led.create_oval(2, 2, 10, 10, fill=color, outline=color)
    
    def _create_processor_panel(self, parent):
        """Create the modern processor visualization panel."""
        panel = tk.Frame(parent, bg=ModernColors.BG_CARD,
                        highlightbackground=ModernColors.BG_CARD_HOVER,
                        highlightthickness=1)
        panel.pack(side=tk.LEFT, fill=tk.Y)
        
        inner = tk.Frame(panel, bg=ModernColors.BG_CARD, padx=15, pady=15)
        inner.pack(fill=tk.BOTH, expand=True)
        
        # Header
        tk.Label(inner, text="🖥️ Processors",
                font=("Segoe UI", 14, "bold"),
                bg=ModernColors.BG_CARD,
                fg=ModernColors.PRIMARY).pack(anchor=tk.W, pady=(0, 15))
        
        # Scrollable container for processors
        canvas = tk.Canvas(inner, bg=ModernColors.BG_CARD, highlightthickness=0,
                          width=240)
        scrollbar = ttk.Scrollbar(inner, orient="vertical", command=canvas.yview)
        
        self.processor_container = tk.Frame(canvas, bg=ModernColors.BG_CARD)
        
        canvas.create_window((0, 0), window=self.processor_container, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        def on_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        self.processor_container.bind("<Configure>", on_configure)
        
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
            widget = ModernProcessorCard(self.processor_container, processor_id=i)
            widget.pack(fill=tk.X, pady=(0, 10))
            self.processor_widgets.append(widget)
    
    def _create_gantt_chart(self, parent):
        """Create the modern Gantt chart visualization."""
        panel = tk.Frame(parent, bg=ModernColors.BG_CARD,
                        highlightbackground=ModernColors.BG_CARD_HOVER,
                        highlightthickness=1)
        panel.pack(fill=tk.BOTH, expand=True)
        
        inner = tk.Frame(panel, bg=ModernColors.BG_CARD, padx=15, pady=15)
        inner.pack(fill=tk.BOTH, expand=True)
        
        # Header
        tk.Label(inner, text="📊 Process Execution Timeline",
                font=("Segoe UI", 14, "bold"),
                bg=ModernColors.BG_CARD,
                fg=ModernColors.PRIMARY).pack(anchor=tk.W, pady=(0, 10))
        
        # Create matplotlib figure with dark theme
        self.gantt_fig = Figure(figsize=(9, 3), dpi=100, facecolor=ModernColors.BG_CARD)
        self.gantt_ax = self.gantt_fig.add_subplot(111)
        self._style_chart_axes(self.gantt_ax)
        self.gantt_fig.tight_layout()
        
        # Embed in Tkinter
        self.gantt_canvas = FigureCanvasTkAgg(self.gantt_fig, master=inner)
        self.gantt_canvas.draw()
        self.gantt_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _style_chart_axes(self, ax):
        """Apply modern dark styling to chart axes."""
        ax.set_facecolor(ModernColors.BG_CARD)
        ax.tick_params(colors=ModernColors.TEXT_SECONDARY, labelsize=9)
        ax.spines['bottom'].set_color(ModernColors.TEXT_MUTED)
        ax.spines['top'].set_color(ModernColors.BG_CARD)
        ax.spines['left'].set_color(ModernColors.TEXT_MUTED)
        ax.spines['right'].set_color(ModernColors.BG_CARD)
        ax.xaxis.label.set_color(ModernColors.TEXT_SECONDARY)
        ax.yaxis.label.set_color(ModernColors.TEXT_SECONDARY)
        ax.set_xlabel("Time", fontsize=10, fontfamily='Segoe UI')
        ax.set_ylabel("Processor", fontsize=10, fontfamily='Segoe UI')
    
    def _create_process_table(self, parent):
        """Create the modern process table display."""
        panel = tk.Frame(parent, bg=ModernColors.BG_CARD,
                        highlightbackground=ModernColors.BG_CARD_HOVER,
                        highlightthickness=1)
        panel.pack(fill=tk.BOTH, expand=True, pady=(15, 0))
        
        inner = tk.Frame(panel, bg=ModernColors.BG_CARD, padx=15, pady=15)
        inner.pack(fill=tk.BOTH, expand=True)
        
        # Header
        tk.Label(inner, text="📋 Process Details",
                font=("Segoe UI", 14, "bold"),
                bg=ModernColors.BG_CARD,
                fg=ModernColors.PRIMARY).pack(anchor=tk.W, pady=(0, 10))
        
        # Create treeview with scrollbar
        tree_frame = tk.Frame(inner, bg=ModernColors.BG_CARD)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = ("PID", "Arrival", "Burst", "Remaining", "Priority",
                   "State", "Processor", "Wait", "Turnaround")
        
        self.process_tree = ttk.Treeview(tree_frame, columns=columns,
                                          show="headings", height=8)
        
        # Configure columns
        col_widths = {"PID": 55, "Arrival": 65, "Burst": 60, "Remaining": 80,
                      "Priority": 75, "State": 90, "Processor": 75,
                      "Wait": 55, "Turnaround": 90}
        
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
        """Create the modern metrics dashboard."""
        panel = tk.Frame(parent, bg=ModernColors.BG_CARD,
                        highlightbackground=ModernColors.BG_CARD_HOVER,
                        highlightthickness=1)
        panel.pack(fill=tk.X)
        
        inner = tk.Frame(panel, bg=ModernColors.BG_CARD, padx=20, pady=15)
        inner.pack(fill=tk.X)
        
        # Header row
        header_frame = tk.Frame(inner, bg=ModernColors.BG_CARD)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        tk.Label(header_frame, text="📈 Performance Metrics",
                font=("Segoe UI", 14, "bold"),
                bg=ModernColors.BG_CARD,
                fg=ModernColors.PRIMARY).pack(side=tk.LEFT)
        
        # Current algorithm label
        algo_frame = tk.Frame(header_frame, bg=ModernColors.BG_CARD)
        algo_frame.pack(side=tk.RIGHT)
        
        tk.Label(algo_frame, text="Algorithm:",
                font=("Segoe UI", 10),
                bg=ModernColors.BG_CARD,
                fg=ModernColors.TEXT_SECONDARY).pack(side=tk.LEFT)
        
        self.current_algo_label = tk.Label(algo_frame, text="—",
                                            font=("Segoe UI", 11, "bold"),
                                            bg=ModernColors.BG_CARD,
                                            fg=ModernColors.PRIMARY)
        self.current_algo_label.pack(side=tk.LEFT, padx=(8, 0))
        
        # Metrics cards grid
        metrics_frame = tk.Frame(inner, bg=ModernColors.BG_CARD)
        metrics_frame.pack(fill=tk.X)
        
        # Row 1: Process metrics
        metrics_row1 = [
            ("Completed", "completed", "0/0", "✅", ModernColors.SUCCESS),
            ("Avg Turnaround", "turnaround", "0.00", "⏱️", ModernColors.INFO),
            ("Avg Waiting", "waiting", "0.00", "⏳", ModernColors.WARNING),
            ("Avg Response", "response", "0.00", "⚡", ModernColors.PRIMARY),
        ]
        
        row1_frame = tk.Frame(metrics_frame, bg=ModernColors.BG_CARD)
        row1_frame.pack(fill=tk.X, pady=(0, 10))
        
        for label, key, default, icon, color in metrics_row1:
            card = ModernMetricCard(row1_frame, label, default, icon, color)
            card.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)
            self.metric_cards[key] = card
        
        # Row 2: System metrics
        metrics_row2 = [
            ("Avg Utilization", "utilization", "0.0%", "📊", ModernColors.SUCCESS),
            ("Load Balance", "lbi", "0.0000", "⚖️", ModernColors.INFO),
            ("Jain's Fairness", "fairness", "0.0000", "🎯", ModernColors.SECONDARY),
            ("Migrations", "migrations", "0", "🔄", ModernColors.ACCENT),
        ]
        
        row2_frame = tk.Frame(metrics_frame, bg=ModernColors.BG_CARD)
        row2_frame.pack(fill=tk.X)
        
        for label, key, default, icon, color in metrics_row2:
            card = ModernMetricCard(row2_frame, label, default, icon, color)
            card.pack(side=tk.LEFT, padx=(0, 10), fill=tk.X, expand=True)
            self.metric_cards[key] = card
    
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
        self.status_label.config(text="Running", fg=ModernColors.SUCCESS)
        self._draw_status_led(ModernColors.SUCCESS)
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
            self.pause_btn.config(text="▶ Resume", bg=ModernColors.SUCCESS)
            self.status_label.config(text="Paused", fg=ModernColors.WARNING)
            self._draw_status_led(ModernColors.WARNING)
        else:
            self.pause_btn.config(text="⏸ Pause", bg=ModernColors.WARNING)
            self.status_label.config(text="Running", fg=ModernColors.SUCCESS)
            self._draw_status_led(ModernColors.SUCCESS)
    
    def _stop_simulation(self):
        """Stop the simulation."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.engine:
            self.engine.stop()
        
        self._update_button_states()
        self.status_label.config(text="Stopped", fg=ModernColors.DANGER)
        self._draw_status_led(ModernColors.DANGER)
    
    def _reset_simulation(self):
        """Reset the simulation."""
        self._stop_simulation()
        
        # Clear visualizations
        self._clear_gantt_chart()
        self._clear_process_table()
        self._reset_metrics()
        self._reset_processor_displays()
        
        self.status_label.config(text="Ready", fg=ModernColors.TEXT_SECONDARY)
        self._draw_status_led(ModernColors.TEXT_MUTED)
        self.time_label.config(text="⏱️ Time: 0")
        self.progress_var.set(0)
        self.current_algo_label.config(text="—")
    
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
        self.time_label.config(text=f"⏱️ Time: {state['time']}")
        
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
        self.status_label.config(text="Completed", fg=ModernColors.SUCCESS)
        self._draw_status_led(ModernColors.PRIMARY)
        
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
        """Update the Gantt chart visualization with modern styling."""
        self.gantt_ax.clear()
        self._style_chart_axes(self.gantt_ax)
        
        if not self.gantt_data:
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
        
        # Draw rectangles with modern styling
        num_processors = self.processors_var.get()
        process_colors = {}
        color_index = 0
        
        for entry in consolidated:
            pid = entry['process']
            if pid not in process_colors:
                process_colors[pid] = ModernColors.PROCESSOR_COLORS[color_index % len(ModernColors.PROCESSOR_COLORS)]
                color_index += 1
            
            # Create rounded rectangle effect
            rect = FancyBboxPatch(
                (entry['start'], entry['processor'] - 0.35),
                entry['end'] - entry['start'],
                0.7,
                boxstyle="round,pad=0.02,rounding_size=0.15",
                facecolor=process_colors[pid],
                edgecolor='white',
                linewidth=0.5,
                alpha=0.9
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
                    fontsize=8, fontweight='bold',
                    color='white'
                )
        
        # Configure axes with modern styling
        max_time = max(e['end'] for e in consolidated) if consolidated else 10
        self.gantt_ax.set_xlim(-0.5, max_time + 1)
        self.gantt_ax.set_ylim(-0.5, num_processors - 0.5)
        self.gantt_ax.set_yticks(range(num_processors))
        self.gantt_ax.set_yticklabels([f"CPU {i}" for i in range(num_processors)])
        self.gantt_ax.grid(True, axis='x', alpha=0.2, color=ModernColors.TEXT_MUTED)
        
        self.gantt_fig.tight_layout()
        self.gantt_canvas.draw()
    
    def _clear_gantt_chart(self):
        """Clear the Gantt chart."""
        self.gantt_data.clear()
        self.gantt_ax.clear()
        self._style_chart_axes(self.gantt_ax)
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
        
        # Show progress dialog with modern styling
        progress_window = tk.Toplevel(self.root)
        progress_window.title("⏳ Comparing Algorithms")
        progress_window.geometry("450x180")
        progress_window.configure(bg=ModernColors.BG_DARK)
        progress_window.transient(self.root)
        progress_window.grab_set()
        
        # Center the window
        progress_window.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - 450) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 180) // 2
        progress_window.geometry(f"+{x}+{y}")
        
        container = tk.Frame(progress_window, bg=ModernColors.BG_CARD, padx=30, pady=25)
        container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        tk.Label(container, text="🔄 Running simulations...",
                font=("Segoe UI", 14, "bold"),
                bg=ModernColors.BG_CARD,
                fg=ModernColors.TEXT_PRIMARY).pack(pady=(0, 15))
        
        progress_bar = ttk.Progressbar(container, length=350, mode='determinate')
        progress_bar.pack(pady=10)
        
        status_label = tk.Label(container, text="",
                               font=("Segoe UI", 10),
                               bg=ModernColors.BG_CARD,
                               fg=ModernColors.TEXT_SECONDARY)
        status_label.pack(pady=10)
        
        self.root.update()
        
        # Run batch comparison
        batch = BatchSimulator(self.config)
        algorithms = list(LoadBalancingAlgorithm)
        results = {}
        
        for i, algo in enumerate(algorithms):
            status_label.config(text=f"Testing {algo.value}...")
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
        status_label.config(text="✅ Complete!")
        self.root.update()
        time.sleep(0.3)
        progress_window.destroy()
        
        # Show comparison results
        self._show_comparison_results(results, batch)
    
    def _show_comparison_results(self, results: Dict, batch: BatchSimulator):
        """Display comparison results in a modern styled window."""
        window = tk.Toplevel(self.root)
        window.title("📊 Algorithm Comparison Results")
        window.geometry("1100x750")
        window.configure(bg=ModernColors.BG_DARK)
        
        # Create modern notebook
        style = ttk.Style()
        style.configure("Modern.TNotebook", background=ModernColors.BG_DARK)
        style.configure("Modern.TNotebook.Tab", 
                       background=ModernColors.BG_CARD,
                       foreground=ModernColors.TEXT_PRIMARY,
                       padding=[20, 10],
                       font=("Segoe UI", 10, "bold"))
        style.map("Modern.TNotebook.Tab",
                 background=[("selected", ModernColors.PRIMARY)],
                 foreground=[("selected", "#ffffff")])
        
        notebook = ttk.Notebook(window, style="Modern.TNotebook")
        notebook.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Tab 1: Summary Table
        table_frame = tk.Frame(notebook, bg=ModernColors.BG_CARD)
        notebook.add(table_frame, text="📋 Summary")
        
        inner = tk.Frame(table_frame, bg=ModernColors.BG_CARD, padx=20, pady=20)
        inner.pack(fill=tk.BOTH, expand=True)
        
        # Header
        tk.Label(inner, text="🏆 Algorithm Performance Comparison",
                font=("Segoe UI", 16, "bold"),
                bg=ModernColors.BG_CARD,
                fg=ModernColors.PRIMARY).pack(anchor=tk.W, pady=(0, 15))
        
        columns = ("Algorithm", "Time", "Avg Turnaround", "Avg Waiting",
                   "Utilization", "LBI", "Fairness", "Migrations")
        
        tree = ttk.Treeview(inner, columns=columns, show="headings", height=5)
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor=tk.CENTER)
        
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
        
        # Best algorithm summary with cards
        best_frame = tk.Frame(inner, bg=ModernColors.BG_CARD)
        best_frame.pack(fill=tk.X, pady=15)
        
        tk.Label(best_frame, text="🎯 Best Performers",
                font=("Segoe UI", 14, "bold"),
                bg=ModernColors.BG_CARD,
                fg=ModernColors.TEXT_PRIMARY).pack(anchor=tk.W, pady=(0, 10))
        
        best_turnaround = batch.get_best_algorithm('avg_turnaround_time')
        best_waiting = batch.get_best_algorithm('avg_waiting_time')
        best_fairness = batch.get_best_algorithm('jains_fairness_index')
        
        cards_frame = tk.Frame(best_frame, bg=ModernColors.BG_CARD)
        cards_frame.pack(fill=tk.X)
        
        for label, value, color in [
            ("⏱️ Lowest Turnaround", best_turnaround, ModernColors.SUCCESS),
            ("⏳ Lowest Waiting", best_waiting, ModernColors.INFO),
            ("🎯 Best Fairness", best_fairness, ModernColors.ACCENT)
        ]:
            card = tk.Frame(cards_frame, bg=ModernColors.BG_INPUT, padx=15, pady=10)
            card.pack(side=tk.LEFT, padx=(0, 10))
            tk.Label(card, text=label, font=("Segoe UI", 9),
                    bg=ModernColors.BG_INPUT, fg=ModernColors.TEXT_SECONDARY).pack(anchor=tk.W)
            tk.Label(card, text=str(value), font=("Segoe UI", 12, "bold"),
                    bg=ModernColors.BG_INPUT, fg=color).pack(anchor=tk.W)
        
        # Tab 2: Comparison Charts
        chart_frame = tk.Frame(notebook, bg=ModernColors.BG_CARD)
        notebook.add(chart_frame, text="📊 Charts")
        
        fig = Figure(figsize=(11, 6.5), dpi=100, facecolor=ModernColors.BG_CARD)
        
        # Apply dark theme to all subplots
        algo_names = [a.value for a in results.keys()]
        bar_colors = [ModernColors.PRIMARY, ModernColors.SUCCESS, ModernColors.ACCENT]
        
        # Turnaround Time comparison
        ax1 = fig.add_subplot(221)
        ax1.set_facecolor(ModernColors.BG_CARD)
        turnaround_values = [r.system_metrics.avg_turnaround_time for r in results.values()]
        bars = ax1.bar(algo_names, turnaround_values, color=bar_colors)
        ax1.set_ylabel("Time", color=ModernColors.TEXT_SECONDARY)
        ax1.set_title("Average Turnaround Time", color=ModernColors.TEXT_PRIMARY, fontweight='bold')
        ax1.tick_params(axis='x', rotation=15, colors=ModernColors.TEXT_SECONDARY)
        ax1.tick_params(axis='y', colors=ModernColors.TEXT_SECONDARY)
        ax1.spines['bottom'].set_color(ModernColors.TEXT_MUTED)
        ax1.spines['left'].set_color(ModernColors.TEXT_MUTED)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Utilization comparison
        ax2 = fig.add_subplot(222)
        ax2.set_facecolor(ModernColors.BG_CARD)
        util_values = [r.system_metrics.avg_utilization * 100 for r in results.values()]
        bars = ax2.bar(algo_names, util_values, color=bar_colors)
        ax2.set_ylabel("Percentage", color=ModernColors.TEXT_SECONDARY)
        ax2.set_title("Average CPU Utilization", color=ModernColors.TEXT_PRIMARY, fontweight='bold')
        ax2.tick_params(axis='x', rotation=15, colors=ModernColors.TEXT_SECONDARY)
        ax2.tick_params(axis='y', colors=ModernColors.TEXT_SECONDARY)
        ax2.spines['bottom'].set_color(ModernColors.TEXT_MUTED)
        ax2.spines['left'].set_color(ModernColors.TEXT_MUTED)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        # Fairness Index comparison
        ax3 = fig.add_subplot(223)
        ax3.set_facecolor(ModernColors.BG_CARD)
        fairness_values = [r.system_metrics.jains_fairness_index for r in results.values()]
        bars = ax3.bar(algo_names, fairness_values, color=bar_colors)
        ax3.set_ylabel("Index (0-1)", color=ModernColors.TEXT_SECONDARY)
        ax3.set_title("Jain's Fairness Index", color=ModernColors.TEXT_PRIMARY, fontweight='bold')
        ax3.tick_params(axis='x', rotation=15, colors=ModernColors.TEXT_SECONDARY)
        ax3.tick_params(axis='y', colors=ModernColors.TEXT_SECONDARY)
        ax3.spines['bottom'].set_color(ModernColors.TEXT_MUTED)
        ax3.spines['left'].set_color(ModernColors.TEXT_MUTED)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        
        # Migrations comparison
        ax4 = fig.add_subplot(224)
        ax4.set_facecolor(ModernColors.BG_CARD)
        migration_values = [r.system_metrics.total_migrations for r in results.values()]
        bars = ax4.bar(algo_names, migration_values, color=bar_colors)
        ax4.set_ylabel("Count", color=ModernColors.TEXT_SECONDARY)
        ax4.set_title("Process Migrations", color=ModernColors.TEXT_PRIMARY, fontweight='bold')
        ax4.tick_params(axis='x', rotation=15, colors=ModernColors.TEXT_SECONDARY)
        ax4.tick_params(axis='y', colors=ModernColors.TEXT_SECONDARY)
        ax4.spines['bottom'].set_color(ModernColors.TEXT_MUTED)
        ax4.spines['left'].set_color(ModernColors.TEXT_MUTED)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        
        fig.tight_layout(pad=3.0)
        
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Export button frame
        export_frame = tk.Frame(window, bg=ModernColors.BG_DARK)
        export_frame.pack(fill=tk.X, padx=15, pady=(0, 15))
        
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
        
        export_btn = ModernButton(export_frame, "Export Results", export_comparison,
                                   style="primary", icon="📤")
        export_btn.pack(side=tk.RIGHT)
    
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
        """Show modern about dialog."""
        about_window = tk.Toplevel(self.root)
        about_window.title("About")
        about_window.geometry("500x550")
        about_window.configure(bg=ModernColors.BG_DARK)
        about_window.resizable(False, False)
        
        # Center the window
        about_window.update_idletasks()
        x = self.root.winfo_x() + (self.root.winfo_width() - 500) // 2
        y = self.root.winfo_y() + (self.root.winfo_height() - 550) // 2
        about_window.geometry(f"+{x}+{y}")
        
        container = tk.Frame(about_window, bg=ModernColors.BG_CARD, padx=40, pady=30)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Logo/Icon
        tk.Label(container, text="⚡", font=("Segoe UI", 48),
                bg=ModernColors.BG_CARD, fg=ModernColors.PRIMARY).pack(pady=(0, 10))
        
        # Title
        tk.Label(container, text=APP_NAME,
                font=("Segoe UI", 18, "bold"),
                bg=ModernColors.BG_CARD,
                fg=ModernColors.TEXT_PRIMARY).pack()
        
        # Version
        tk.Label(container, text=f"Version {VERSION}",
                font=("Segoe UI", 11),
                bg=ModernColors.BG_CARD,
                fg=ModernColors.TEXT_MUTED).pack(pady=(5, 20))
        
        # Description
        desc = """An educational simulation demonstrating dynamic
load balancing algorithms in multiprocessor systems."""
        tk.Label(container, text=desc,
                font=("Segoe UI", 10),
                bg=ModernColors.BG_CARD,
                fg=ModernColors.TEXT_SECONDARY,
                justify=tk.CENTER).pack(pady=(0, 20))
        
        # Features
        features_frame = tk.Frame(container, bg=ModernColors.BG_INPUT, padx=20, pady=15)
        features_frame.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(features_frame, text="✨ Key Features",
                font=("Segoe UI", 11, "bold"),
                bg=ModernColors.BG_INPUT,
                fg=ModernColors.PRIMARY).pack(anchor=tk.W, pady=(0, 10))
        
        features = [
            "🔀 Multiple load balancing algorithms",
            "📊 Real-time visualization",
            "📈 Gantt chart process timeline",
            "🎯 Performance metrics & comparison"
        ]
        
        for f in features:
            tk.Label(features_frame, text=f,
                    font=("Segoe UI", 10),
                    bg=ModernColors.BG_INPUT,
                    fg=ModernColors.TEXT_PRIMARY).pack(anchor=tk.W, pady=2)
        
        # OS Concepts
        concepts_frame = tk.Frame(container, bg=ModernColors.BG_INPUT, padx=20, pady=15)
        concepts_frame.pack(fill=tk.X)
        
        tk.Label(concepts_frame, text="📚 OS Concepts Demonstrated",
                font=("Segoe UI", 11, "bold"),
                bg=ModernColors.BG_INPUT,
                fg=ModernColors.SECONDARY).pack(anchor=tk.W, pady=(0, 10))
        
        concepts = ["Process Management", "CPU Scheduling", 
                   "Load Balancing", "Resource Utilization"]
        
        for c in concepts:
            tk.Label(concepts_frame, text=f"• {c}",
                    font=("Segoe UI", 10),
                    bg=ModernColors.BG_INPUT,
                    fg=ModernColors.TEXT_PRIMARY).pack(anchor=tk.W, pady=1)
        
        # Footer
        tk.Label(container, text="Made with ❤️ for learning OS concepts",
                font=("Segoe UI", 9),
                bg=ModernColors.BG_CARD,
                fg=ModernColors.TEXT_MUTED).pack(pady=(20, 0))
    
    def _show_algorithm_info(self):
        """Show modern information about load balancing algorithms."""
        info_window = tk.Toplevel(self.root)
        info_window.title("📖 Load Balancing Algorithms")
        info_window.geometry("700x600")
        info_window.configure(bg=ModernColors.BG_DARK)
        
        container = tk.Frame(info_window, bg=ModernColors.BG_CARD, padx=25, pady=20)
        container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Header
        tk.Label(container, text="📖 Load Balancing Algorithms",
                font=("Segoe UI", 16, "bold"),
                bg=ModernColors.BG_CARD,
                fg=ModernColors.PRIMARY).pack(anchor=tk.W, pady=(0, 15))
        
        # Scrollable text area
        text_frame = tk.Frame(container, bg=ModernColors.BG_CARD)
        text_frame.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text = tk.Text(text_frame, wrap=tk.WORD, 
                      bg=ModernColors.BG_INPUT,
                      fg=ModernColors.TEXT_PRIMARY,
                      font=("Consolas", 10),
                      padx=15, pady=15,
                      relief=tk.FLAT,
                      yscrollcommand=scrollbar.set)
        text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=text.yview)
        
        info_text = """
╔══════════════════════════════════════════════════════════════════╗
║                    LOAD BALANCING ALGORITHMS                      ║
╚══════════════════════════════════════════════════════════════════╝

┌──────────────────────────────────────────────────────────────────┐
│  1. ROUND ROBIN                                                  │
└──────────────────────────────────────────────────────────────────┘

  Distribution: Assigns processes to processors in cyclic order
                (P0 → P1 → P2 → P3 → P0 → ...)

  ✅ Advantages:
     • Simple and fair
     • Equal distribution by count
     • Low computational overhead
     • Deterministic behavior

  ❌ Disadvantages:
     • Ignores actual processor load
     • Can cause imbalance with varied process sizes
     • No dynamic adaptation

  🎯 Best for: Homogeneous workloads with similar process sizes


┌──────────────────────────────────────────────────────────────────┐
│  2. LEAST LOADED FIRST                                           │
└──────────────────────────────────────────────────────────────────┘

  Distribution: Assigns each new process to the processor with
                minimum current load.

  ✅ Advantages:
     • Better load distribution
     • Adapts to current system state
     • Efficient for varied workloads
     • Considers actual work remaining

  ❌ Disadvantages:
     • Slightly higher overhead (requires load monitoring)
     • No process migration after initial assignment
     • May cause "herd behavior"

  🎯 Best for: Variable workloads with different burst times


┌──────────────────────────────────────────────────────────────────┐
│  3. THRESHOLD-BASED                                              │
└──────────────────────────────────────────────────────────────────┘

  Distribution: Uses least loaded for initial assignment, then
                migrates processes when load difference exceeds
                a threshold.

  ✅ Advantages:
     • Dynamic rebalancing
     • Handles changing workloads
     • Prevents severe imbalances
     • Combines best of other approaches

  ❌ Disadvantages:
     • Migration overhead (context switch cost)
     • Requires careful threshold tuning
     • More complex implementation
     • May cause oscillation if thresholds wrong

  🎯 Best for: Dynamic workloads where load changes over time


╔══════════════════════════════════════════════════════════════════╗
║                      METRICS EXPLAINED                            ║
╚══════════════════════════════════════════════════════════════════╝

  📊 Turnaround Time:
     Total time from process arrival to completion

  ⏳ Waiting Time:
     Time spent waiting in ready queue

  📈 CPU Utilization:
     Percentage of time processor is busy

  ⚖️ Load Balance Index:
     Measure of load distribution (1.0 = perfect balance)

  🎯 Jain's Fairness Index:
     Statistical fairness measure (1.0 = perfectly fair)
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
