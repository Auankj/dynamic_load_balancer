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
import sys
import ctypes
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
import matplotlib.patheffects

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
    STATE_READY = "#3559f8"
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


# Platform font selection
def _get_default_font_family():
    """Return a sensible default font family depending on the OS."""
    plat = sys.platform
    if plat.startswith("win"):
        return "Segoe UI"
    if plat == "darwin":
        # macOS: San Francisco is the system font, but not directly named; fall back to Helvetica
        return "Helvetica"
    # Linux and others
    return "DejaVu Sans"

DEFAULT_FONT_FAMILY = _get_default_font_family()


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
                        text=text, fill="#000000", font=(DEFAULT_FONT_FAMILY, 11, "bold"))
        # Main text
        self.create_text(self.bar_width // 2 + 4, self.bar_height // 2 + 4,
                        text=text, fill=ModernColors.TEXT_PRIMARY, 
                        font=(DEFAULT_FONT_FAMILY, 11, "bold"))
    
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
                              font=(DEFAULT_FONT_FAMILY, 12, "bold"), fg=color,
                              bg=ModernColors.BG_CARD)
        self.badge.pack(side=tk.LEFT)
        
        # Status LED indicator
        self.led_canvas = tk.Canvas(header_frame, width=12, height=12,
                                     bg=ModernColors.BG_CARD, highlightthickness=0)
        self.led_canvas.pack(side=tk.RIGHT, padx=(0, 5))
        self._draw_led(ModernColors.TEXT_MUTED)
        
        # Status text
        self.status_label = tk.Label(header_frame, text="Idle",
                                      font=(DEFAULT_FONT_FAMILY, 10),
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
        
        tk.Label(queue_frame, text="📋", font=(DEFAULT_FONT_FAMILY, 10),
                bg=ModernColors.BG_CARD).pack(side=tk.LEFT)
        self.queue_label = tk.Label(queue_frame, text="0",
                                     font=(DEFAULT_FONT_FAMILY, 10, "bold"),
                                     fg=ModernColors.TEXT_PRIMARY,
                                     bg=ModernColors.BG_CARD)
        self.queue_label.pack(side=tk.LEFT, padx=(2, 0))
        
        # Current process indicator
        current_frame = tk.Frame(stats_frame, bg=ModernColors.BG_CARD)
        current_frame.pack(side=tk.RIGHT)
        
        tk.Label(current_frame, text="▶", font=(DEFAULT_FONT_FAMILY, 10),
                fg=ModernColors.SUCCESS, bg=ModernColors.BG_CARD).pack(side=tk.LEFT)
        self.current_label = tk.Label(current_frame, text="—",
                                      font=(DEFAULT_FONT_FAMILY, 10, "bold"),
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
        
        tk.Label(header, text=icon, font=(DEFAULT_FONT_FAMILY, 14),
                bg=ModernColors.BG_CARD).pack(side=tk.LEFT)
        
        self.label = tk.Label(header, text=label, font=(DEFAULT_FONT_FAMILY, 10),
                              fg=ModernColors.TEXT_SECONDARY,
                              bg=ModernColors.BG_CARD)
        self.label.pack(side=tk.LEFT, padx=(8, 0))
        
        # Value with accent color
        self.value_label = tk.Label(container, text=value,
                                     font=(DEFAULT_FONT_FAMILY, 20, "bold"),
                                     fg=self.accent,
                                     bg=ModernColors.BG_CARD)
        self.value_label.pack(anchor=tk.W, pady=(8, 0))
    
    def set_value(self, value: str):
        """Update the displayed value."""
        self.value_label.config(text=value)


class ModernButton(tk.Frame):
    """
    Modern styled button with border, hover effects, and better visibility.
    Uses a Frame-based approach for better cross-platform rendering.
    """
    
    def __init__(self, parent, text: str, command=None, 
                 style: str = "primary", icon: str = "", **kwargs):
        self.button_style = style
        self.command = command
        self.colors = self._get_style_colors()
        self._disabled = False
        
        # Create outer frame for border effect
        super().__init__(parent, bg=self.colors['border'], 
                        highlightthickness=0, **kwargs)
        
        # Inner frame for background
        self.inner = tk.Frame(self, bg=self.colors['bg'], padx=2, pady=2)
        self.inner.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Label as the clickable content
        self.label = tk.Label(
            self.inner,
            text=f"{icon}  {text}".strip() if icon else text,
            font=("Helvetica Neue", 11, "bold"),
            fg=self.colors['fg'],
            bg=self.colors['bg'],
            cursor="hand2",
            padx=14,
            pady=6
        )
        self.label.pack(fill=tk.BOTH, expand=True)
        
        # Bind click and hover events to all parts
        for widget in [self, self.inner, self.label]:
            widget.bind("<Button-1>", self._on_click)
            widget.bind("<Enter>", self._on_enter)
            widget.bind("<Leave>", self._on_leave)
    
    def _get_style_colors(self) -> dict:
        """Get colors based on button style with high contrast."""
        styles = {
            'primary': {
                'bg': '#3b82f6',      # Bright blue
                'fg': '#ffffff',
                'hover': '#2563eb',   # Darker blue
                'border': '#1d4ed8',  # Blue border
                'disabled_bg': '#4a5568',
                'disabled_fg': '#9ca3af'
            },
            'success': {
                'bg': '#10b981',      # Emerald green
                'fg': '#ffffff',
                'hover': '#059669',   # Darker green
                'border': '#047857',  # Green border
                'disabled_bg': '#4a5568',
                'disabled_fg': '#9ca3af'
            },
            'danger': {
                'bg': '#ef4444',      # Bright red
                'fg': '#ffffff',
                'hover': '#dc2626',   # Darker red
                'border': '#b91c1c',  # Red border
                'disabled_bg': '#4a5568',
                'disabled_fg': '#9ca3af'
            },
            'warning': {
                'bg': '#f59e0b',      # Amber
                'fg': '#1a1a2e',      # Dark text for contrast
                'hover': '#d97706',   # Darker amber
                'border': '#b45309',  # Amber border
                'disabled_bg': '#4a5568',
                'disabled_fg': '#9ca3af'
            },
            'secondary': {
                'bg': '#4b5563',      # Gray
                'fg': '#ffffff',
                'hover': '#374151',   # Darker gray
                'border': '#1f2937',  # Dark border
                'disabled_bg': '#4a5568',
                'disabled_fg': '#9ca3af'
            },
        }
        return styles.get(self.button_style, styles['primary'])
    
    def _on_click(self, event):
        """Handle button click."""
        if not self._disabled and self.command:
            self.command()
    
    def _on_enter(self, event):
        """Handle mouse enter - show hover state."""
        if not self._disabled:
            self.inner.config(bg=self.colors['hover'])
            self.label.config(bg=self.colors['hover'])
    
    def _on_leave(self, event):
        """Handle mouse leave - restore normal state."""
        if not self._disabled:
            self.inner.config(bg=self.colors['bg'])
            self.label.config(bg=self.colors['bg'])
    
    def config(self, **kwargs):
        """Handle config changes, especially state."""
        if 'state' in kwargs:
            state_val = kwargs.pop('state')
            if state_val == tk.DISABLED:
                self._disabled = True
                self.inner.config(bg=self.colors['disabled_bg'])
                self.label.config(
                    bg=self.colors['disabled_bg'],
                    fg=self.colors['disabled_fg'],
                    cursor="arrow"
                )
                super().configure(bg=self.colors['disabled_bg'])
            else:
                self._disabled = False
                self.inner.config(bg=self.colors['bg'])
                self.label.config(
                    bg=self.colors['bg'],
                    fg=self.colors['fg'],
                    cursor="hand2"
                )
                super().configure(bg=self.colors['border'])
        # Handle other kwargs by passing to parent
        if kwargs:
            super().configure(**kwargs)
    
    def configure(self, **kwargs):
        """Alias for config."""
        self.config(**kwargs)


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
        # Enable Windows DPI awareness before creating the Tk root
        if sys.platform.startswith("win"):
            try:
                # Try newer API (Windows 8.1+)
                ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
            except Exception:
                try:
                    # Fallback for older Windows
                    ctypes.windll.user32.SetProcessDPIAware()
                except Exception:
                    pass

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
        self.gantt_rectangles: List[Dict] = []  # For hover detection
        
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
                   font=(DEFAULT_FONT_FAMILY, 10))
        
        style.configure("Header.TLabel",
                   font=(DEFAULT_FONT_FAMILY, 16, "bold"),
                   foreground=ModernColors.TEXT_PRIMARY)
        
        style.configure("SubHeader.TLabel",
                   font=(DEFAULT_FONT_FAMILY, 12, "bold"),
                   foreground=ModernColors.TEXT_SECONDARY)
        
        style.configure("TLabelframe",
                       background=ModernColors.BG_CARD,
                       foreground=ModernColors.TEXT_PRIMARY)
        style.configure("TLabelframe.Label",
                   background=ModernColors.BG_CARD,
                   foreground=ModernColors.PRIMARY,
                   font=(DEFAULT_FONT_FAMILY, 11, "bold"))
        
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
                   font=(DEFAULT_FONT_FAMILY, 9))
        style.configure("Treeview.Heading",
                   background=ModernColors.BG_CARD_HOVER,
                   foreground=ModernColors.TEXT_PRIMARY,
                   font=(DEFAULT_FONT_FAMILY, 10, "bold"))
        style.map("Treeview",
                 background=[('selected', ModernColors.PRIMARY)],
                 foreground=[('selected', '#ffffff')])
        
        # Scale style
        style.configure("TScale",
                       background=ModernColors.BG_DARK,
                       troughcolor=ModernColors.BG_INPUT)

    # --- Platform-aware mouse wheel helpers ---
    def _on_mousewheel(self, event, widget):
        """Normalize mouse-wheel event across platforms and scroll `widget`."""
        try:
            if sys.platform.startswith("win"):
                # Windows: event.delta is multiple of 120
                delta = int(event.delta / 120)
                widget.yview_scroll(-delta, "units")
            elif sys.platform == "darwin":
                # macOS: event.delta is small; invert as needed
                delta = int(event.delta)
                widget.yview_scroll(-delta, "units")
            else:
                # X11: use Button-4 (up) and Button-5 (down)
                if hasattr(event, 'num'):
                    if event.num == 4:
                        widget.yview_scroll(-1, "units")
                    elif event.num == 5:
                        widget.yview_scroll(1, "units")
        except Exception:
            pass

    def _bind_mousewheel(self, widget):
        """Bind platform-appropriate mouse wheel events to a scrollable widget.

        This binds handlers on Enter/Leave so the wheel affects the widget
        under the cursor instead of global scrolling.
        """
        if sys.platform.startswith("win") or sys.platform == "darwin":
            def _on_enter(e):
                widget.bind_all("<MouseWheel>", lambda ev: self._on_mousewheel(ev, widget))
            def _on_leave(e):
                widget.unbind_all("<MouseWheel>")
        else:
            def _on_enter(e):
                widget.bind_all("<Button-4>", lambda ev: self._on_mousewheel(ev, widget))
                widget.bind_all("<Button-5>", lambda ev: self._on_mousewheel(ev, widget))
            def _on_leave(e):
                widget.unbind_all("<Button-4>")
                widget.unbind_all("<Button-5>")

        widget.bind("<Enter>", _on_enter)
        widget.bind("<Leave>", _on_leave)
    
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
        """Create the modern main window layout that fills the screen."""
        # Main container - fills entire window
        main_frame = tk.Frame(self.root, bg=ModernColors.BG_DARK, padx=15, pady=15)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header with title
        self._create_header(main_frame)
        
        # Top section: Control Panel
        self._create_control_panel(main_frame)
        
        # Middle section: Visualization (Processors + Charts) - fills remaining space
        middle_frame = tk.Frame(main_frame, bg=ModernColors.BG_DARK)
        middle_frame.pack(fill=tk.BOTH, expand=True, pady=15)
        
        # Left: Processors Panel
        self._create_processor_panel(middle_frame)
        
        # Right: Charts and Process Table with resizable panes
        right_frame = tk.Frame(middle_frame, bg=ModernColors.BG_DARK)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(15, 0))
        
        # Create PanedWindow for resizable sections
        self.right_paned = tk.PanedWindow(right_frame, orient=tk.VERTICAL,
                                          bg=ModernColors.BG_DARK,
                                          sashwidth=8,
                                          sashrelief=tk.FLAT,
                                          sashpad=2)
        self.right_paned.pack(fill=tk.BOTH, expand=True)
        
        # Top pane: Gantt chart
        gantt_frame = tk.Frame(self.right_paned, bg=ModernColors.BG_DARK)
        self._create_gantt_chart(gantt_frame)
        self.right_paned.add(gantt_frame, minsize=200, height=320)
        
        # Bottom pane: Process table - fills remaining space
        table_frame = tk.Frame(self.right_paned, bg=ModernColors.BG_DARK)
        self._create_process_table(table_frame)
        self.right_paned.add(table_frame, minsize=200)
    
    def _create_header(self, parent):
        """Create the header section with title and status."""
        header_frame = tk.Frame(parent, bg=ModernColors.BG_DARK)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Title with icon
        title_frame = tk.Frame(header_frame, bg=ModernColors.BG_DARK)
        title_frame.pack(side=tk.LEFT)
        
        tk.Label(title_frame, text="⚡", font=(DEFAULT_FONT_FAMILY, 24),
                bg=ModernColors.BG_DARK, fg=ModernColors.PRIMARY).pack(side=tk.LEFT)
        
        tk.Label(title_frame, text=APP_NAME,
                font=(DEFAULT_FONT_FAMILY, 20, "bold"),
                bg=ModernColors.BG_DARK,
                fg=ModernColors.TEXT_PRIMARY).pack(side=tk.LEFT, padx=(10, 0))
        
        tk.Label(title_frame, text=f"v{VERSION}",
                font=(DEFAULT_FONT_FAMILY, 12),
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
            font=(DEFAULT_FONT_FAMILY, 14, "bold"),
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
            font=(DEFAULT_FONT_FAMILY, 10),
                bg=ModernColors.BG_CARD,
                fg=ModernColors.TEXT_SECONDARY).pack(anchor=tk.W)
        
        self.algorithm_var = tk.StringVar(value=self.config.default_algorithm.value)
        algorithm_combo = ttk.Combobox(algo_frame, textvariable=self.algorithm_var,
                                        values=[a.value for a in LoadBalancingAlgorithm],
                                        state="readonly", width=18,
                                        font=(DEFAULT_FONT_FAMILY, 10))
        algorithm_combo.pack(pady=(5, 0))
        algorithm_combo.bind("<<ComboboxSelected>>", self._on_algorithm_changed)
        
        # AI Training Mode toggle (shown only for Q-Learning)
        self.ai_frame = tk.Frame(config_frame, bg=ModernColors.BG_CARD)
        # Don't pack yet - will show/hide based on algorithm selection
        
        tk.Label(self.ai_frame, text="🤖 AI Mode",
                font=(DEFAULT_FONT_FAMILY, 10),
                bg=ModernColors.BG_CARD,
                fg=ModernColors.TEXT_SECONDARY).pack(anchor=tk.W)
        
        ai_inner = tk.Frame(self.ai_frame, bg=ModernColors.BG_CARD)
        ai_inner.pack(pady=(5, 0))
        
        self.ai_training_var = tk.BooleanVar(value=True)
        self.ai_train_radio = tk.Radiobutton(ai_inner, text="Train",
                                              variable=self.ai_training_var, value=True,
                                              bg=ModernColors.BG_CARD, fg=ModernColors.TEXT_PRIMARY,
                                              selectcolor=ModernColors.BG_INPUT,
                                              activebackground=ModernColors.BG_CARD,
                                              font=(DEFAULT_FONT_FAMILY, 9))
        self.ai_train_radio.pack(side=tk.LEFT)
        
        self.ai_exploit_radio = tk.Radiobutton(ai_inner, text="Exploit",
                                                variable=self.ai_training_var, value=False,
                                                bg=ModernColors.BG_CARD, fg=ModernColors.TEXT_PRIMARY,
                                                selectcolor=ModernColors.BG_INPUT,
                                                activebackground=ModernColors.BG_CARD,
                                                font=(DEFAULT_FONT_FAMILY, 9))
        self.ai_exploit_radio.pack(side=tk.LEFT, padx=(5, 0))
        
        # Speed control
        speed_frame = tk.Frame(config_frame, bg=ModernColors.BG_CARD)
        speed_frame.pack(side=tk.LEFT, padx=(0, 30))
        
        tk.Label(speed_frame, text="⚡ Speed",
            font=(DEFAULT_FONT_FAMILY, 10),
                bg=ModernColors.BG_CARD,
                fg=ModernColors.TEXT_SECONDARY).pack(anchor=tk.W)
        
        speed_inner = tk.Frame(speed_frame, bg=ModernColors.BG_CARD)
        speed_inner.pack(pady=(5, 0))
        
        self.speed_var = tk.IntVar(value=50)
        speed_scale = ttk.Scale(speed_inner, from_=1, to=100, orient=tk.HORIZONTAL,
                                 variable=self.speed_var, length=120)
        speed_scale.pack(side=tk.LEFT)
        
        self.speed_label = tk.Label(speed_inner, text="50%",
                        font=(DEFAULT_FONT_FAMILY, 10, "bold"),
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
                          font=(DEFAULT_FONT_FAMILY, 11, "bold"),
                          bg=ModernColors.BG_CARD,
                          fg=ModernColors.TEXT_SECONDARY)
        self.status_label.pack(side=tk.LEFT, padx=(8, 0))
        
        # Time and progress
        status_right = tk.Frame(status_frame, bg=ModernColors.BG_CARD)
        status_right.pack(side=tk.RIGHT)
        
        self.time_label = tk.Label(status_right, text="⏱️ Time: 0",
                        font=(DEFAULT_FONT_FAMILY, 11),
                        bg=ModernColors.BG_CARD,
                        fg=ModernColors.TEXT_PRIMARY)
        self.time_label.pack(side=tk.RIGHT, padx=(20, 0))
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(status_right, variable=self.progress_var,
                                             maximum=100, length=250, mode='determinate')
        self.progress_bar.pack(side=tk.RIGHT)
        
        tk.Label(status_right, text="Progress:",
            font=(DEFAULT_FONT_FAMILY, 10),
            bg=ModernColors.BG_CARD,
            fg=ModernColors.TEXT_SECONDARY).pack(side=tk.RIGHT, padx=(0, 10))
    
    def _create_config_input(self, parent, label: str, var_name: str,
                            min_val: int, max_val: int, default: int):
        """Create a modern config input field."""
        frame = tk.Frame(parent, bg=ModernColors.BG_CARD)
        frame.pack(side=tk.LEFT, padx=(0, 30))
        
        tk.Label(frame, text=label,
            font=(DEFAULT_FONT_FAMILY, 10),
            bg=ModernColors.BG_CARD,
            fg=ModernColors.TEXT_SECONDARY).pack(anchor=tk.W)
        
        var = tk.IntVar(value=default)
        setattr(self, f"{var_name}_var", var)
        
        spinbox = ttk.Spinbox(frame, from_=min_val, to=max_val, width=8,
                      textvariable=var, font=(DEFAULT_FONT_FAMILY, 10))
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
            font=(DEFAULT_FONT_FAMILY, 14, "bold"),
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
        """Create the modern Gantt chart visualization with enhanced features."""
        panel = tk.Frame(parent, bg=ModernColors.BG_CARD,
                        highlightbackground=ModernColors.BG_CARD_HOVER,
                        highlightthickness=1)
        panel.pack(fill=tk.BOTH, expand=True)
        
        inner = tk.Frame(panel, bg=ModernColors.BG_CARD, padx=15, pady=15)
        inner.pack(fill=tk.BOTH, expand=True)
        
        # Header with title and controls
        header_frame = tk.Frame(inner, bg=ModernColors.BG_CARD)
        header_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(header_frame, text="📊 Process Execution Timeline",
                font=("Helvetica Neue", 14, "bold"),
                bg=ModernColors.BG_CARD,
                fg=ModernColors.PRIMARY).pack(side=tk.LEFT)
        
        # Timeline stats frame (right side of header)
        self.timeline_stats_frame = tk.Frame(header_frame, bg=ModernColors.BG_CARD)
        self.timeline_stats_frame.pack(side=tk.RIGHT)
        
        self.timeline_time_label = tk.Label(
            self.timeline_stats_frame, 
            text="⏱ Duration: 0",
            font=("Helvetica Neue", 10),
            bg=ModernColors.BG_CARD,
            fg=ModernColors.TEXT_SECONDARY
        )
        self.timeline_time_label.pack(side=tk.LEFT, padx=(0, 15))
        
        self.timeline_processes_label = tk.Label(
            self.timeline_stats_frame, 
            text="📦 Processes: 0",
            font=("Helvetica Neue", 10),
            bg=ModernColors.BG_CARD,
            fg=ModernColors.TEXT_SECONDARY
        )
        self.timeline_processes_label.pack(side=tk.LEFT)
        
        # Create matplotlib figure with dark theme - larger and better proportioned
        self.gantt_fig = Figure(figsize=(10, 4), dpi=100, facecolor=ModernColors.BG_CARD)
        self.gantt_ax = self.gantt_fig.add_subplot(111)
        self._style_gantt_axes(self.gantt_ax)
        
        # Add padding for the legend
        self.gantt_fig.subplots_adjust(left=0.08, right=0.82, top=0.92, bottom=0.15)
        
        # Embed in Tkinter
        self.gantt_canvas = FigureCanvasTkAgg(self.gantt_fig, master=inner)
        self.gantt_canvas.draw()
        canvas_widget = self.gantt_canvas.get_tk_widget()
        canvas_widget.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events for tooltips
        self.gantt_canvas.mpl_connect('motion_notify_event', self._on_gantt_hover)
        
        # Tooltip label (hidden by default)
        self.gantt_tooltip = tk.Label(
            inner,
            text="",
            font=("Helvetica Neue", 9),
            bg="#2d2d44",
            fg=ModernColors.TEXT_PRIMARY,
            padx=8,
            pady=4,
            relief=tk.SOLID,
            borderwidth=1
        )
        
        # Store rectangle data for hover detection
        self.gantt_rectangles = []
        
        # Legend frame below chart
        self.legend_frame = tk.Frame(inner, bg=ModernColors.BG_CARD)
        self.legend_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Legend title
        tk.Label(self.legend_frame, text="Legend:",
                font=("Helvetica Neue", 9, "bold"),
                bg=ModernColors.BG_CARD,
                fg=ModernColors.TEXT_SECONDARY).pack(side=tk.LEFT, padx=(0, 10))
        
        # Legend items container
        self.legend_items_frame = tk.Frame(self.legend_frame, bg=ModernColors.BG_CARD)
        self.legend_items_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    def _style_gantt_axes(self, ax):
        """Apply modern dark styling to Gantt chart axes."""
        ax.set_facecolor("#0d1117")  # Darker background for better contrast
        ax.tick_params(colors=ModernColors.TEXT_SECONDARY, labelsize=9)
        ax.spines['bottom'].set_color(ModernColors.TEXT_MUTED)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_color(ModernColors.TEXT_MUTED)
        ax.spines['right'].set_visible(False)
        ax.xaxis.label.set_color(ModernColors.TEXT_SECONDARY)
        ax.yaxis.label.set_color(ModernColors.TEXT_SECONDARY)
        ax.set_xlabel("Time Units", fontsize=10, fontweight='bold')
        ax.set_ylabel("Processors", fontsize=10, fontweight='bold')
        ax.title.set_color(ModernColors.TEXT_PRIMARY)
    
    def _create_process_table(self, parent):
        """Create the modern process table display using Text widget."""
        # Main panel - now resizable via PanedWindow
        panel = tk.Frame(parent, bg=ModernColors.BG_CARD,
                        highlightbackground=ModernColors.BG_CARD_HOVER,
                        highlightthickness=1)
        panel.pack(fill=tk.BOTH, expand=True, pady=(5, 0))
        
        # Header row with title and stats
        header_frame = tk.Frame(panel, bg=ModernColors.BG_CARD)
        header_frame.pack(fill=tk.X, padx=15, pady=(10, 5))
        
        tk.Label(header_frame, text="📋 Process Details",
                font=("Helvetica Neue", 14, "bold"),
                bg=ModernColors.BG_CARD,
                fg=ModernColors.PRIMARY).pack(side=tk.LEFT)
        
        # Process stats on right
        stats_frame = tk.Frame(header_frame, bg=ModernColors.BG_CARD)
        stats_frame.pack(side=tk.RIGHT)
        
        self.running_count_label = tk.Label(stats_frame, text="🏃 Running: 0",
            font=("Helvetica Neue", 10), bg=ModernColors.BG_CARD, fg=ModernColors.SUCCESS)
        self.running_count_label.pack(side=tk.LEFT, padx=(0, 15))
        
        self.waiting_count_label = tk.Label(stats_frame, text="⏳ Waiting: 0",
            font=("Helvetica Neue", 10), bg=ModernColors.BG_CARD, fg=ModernColors.WARNING)
        self.waiting_count_label.pack(side=tk.LEFT, padx=(0, 15))
        
        self.completed_count_label = tk.Label(stats_frame, text="✅ Done: 0",
            font=("Helvetica Neue", 10), bg=ModernColors.BG_CARD, fg=ModernColors.INFO)
        self.completed_count_label.pack(side=tk.LEFT)
        
        # Table area with border
        table_border = tk.Frame(panel, bg="#30363d")
        table_border.pack(fill=tk.BOTH, expand=True, padx=15, pady=(5, 10))
        
        table_inner = tk.Frame(table_border, bg="#0d1117")
        table_inner.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)
        
        # Column headers
        columns = ["PID", "Arrival", "Burst", "Remain", "Priority", "State", "CPU", "Wait", "TAT"]
        col_widths = [6, 7, 6, 7, 8, 10, 6, 6, 8]
        
        header_row = tk.Frame(table_inner, bg="#161b22")
        header_row.pack(fill=tk.X)
        
        for col, width in zip(columns, col_widths):
            tk.Label(header_row, text=col, font=("Courier", 10, "bold"),
                    bg="#161b22", fg="#58a6ff", width=width).pack(side=tk.LEFT, padx=1, pady=4)
        
        # Text widget for data rows
        text_container = tk.Frame(table_inner, bg="#0d1117")
        text_container.pack(fill=tk.BOTH, expand=True)
        
        self.process_text = tk.Text(text_container, 
                                    bg="#0d1117", fg="#e6edf3",
                                    font=("Courier", 10),
                                    height=6, wrap=tk.NONE,
                                    cursor="arrow",
                                    highlightthickness=0, borderwidth=0,
                                    padx=3, pady=3)
        
        scrollbar = tk.Scrollbar(text_container, command=self.process_text.yview)
        self.process_text.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.process_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Configure text tags for states
        self.process_text.tag_configure("running", foreground="#4ade80", background="#1a4d1a")
        self.process_text.tag_configure("ready", foreground="#60a5fa", background="#1a3a5c")
        self.process_text.tag_configure("waiting", foreground="#fbbf24", background="#4d3d1a")
        self.process_text.tag_configure("completed", foreground="#9ca3af", background="#1a1a2e")
        self.process_text.tag_configure("new", foreground="#c084fc", background="#2d1a4d")
        self.process_text.tag_configure("default", foreground="#e6edf3", background="#0d1117")
        
        # Store column widths
        self.table_col_widths = col_widths
        self.table_row_widgets = {}
        
        # Add placeholder
        self.process_text.insert(tk.END, " Click 'Start' to see process data...\n", "default")
        self.process_text.configure(state=tk.DISABLED)
    
    def _update_speed_label(self, event=None):
        """Update the speed label when slider changes."""
        self.speed_label.config(text=f"{self.speed_var.get()}%")
    
    def _on_algorithm_changed(self, event=None):
        """Handle algorithm selection change - show/hide AI controls."""
        algo_name = self.algorithm_var.get()
        if algo_name in (LoadBalancingAlgorithm.Q_LEARNING.value, 
                         LoadBalancingAlgorithm.DQN.value):
            # Show AI controls for both Q-Learning and DQN
            self.ai_frame.pack(side=tk.LEFT, padx=(0, 30))
        else:
            # Hide AI controls
            self.ai_frame.pack_forget()
    
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
        
        # Configure AI mode if Q-Learning or DQN is selected
        if algorithm in (LoadBalancingAlgorithm.Q_LEARNING, LoadBalancingAlgorithm.DQN):
            training_mode = self.ai_training_var.get()
            if hasattr(self.engine.load_balancer, 'set_training_mode'):
                self.engine.load_balancer.set_training_mode(training_mode)
                # Try to load existing model
                if hasattr(self.engine.load_balancer, 'load_model'):
                    self.engine.load_balancer.load_model()
        
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
        
        # Update AI stats if using Q-Learning
        if self.engine and hasattr(self.engine.load_balancer, 'get_statistics'):
            self._update_ai_stats()
        
        # Update process table
        self._update_process_table()
        
        # Update Gantt chart more frequently (every 2 time units or when there's data)
        if state['time'] % 2 == 0 or state['time'] <= 5:
            self._update_gantt_chart()
    
    def _update_ai_stats(self):
        """Update AI statistics display during simulation."""
        if not self.engine or not hasattr(self.engine.load_balancer, 'get_statistics'):
            return
        
        stats = self.engine.load_balancer.get_statistics()
        
        # Update status label with AI info when running Q-Learning or DQN
        algo_name = self.algorithm_var.get()
        if algo_name == LoadBalancingAlgorithm.Q_LEARNING.value:
            mode = "Training" if stats.get('training_mode', True) else "Exploiting"
            epsilon = stats.get('epsilon', 0) * 100
            q_states = stats.get('q_table_size', 0)
            self.status_label.config(
                text=f"🤖 {mode} (ε={epsilon:.1f}%, Q-states={q_states})",
                fg=ModernColors.PRIMARY if stats.get('training_mode') else ModernColors.SUCCESS
            )
        elif algo_name == LoadBalancingAlgorithm.DQN.value:
            mode = "Training" if stats.get('training_mode', True) else "Evaluating"
            epsilon = stats.get('epsilon', 0) * 100
            steps = stats.get('total_steps', 0)
            avg_loss = stats.get('avg_loss', 0)
            self.status_label.config(
                text=f"🧠 {mode} (ε={epsilon:.1f}%, steps={steps}, loss={avg_loss:.4f})",
                fg=ModernColors.PRIMARY if stats.get('training_mode') else ModernColors.SUCCESS
            )
    
    def _handle_completion(self, result):
        """Handle simulation completion."""
        self.is_running = False
        self._update_button_states()
        self.status_label.config(text="Completed", fg=ModernColors.SUCCESS)
        self._draw_status_led(ModernColors.PRIMARY)
        
        # Get metrics for completion message
        metrics = result.system_metrics
        
        # Final gantt chart update
        self._update_gantt_chart()
        
        # Final process table update
        self._update_process_table()
        
        # Save AI model if using Q-Learning or DQN
        algo_name = self.algorithm_var.get()
        if algo_name in (LoadBalancingAlgorithm.Q_LEARNING.value, 
                         LoadBalancingAlgorithm.DQN.value):
            if self.engine and hasattr(self.engine.load_balancer, 'save_model'):
                self.engine.load_balancer.save_model()
                ai_stats = ""
                if hasattr(self.engine.load_balancer, 'get_statistics'):
                    stats = self.engine.load_balancer.get_statistics()
                    if algo_name == LoadBalancingAlgorithm.Q_LEARNING.value:
                        ai_stats = f"\n\n🤖 AI Stats:\nEpisodes: {stats.get('episode_count', 0)}\nQ-States: {stats.get('q_table_size', 0)}\nExploration: {stats.get('epsilon', 0)*100:.1f}%"
                    else:
                        ai_stats = f"\n\n🧠 DQN Stats:\nEpisodes: {stats.get('episode_count', 0)}\nSteps: {stats.get('total_steps', 0)}\nAvg Loss: {stats.get('avg_loss', 0):.4f}\nExploration: {stats.get('epsilon', 0)*100:.1f}%"
                messagebox.showinfo("Simulation Complete", 
                               f"Simulation completed in {result.total_time} time units.\n"
                               f"Completed: {metrics.completed_processes}/{metrics.total_processes} processes\n"
                               f"Average Turnaround: {metrics.avg_turnaround_time:.2f}\n"
                               f"Average Utilization: {metrics.avg_utilization*100:.1f}%"
                               f"{ai_stats}\n\n✅ AI model saved!")
                return
        
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
        """Update the Gantt chart visualization with modern styling and animations."""
        self.gantt_ax.clear()
        self._style_gantt_axes(self.gantt_ax)
        self.gantt_rectangles.clear()
        
        if not self.gantt_data:
            # Show empty state message
            self.gantt_ax.text(0.5, 0.5, "Waiting for simulation data...",
                             transform=self.gantt_ax.transAxes,
                             ha='center', va='center',
                             fontsize=12, color=ModernColors.TEXT_MUTED,
                             style='italic')
            self.gantt_canvas.draw()
            self._update_timeline_stats(0, 0)
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
        
        # Modern color palette for processes - vibrant and distinct
        modern_colors = [
            '#6366f1',  # Indigo
            '#22c55e',  # Green
            '#f59e0b',  # Amber
            '#ec4899',  # Pink
            '#06b6d4',  # Cyan
            '#8b5cf6',  # Violet
            '#ef4444',  # Red
            '#14b8a6',  # Teal
            '#f97316',  # Orange
            '#84cc16',  # Lime
            '#a855f7',  # Purple
            '#0ea5e9',  # Sky
        ]
        
        for entry in consolidated:
            pid = entry['process']
            if pid not in process_colors:
                process_colors[pid] = modern_colors[color_index % len(modern_colors)]
                color_index += 1
            
            # Create 3D-effect rounded rectangle
            x = entry['start']
            y = entry['processor'] - 0.4
            width = entry['end'] - entry['start']
            height = 0.8
            
            # Shadow effect (darker rectangle behind)
            shadow = FancyBboxPatch(
                (x + 0.05, y - 0.05),
                width, height,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                facecolor='#000000',
                alpha=0.3,
                linewidth=0
            )
            self.gantt_ax.add_patch(shadow)
            
            # Main rectangle with gradient effect
            rect = FancyBboxPatch(
                (x, y),
                width, height,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                facecolor=process_colors[pid],
                edgecolor='white',
                linewidth=1,
                alpha=0.95
            )
            self.gantt_ax.add_patch(rect)
            
            # Store rectangle data for hover
            self.gantt_rectangles.append({
                'x': x, 'y': y, 'width': width, 'height': height,
                'pid': pid, 'processor': entry['processor'],
                'start': entry['start'], 'end': entry['end'],
                'color': process_colors[pid]
            })
            
            # Add process label with better styling
            if width >= 1.5:
                # White text with subtle shadow for readability
                self.gantt_ax.text(
                    x + width/2,
                    entry['processor'],
                    f"P{pid}",
                    ha='center', va='center',
                    fontsize=9 if width >= 3 else 7,
                    fontweight='bold',
                    color='white',
                    path_effects=[
                        matplotlib.patheffects.withStroke(linewidth=2, foreground='black')
                    ] if width >= 2 else None
                )
        
        # Configure axes with modern styling
        max_time = max(e['end'] for e in consolidated) if consolidated else 10
        self.gantt_ax.set_xlim(-0.5, max_time + 1)
        self.gantt_ax.set_ylim(-0.6, num_processors - 0.4)
        self.gantt_ax.set_yticks(range(num_processors))
        self.gantt_ax.set_yticklabels([f"CPU {i}" for i in range(num_processors)],
                                      fontweight='bold')
        
        # Add subtle grid lines
        self.gantt_ax.grid(True, axis='x', alpha=0.15, color='white', linestyle='--')
        self.gantt_ax.grid(True, axis='y', alpha=0.1, color='white', linestyle='-')
        
        # Add time markers at intervals
        if max_time > 0:
            interval = max(1, int(max_time / 10))
            for t in range(0, int(max_time) + 1, interval):
                self.gantt_ax.axvline(x=t, color=ModernColors.TEXT_MUTED, 
                                     alpha=0.3, linestyle=':', linewidth=0.5)
        
        # Add current time indicator if simulation is running
        if self.is_running and self.engine:
            current_t = self.engine.current_time
            self.gantt_ax.axvline(x=current_t, color='#ef4444', 
                                 linestyle='-', linewidth=2, alpha=0.8)
            self.gantt_ax.text(current_t, num_processors - 0.3, f"t={current_t}",
                             fontsize=8, color='#ef4444', ha='center', fontweight='bold')
        
        self.gantt_fig.tight_layout(rect=[0.05, 0.1, 0.78, 0.95])
        self.gantt_canvas.draw()
        
        # Update legend with process colors
        self._update_gantt_legend(process_colors)
        
        # Update timeline stats
        unique_processes = len(process_colors)
        self._update_timeline_stats(max_time, unique_processes)
    
    def _on_gantt_hover(self, event):
        """Handle mouse hover on Gantt chart for tooltips."""
        if event.inaxes != self.gantt_ax or not self.gantt_rectangles:
            self.gantt_tooltip.place_forget()
            return
        
        x, y = event.xdata, event.ydata
        if x is None or y is None:
            self.gantt_tooltip.place_forget()
            return
        
        # Check if hovering over any rectangle
        for rect in self.gantt_rectangles:
            if (rect['x'] <= x <= rect['x'] + rect['width'] and
                rect['y'] <= y <= rect['y'] + rect['height']):
                # Show tooltip
                duration = rect['end'] - rect['start']
                tooltip_text = (f"Process P{rect['pid']}\n"
                               f"CPU: {rect['processor']}\n"
                               f"Time: {rect['start']} → {rect['end']}\n"
                               f"Duration: {duration} units")
                self.gantt_tooltip.config(text=tooltip_text)
                
                # Position tooltip near cursor
                canvas_widget = self.gantt_canvas.get_tk_widget()
                x_pos = event.x + 15
                y_pos = event.y + 15
                self.gantt_tooltip.place(in_=canvas_widget, x=x_pos, y=y_pos)
                return
        
        self.gantt_tooltip.place_forget()
    
    def _update_gantt_legend(self, process_colors: dict):
        """Update the legend with process colors."""
        # Clear existing legend items
        for widget in self.legend_items_frame.winfo_children():
            widget.destroy()
        
        # Create legend items for each process (limit to 12 to avoid overflow)
        items_shown = 0
        for pid, color in sorted(process_colors.items()):
            if items_shown >= 12:
                # Show "and more..." indicator
                tk.Label(self.legend_items_frame, 
                        text=f"+{len(process_colors) - 12} more",
                        font=("Helvetica Neue", 8),
                        bg=ModernColors.BG_CARD,
                        fg=ModernColors.TEXT_MUTED).pack(side=tk.LEFT, padx=5)
                break
            
            item_frame = tk.Frame(self.legend_items_frame, bg=ModernColors.BG_CARD)
            item_frame.pack(side=tk.LEFT, padx=4)
            
            # Color box
            color_box = tk.Canvas(item_frame, width=12, height=12, 
                                 bg=ModernColors.BG_CARD, highlightthickness=0)
            color_box.pack(side=tk.LEFT, padx=(0, 3))
            color_box.create_rectangle(1, 1, 11, 11, fill=color, outline='white', width=1)
            
            # Process label
            tk.Label(item_frame, text=f"P{pid}",
                    font=("Helvetica Neue", 8),
                    bg=ModernColors.BG_CARD,
                    fg=ModernColors.TEXT_SECONDARY).pack(side=tk.LEFT)
            
            items_shown += 1
    
    def _update_timeline_stats(self, duration: float, num_processes: int):
        """Update timeline statistics labels."""
        self.timeline_time_label.config(text=f"⏱ Duration: {int(duration)} units")
        self.timeline_processes_label.config(text=f"📦 Processes: {num_processes}")
    
    def _clear_gantt_chart(self):
        """Clear the Gantt chart."""
        self.gantt_data.clear()
        self.gantt_rectangles.clear()
        self.gantt_ax.clear()
        self._style_gantt_axes(self.gantt_ax)
        
        # Show waiting message
        self.gantt_ax.text(0.5, 0.5, "Waiting for simulation data...",
                         transform=self.gantt_ax.transAxes,
                         ha='center', va='center',
                         fontsize=12, color=ModernColors.TEXT_MUTED,
                         style='italic')
        
        self.gantt_canvas.draw()
        
        # Clear legend
        for widget in self.legend_items_frame.winfo_children():
            widget.destroy()
        
        # Reset stats
        self._update_timeline_stats(0, 0)
        
        # Hide tooltip
        self.gantt_tooltip.place_forget()
    
    def _populate_process_table(self):
        """Populate the process table with initial data."""
        self._clear_process_table()
        
        if not self.engine:
            return
        
        for p in self.engine.all_processes:
            self._add_process_row(p)
        
        self._update_process_stats()
    
    def _format_row(self, values):
        """Format a row of values with fixed widths."""
        formatted = []
        for val, width in zip(values, self.table_col_widths):
            formatted.append(str(val).center(width))
        return " ".join(formatted)
    
    def _add_process_row(self, process):
        """Add a single process row to the text widget."""
        turnaround = process.get_turnaround_time()
        
        values = [
            f"P{process.pid}",
            str(process.arrival_time),
            str(process.burst_time),
            str(process.remaining_time),
            process.priority.name[:6].capitalize(),
            process.state.name.replace('_', ' ')[:9].title(),
            f"CPU{process.processor_id}" if process.processor_id is not None else "—",
            str(process.waiting_time),
            str(turnaround) if turnaround is not None else "—"
        ]
        
        row_text = self._format_row(values) + "\n"
        tag = self._get_state_tag(process.state.name)
        
        self.process_text.configure(state=tk.NORMAL)
        self.process_text.insert(tk.END, row_text, tag)
        self.process_text.configure(state=tk.DISABLED)
    
    def _update_process_table(self):
        """Update process table with current data and state-based coloring."""
        if not self.engine:
            return
        
        # Clear and rebuild the text content
        self.process_text.configure(state=tk.NORMAL)
        self.process_text.delete("1.0", tk.END)
        
        for p in self.engine.all_processes:
            turnaround = p.get_turnaround_time()
            tag = self._get_state_tag(p.state.name)
            
            values = [
                f"P{p.pid}",
                str(p.arrival_time),
                str(p.burst_time),
                str(p.remaining_time),
                p.priority.name[:6].capitalize(),
                p.state.name.replace('_', ' ')[:9].title(),
                f"CPU{p.processor_id}" if p.processor_id is not None else "—",
                str(p.waiting_time),
                str(turnaround) if turnaround is not None else "—"
            ]
            
            row_text = self._format_row(values) + "\n"
            self.process_text.insert(tk.END, row_text, tag)
        
        self.process_text.configure(state=tk.DISABLED)
        self._update_process_stats()
    
    def _get_state_tag(self, state_name: str) -> str:
        """Get the tag name for a process state."""
        tag_map = {
            'RUNNING': 'running',
            'READY': 'ready',
            'WAITING': 'waiting',
            'COMPLETED': 'completed',
            'NEW': 'new',
            'TERMINATED': 'completed'
        }
        return tag_map.get(state_name, 'default')
    
    def _update_process_stats(self):
        """Update the process statistics in the header."""
        if not self.engine:
            self.running_count_label.config(text="🏃 Running: 0")
            self.waiting_count_label.config(text="⏳ Waiting: 0")
            self.completed_count_label.config(text="✅ Done: 0")
            return
        
        running = sum(1 for p in self.engine.all_processes if p.state.name == 'RUNNING')
        waiting = sum(1 for p in self.engine.all_processes if p.state.name in ['READY', 'WAITING', 'NEW'])
        completed = sum(1 for p in self.engine.all_processes if p.state.name in ['COMPLETED', 'TERMINATED'])
        
        self.running_count_label.config(text=f"🏃 Running: {running}")
        self.waiting_count_label.config(text=f"⏳ Waiting: {waiting}")
        self.completed_count_label.config(text=f"✅ Done: {completed}")
    
    def _show_process_details_by_pid(self, pid):
        """Show detailed information about a process by PID."""
        if not self.engine:
            return
        
        # Find the process
        process = None
        for p in self.engine.all_processes:
            if p.pid == pid:
                process = p
                break
        
        if not process:
            return
        
        self._display_process_details_window(process)
    
    def _show_process_details(self, event):
        """Show detailed information about a process when double-clicked (legacy)."""
        pass  # Now using _show_process_details_by_pid instead
    
    def _display_process_details_window(self, process):
        """Display the process details window."""
        
        # Create detail popup
        detail_window = tk.Toplevel(self.root)
        detail_window.title(f"Process P{process.pid} Details")
        detail_window.geometry("380x420")
        detail_window.configure(bg=ModernColors.BG_DARK)
        detail_window.resizable(False, False)
        
        # Center on parent
        detail_window.transient(self.root)
        
        # Content
        content = tk.Frame(detail_window, bg=ModernColors.BG_CARD, padx=25, pady=25)
        content.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Header
        tk.Label(content, text=f"📄 Process P{process.pid}",
                font=("Helvetica Neue", 16, "bold"),
                bg=ModernColors.BG_CARD,
                fg=ModernColors.PRIMARY).pack(anchor=tk.W, pady=(0, 20))
        
        # Details grid
        details = [
            ("State", process.state.name.replace('_', ' ').title(), self._get_state_color(process.state.name)),
            ("Priority", process.priority.name.capitalize(), ModernColors.TEXT_PRIMARY),
            ("Arrival Time", str(process.arrival_time), ModernColors.TEXT_SECONDARY),
            ("Burst Time", str(process.burst_time), ModernColors.TEXT_SECONDARY),
            ("Remaining Time", str(process.remaining_time), ModernColors.WARNING if process.remaining_time > 0 else ModernColors.SUCCESS),
            ("Processor", f"CPU {process.processor_id}" if process.processor_id is not None else "Not Assigned", ModernColors.TEXT_SECONDARY),
            ("Start Time", str(process.start_time) if process.start_time is not None else "—", ModernColors.TEXT_SECONDARY),
            ("Completion Time", str(process.completion_time) if process.completion_time is not None else "—", ModernColors.TEXT_SECONDARY),
            ("Waiting Time", str(process.waiting_time), ModernColors.INFO),
            ("Turnaround Time", str(process.get_turnaround_time()) if process.get_turnaround_time() is not None else "—", ModernColors.INFO),
        ]
        
        for label, value, color in details:
            row = tk.Frame(content, bg=ModernColors.BG_CARD)
            row.pack(fill=tk.X, pady=4)
            
            tk.Label(row, text=label + ":",
                    font=("Helvetica Neue", 11),
                    bg=ModernColors.BG_CARD,
                    fg=ModernColors.TEXT_MUTED,
                    width=16, anchor=tk.W).pack(side=tk.LEFT)
            
            tk.Label(row, text=value,
                    font=("Helvetica Neue", 11, "bold"),
                    bg=ModernColors.BG_CARD,
                    fg=color).pack(side=tk.LEFT)
        
        # Close button
        close_btn = ModernButton(content, "Close", detail_window.destroy,
                                style="secondary", icon="✕")
        close_btn.pack(pady=(20, 0))
    
    def _get_state_color(self, state_name: str) -> str:
        """Get color for a process state."""
        colors = {
            'RUNNING': ModernColors.SUCCESS,
            'READY': ModernColors.INFO,
            'WAITING': ModernColors.WARNING,
            'COMPLETED': '#9ca3af',
            'NEW': '#c084fc',
            'TERMINATED': '#9ca3af'
        }
        return colors.get(state_name, ModernColors.TEXT_PRIMARY)
    
    def _clear_process_table(self):
        """Clear all items from the process table."""
        self.process_text.configure(state=tk.NORMAL)
        self.process_text.delete("1.0", tk.END)
        self.process_text.configure(state=tk.DISABLED)
        self.table_row_widgets = {}
    
    def _reset_metrics(self):
        """Reset all metric displays."""
        pass  # Metrics panel removed
    
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
            font=(DEFAULT_FONT_FAMILY, 14, "bold"),
                bg=ModernColors.BG_CARD,
                fg=ModernColors.TEXT_PRIMARY).pack(pady=(0, 15))
        
        progress_bar = ttk.Progressbar(container, length=350, mode='determinate')
        progress_bar.pack(pady=10)
        
        status_label = tk.Label(container, text="",
                       font=(DEFAULT_FONT_FAMILY, 10),
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
                   font=(DEFAULT_FONT_FAMILY, 10, "bold"))
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
            font=(DEFAULT_FONT_FAMILY, 16, "bold"),
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
            font=(DEFAULT_FONT_FAMILY, 14, "bold"),
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
            tk.Label(card, text=label, font=(DEFAULT_FONT_FAMILY, 9),
                    bg=ModernColors.BG_INPUT, fg=ModernColors.TEXT_SECONDARY).pack(anchor=tk.W)
            tk.Label(card, text=str(value), font=(DEFAULT_FONT_FAMILY, 12, "bold"),
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
        tk.Label(container, text="⚡", font=(DEFAULT_FONT_FAMILY, 48),
            bg=ModernColors.BG_CARD, fg=ModernColors.PRIMARY).pack(pady=(0, 10))
        
        # Title
        tk.Label(container, text=APP_NAME,
            font=(DEFAULT_FONT_FAMILY, 18, "bold"),
                bg=ModernColors.BG_CARD,
                fg=ModernColors.TEXT_PRIMARY).pack()
        
        # Version
        tk.Label(container, text=f"Version {VERSION}",
            font=(DEFAULT_FONT_FAMILY, 11),
                bg=ModernColors.BG_CARD,
                fg=ModernColors.TEXT_MUTED).pack(pady=(5, 20))
        
        # Description
        desc = """An educational simulation demonstrating dynamic
load balancing algorithms in multiprocessor systems."""
        tk.Label(container, text=desc,
            font=(DEFAULT_FONT_FAMILY, 10),
            bg=ModernColors.BG_CARD,
            fg=ModernColors.TEXT_SECONDARY,
            justify=tk.CENTER).pack(pady=(0, 20))
        
        # Features
        features_frame = tk.Frame(container, bg=ModernColors.BG_INPUT, padx=20, pady=15)
        features_frame.pack(fill=tk.X, pady=(0, 20))
        
        tk.Label(features_frame, text="✨ Key Features",
            font=(DEFAULT_FONT_FAMILY, 11, "bold"),
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
                font=(DEFAULT_FONT_FAMILY, 10),
                bg=ModernColors.BG_INPUT,
                fg=ModernColors.TEXT_PRIMARY).pack(anchor=tk.W, pady=2)
        
        # OS Concepts
        concepts_frame = tk.Frame(container, bg=ModernColors.BG_INPUT, padx=20, pady=15)
        concepts_frame.pack(fill=tk.X)
        
        tk.Label(concepts_frame, text="📚 OS Concepts Demonstrated",
            font=(DEFAULT_FONT_FAMILY, 11, "bold"),
            bg=ModernColors.BG_INPUT,
            fg=ModernColors.SECONDARY).pack(anchor=tk.W, pady=(0, 10))
        
        concepts = ["Process Management", "CPU Scheduling", 
                   "Load Balancing", "Resource Utilization"]
        
        for c in concepts:
            tk.Label(concepts_frame, text=f"• {c}",
                font=(DEFAULT_FONT_FAMILY, 10),
                bg=ModernColors.BG_INPUT,
                fg=ModernColors.TEXT_PRIMARY).pack(anchor=tk.W, pady=1)
        
        # Footer
        tk.Label(container, text="Made with ❤️ for learning OS concepts",
            font=(DEFAULT_FONT_FAMILY, 9),
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
            font=(DEFAULT_FONT_FAMILY, 16, "bold"),
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
