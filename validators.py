"""
Validators Module for Dynamic Load Balancing Simulator

This module provides comprehensive input validation, error handling,
and boundary checking for all simulation parameters and operations.

Validation Categories:
1. Configuration Validation: SimulationConfig, GUIConfig parameters
2. Process Validation: Process attributes and state transitions
3. Runtime Validation: Simulation operations and state checks
4. Data Validation: Export/import data integrity

Design Philosophy:
- Fail fast with clear error messages
- Provide recovery suggestions where possible
- Log validation failures for debugging
- Support both strict and lenient validation modes

Author: Student
Date: December 2024
"""

import logging
from typing import Optional, Any, List, Dict, Tuple, Union
from dataclasses import dataclass
from enum import Enum

from config import (
    ProcessState,
    ProcessPriority,
    LoadBalancingAlgorithm,
    SimulationConfig,
    GUIConfig
)


# Configure logger
logger = logging.getLogger(__name__)


# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================

class ValidationError(Exception):
    """Base exception for validation errors."""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        self.field = field
        self.value = value
        self.message = message
        super().__init__(self._format_message())
        
    def _format_message(self) -> str:
        """Format the error message with context."""
        if self.field and self.value is not None:
            return f"Validation error for '{self.field}' (value={self.value}): {self.message}"
        elif self.field:
            return f"Validation error for '{self.field}': {self.message}"
        return f"Validation error: {self.message}"


class ConfigurationError(ValidationError):
    """Exception for configuration-related validation errors."""
    pass


class ProcessError(ValidationError):
    """Exception for process-related validation errors."""
    pass


class SimulationError(ValidationError):
    """Exception for simulation-related validation errors."""
    pass


class StateTransitionError(ValidationError):
    """Exception for invalid state transitions."""
    
    def __init__(self, current_state: Any, target_state: Any, entity: str = "Process"):
        self.current_state = current_state
        self.target_state = target_state
        message = f"Invalid state transition from {current_state} to {target_state}"
        super().__init__(message, field=f"{entity}.state", value=current_state)


# =============================================================================
# VALIDATION RESULT
# =============================================================================

@dataclass
class ValidationResult:
    """
    Result of a validation operation.
    
    Provides detailed information about validation success/failure.
    """
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    field: Optional[str] = None
    
    def __bool__(self) -> bool:
        """Allow using ValidationResult in boolean context."""
        return self.is_valid
    
    @staticmethod
    def success() -> 'ValidationResult':
        """Create a successful validation result."""
        return ValidationResult(is_valid=True, errors=[], warnings=[])
    
    @staticmethod
    def failure(error: str, field: str = None) -> 'ValidationResult':
        """Create a failed validation result."""
        return ValidationResult(
            is_valid=False,
            errors=[error],
            warnings=[],
            field=field
        )
    
    def add_error(self, error: str):
        """Add an error to the result."""
        self.errors.append(error)
        self.is_valid = False
        
    def add_warning(self, warning: str):
        """Add a warning to the result."""
        self.warnings.append(warning)
        
    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        """Merge another validation result into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.is_valid = self.is_valid and other.is_valid
        return self


# =============================================================================
# CONFIGURATION VALIDATORS
# =============================================================================

class ConfigValidator:
    """Validator for simulation configuration parameters."""
    
    # Absolute bounds for configuration values
    MIN_PROCESSORS = 1
    MAX_PROCESSORS = 64
    MIN_PROCESSES = 0
    MAX_PROCESSES = 10000
    MIN_BURST_TIME = 1
    MAX_BURST_TIME = 1000
    MIN_TIME_QUANTUM = 1
    MAX_TIME_QUANTUM = 100
    MIN_THRESHOLD = 0.0
    MAX_THRESHOLD = 1.0
    
    @classmethod
    def validate_config(cls, config: SimulationConfig) -> ValidationResult:
        """
        Validate entire simulation configuration.
        
        Args:
            config: SimulationConfig to validate
            
        Returns:
            ValidationResult with all errors and warnings
        """
        result = ValidationResult.success()
        
        # Validate each parameter
        result.merge(cls.validate_num_processors(config.num_processors))
        result.merge(cls.validate_num_processes(config.num_processes))
        result.merge(cls.validate_burst_time_range(
            config.min_burst_time, 
            config.max_burst_time
        ))
        result.merge(cls.validate_time_quantum(config.time_quantum))
        result.merge(cls.validate_threshold(config.load_threshold))
        
        # Cross-field validation
        if config.num_processes > 0 and config.num_processors > config.num_processes * 2:
            result.add_warning(
                f"More than twice as many processors ({config.num_processors}) as processes "
                f"({config.num_processes}) - some processors will be underutilized"
            )
            
        return result
    
    @classmethod
    def validate_num_processors(cls, value: int) -> ValidationResult:
        """Validate number of processors."""
        if not isinstance(value, int):
            return ValidationResult.failure(
                f"num_processors must be an integer, got {type(value).__name__}",
                "num_processors"
            )
        if value < cls.MIN_PROCESSORS:
            return ValidationResult.failure(
                f"num_processors must be at least {cls.MIN_PROCESSORS}, got {value}",
                "num_processors"
            )
        if value > cls.MAX_PROCESSORS:
            return ValidationResult.failure(
                f"num_processors must be at most {cls.MAX_PROCESSORS}, got {value}",
                "num_processors"
            )
        return ValidationResult.success()
    
    @classmethod
    def validate_num_processes(cls, value: int) -> ValidationResult:
        """Validate number of processes."""
        if not isinstance(value, int):
            return ValidationResult.failure(
                f"num_processes must be an integer, got {type(value).__name__}",
                "num_processes"
            )
        if value < cls.MIN_PROCESSES:
            return ValidationResult.failure(
                f"num_processes must be at least {cls.MIN_PROCESSES}, got {value}",
                "num_processes"
            )
        if value > cls.MAX_PROCESSES:
            return ValidationResult.failure(
                f"num_processes must be at most {cls.MAX_PROCESSES}, got {value}",
                "num_processes"
            )
        return ValidationResult.success()
    
    @classmethod
    def validate_burst_time_range(cls, min_burst: int, max_burst: int) -> ValidationResult:
        """Validate burst time range."""
        result = ValidationResult.success()
        
        if not isinstance(min_burst, int) or not isinstance(max_burst, int):
            result.add_error("Burst times must be integers")
            return result
            
        if min_burst < cls.MIN_BURST_TIME:
            result.add_error(f"min_burst_time must be at least {cls.MIN_BURST_TIME}")
        if max_burst > cls.MAX_BURST_TIME:
            result.add_error(f"max_burst_time must be at most {cls.MAX_BURST_TIME}")
        if min_burst > max_burst:
            result.add_error(f"min_burst_time ({min_burst}) cannot exceed max_burst_time ({max_burst})")
            
        return result
    
    @classmethod
    def validate_time_quantum(cls, value: int) -> ValidationResult:
        """Validate time quantum."""
        if not isinstance(value, int):
            return ValidationResult.failure(
                f"time_quantum must be an integer, got {type(value).__name__}",
                "time_quantum"
            )
        if value < cls.MIN_TIME_QUANTUM:
            return ValidationResult.failure(
                f"time_quantum must be at least {cls.MIN_TIME_QUANTUM}",
                "time_quantum"
            )
        if value > cls.MAX_TIME_QUANTUM:
            return ValidationResult.failure(
                f"time_quantum must be at most {cls.MAX_TIME_QUANTUM}",
                "time_quantum"
            )
        return ValidationResult.success()
    
    @classmethod
    def validate_threshold(cls, value: float) -> ValidationResult:
        """Validate load threshold."""
        if not isinstance(value, (int, float)):
            return ValidationResult.failure(
                f"load_threshold must be a number, got {type(value).__name__}",
                "load_threshold"
            )
        if value < cls.MIN_THRESHOLD or value > cls.MAX_THRESHOLD:
            return ValidationResult.failure(
                f"load_threshold must be between {cls.MIN_THRESHOLD} and {cls.MAX_THRESHOLD}",
                "load_threshold"
            )
        return ValidationResult.success()


# =============================================================================
# PROCESS VALIDATORS
# =============================================================================

class ProcessValidator:
    """Validator for process attributes and operations."""
    
    # Valid state transitions
    VALID_TRANSITIONS = {
        ProcessState.NEW: {ProcessState.READY},
        ProcessState.READY: {ProcessState.RUNNING, ProcessState.MIGRATING},
        ProcessState.RUNNING: {ProcessState.READY, ProcessState.COMPLETED, ProcessState.WAITING},
        ProcessState.WAITING: {ProcessState.READY},
        ProcessState.MIGRATING: {ProcessState.READY},
        ProcessState.COMPLETED: set()  # Terminal state
    }
    
    @classmethod
    def validate_state_transition(
        cls, 
        current: ProcessState, 
        target: ProcessState
    ) -> ValidationResult:
        """
        Validate that a state transition is allowed.
        
        Args:
            current: Current process state
            target: Target process state
            
        Returns:
            ValidationResult
        """
        valid_targets = cls.VALID_TRANSITIONS.get(current, set())
        
        if target not in valid_targets:
            return ValidationResult.failure(
                f"Invalid transition from {current.name} to {target.name}. "
                f"Valid transitions: {[s.name for s in valid_targets]}",
                "state"
            )
        return ValidationResult.success()
    
    @classmethod
    def validate_process_attributes(
        cls,
        pid: int,
        arrival_time: int,
        burst_time: int
    ) -> ValidationResult:
        """Validate process creation attributes."""
        result = ValidationResult.success()
        
        if not isinstance(pid, int) or pid < 0:
            result.add_error(f"pid must be a non-negative integer, got {pid}")
            
        if not isinstance(arrival_time, int) or arrival_time < 0:
            result.add_error(f"arrival_time must be a non-negative integer, got {arrival_time}")
            
        if not isinstance(burst_time, int) or burst_time <= 0:
            result.add_error(f"burst_time must be a positive integer, got {burst_time}")
            
        return result
    
    @classmethod
    def validate_execution_update(
        cls,
        remaining_time: int,
        execution_amount: int
    ) -> ValidationResult:
        """Validate execution update operation."""
        if execution_amount <= 0:
            return ValidationResult.failure(
                f"execution_amount must be positive, got {execution_amount}",
                "execution_amount"
            )
        if execution_amount > remaining_time:
            return ValidationResult.failure(
                f"Cannot execute {execution_amount} time units with only {remaining_time} remaining",
                "execution_amount"
            )
        return ValidationResult.success()


# =============================================================================
# SIMULATION VALIDATORS
# =============================================================================

class SimulationValidator:
    """Validator for simulation operations."""
    
    @classmethod
    def validate_simulation_start(
        cls,
        processor_manager,
        load_balancer,
        processes: List
    ) -> ValidationResult:
        """Validate that simulation is ready to start."""
        result = ValidationResult.success()
        
        if processor_manager is None:
            result.add_error("ProcessorManager not initialized")
            
        if load_balancer is None:
            result.add_error("LoadBalancer not initialized")
            
        if not processes:
            result.add_warning("No processes to simulate")
            
        return result
    
    @classmethod
    def validate_process_assignment(
        cls,
        process,
        processor,
        current_time: int
    ) -> ValidationResult:
        """Validate process assignment to processor."""
        result = ValidationResult.success()
        
        if process is None:
            result.add_error("Cannot assign None process")
            return result
            
        if processor is None:
            result.add_error("Cannot assign to None processor")
            return result
            
        if process.arrival_time > current_time:
            result.add_error(
                f"Process {process.pid} arrives at {process.arrival_time}, "
                f"but current time is {current_time}"
            )
            
        if process.state not in (ProcessState.NEW, ProcessState.READY, ProcessState.MIGRATING):
            result.add_error(
                f"Process {process.pid} is in state {process.state.name}, "
                f"cannot be assigned"
            )
            
        return result
    
    @classmethod
    def validate_migration(
        cls,
        process,
        source_processor,
        dest_processor
    ) -> ValidationResult:
        """Validate process migration."""
        result = ValidationResult.success()
        
        if process is None:
            result.add_error("Cannot migrate None process")
            return result
            
        if source_processor is None or dest_processor is None:
            result.add_error("Source and destination processors must be specified")
            return result
            
        if source_processor.processor_id == dest_processor.processor_id:
            result.add_error("Cannot migrate to the same processor")
            
        if process.state == ProcessState.RUNNING:
            result.add_warning(
                f"Migrating running process {process.pid} will cause context switch overhead"
            )
            
        if process.state == ProcessState.COMPLETED:
            result.add_error(f"Cannot migrate completed process {process.pid}")
            
        return result


# =============================================================================
# INPUT SANITIZERS
# =============================================================================

class InputSanitizer:
    """Sanitize and normalize input values."""
    
    @staticmethod
    def sanitize_int(
        value: Any, 
        min_val: int = None, 
        max_val: int = None,
        default: int = 0
    ) -> int:
        """
        Sanitize input to valid integer.
        
        Args:
            value: Input value to sanitize
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            default: Default value if conversion fails
            
        Returns:
            Sanitized integer value
        """
        try:
            result = int(value)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert {value} to int, using default {default}")
            return default
            
        if min_val is not None and result < min_val:
            logger.warning(f"Value {result} below minimum {min_val}, clamping")
            result = min_val
        if max_val is not None and result > max_val:
            logger.warning(f"Value {result} above maximum {max_val}, clamping")
            result = max_val
            
        return result
    
    @staticmethod
    def sanitize_float(
        value: Any,
        min_val: float = None,
        max_val: float = None,
        default: float = 0.0
    ) -> float:
        """Sanitize input to valid float."""
        try:
            result = float(value)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert {value} to float, using default {default}")
            return default
            
        if min_val is not None and result < min_val:
            result = min_val
        if max_val is not None and result > max_val:
            result = max_val
            
        return result
    
    @staticmethod
    def sanitize_string(
        value: Any,
        max_length: int = 255,
        default: str = ""
    ) -> str:
        """Sanitize input to valid string."""
        if value is None:
            return default
        result = str(value)
        if len(result) > max_length:
            logger.warning(f"String truncated from {len(result)} to {max_length} characters")
            result = result[:max_length]
        return result


# =============================================================================
# GUARD FUNCTIONS
# =============================================================================

def require_positive(value: int, name: str) -> int:
    """Guard that requires a positive integer."""
    if not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{name} must be a positive integer", name, value)
    return value


def require_non_negative(value: int, name: str) -> int:
    """Guard that requires a non-negative integer."""
    if not isinstance(value, int) or value < 0:
        raise ValidationError(f"{name} must be a non-negative integer", name, value)
    return value


def require_in_range(value: Union[int, float], min_val, max_val, name: str):
    """Guard that requires a value within a range."""
    if value < min_val or value > max_val:
        raise ValidationError(
            f"{name} must be between {min_val} and {max_val}",
            name,
            value
        )
    return value


def require_not_none(value: Any, name: str):
    """Guard that requires a non-None value."""
    if value is None:
        raise ValidationError(f"{name} cannot be None", name)
    return value


def require_state(obj: Any, valid_states: List, name: str = "object"):
    """Guard that requires object to be in a valid state."""
    current_state = getattr(obj, 'state', None)
    if current_state not in valid_states:
        raise StateTransitionError(
            current_state, 
            f"one of {[s.name for s in valid_states]}",
            name
        )
    return obj


# =============================================================================
# VALIDATION DECORATORS
# =============================================================================

def validate_config(func):
    """Decorator to validate configuration before function execution."""
    def wrapper(*args, **kwargs):
        # Look for config in args or kwargs
        config = kwargs.get('config') or (args[1] if len(args) > 1 else None)
        if config and isinstance(config, SimulationConfig):
            result = ConfigValidator.validate_config(config)
            if not result.is_valid:
                raise ConfigurationError("; ".join(result.errors))
            for warning in result.warnings:
                logger.warning(warning)
        return func(*args, **kwargs)
    return wrapper


def validate_positive_args(*arg_names):
    """Decorator to validate that specified arguments are positive."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            for name in arg_names:
                if name in kwargs:
                    require_positive(kwargs[name], name)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# =============================================================================
# SAFE OPERATION WRAPPERS
# =============================================================================

def safe_execute(func, *args, default=None, log_errors=True, **kwargs):
    """
    Safely execute a function, returning default on error.
    
    Args:
        func: Function to execute
        *args: Positional arguments
        default: Value to return on error
        log_errors: Whether to log errors
        **kwargs: Keyword arguments
        
    Returns:
        Function result or default value
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger.error(f"Error in {func.__name__}: {e}")
        return default


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers.
    
    Args:
        numerator: Dividend
        denominator: Divisor
        default: Value to return if division fails
        
    Returns:
        Result of division or default
    """
    try:
        if denominator == 0:
            return default
        return numerator / denominator
    except (TypeError, ZeroDivisionError):
        return default


# =============================================================================
# LOGGING UTILITIES
# =============================================================================

def log_validation_result(result: ValidationResult, context: str = ""):
    """Log validation result with appropriate level."""
    prefix = f"[{context}] " if context else ""
    
    if result.is_valid:
        if result.warnings:
            for warning in result.warnings:
                logger.warning(f"{prefix}{warning}")
        logger.debug(f"{prefix}Validation passed")
    else:
        for error in result.errors:
            logger.error(f"{prefix}Validation error: {error}")
        for warning in result.warnings:
            logger.warning(f"{prefix}Validation warning: {warning}")
