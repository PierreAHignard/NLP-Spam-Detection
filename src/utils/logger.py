"""
Professional logging system for Air Quality ML Pipeline.

This module provides a modern, hierarchical logging system with multiple verbosity levels,
professional formatting, and centralized configuration.
"""

import sys
import time
from datetime import datetime
from enum import IntEnum
from typing import Optional, Any, Dict
from contextlib import contextmanager


class LogLevel(IntEnum):
    """Logging levels enumeration."""
    SILENT = 0
    NORMAL = 1
    VERBOSE = 2


class PipelineLogger:
    """
    Professional logger for ML pipeline operations.
    
    Features:
    - 3 verbosity levels (SILENT, NORMAL, VERBOSE)
    - Hierarchical indentation
    - Professional formatting with icons and timestamps
    - Context management for nested operations
    - Performance timing
    """
    
    def __init__(self, level: LogLevel = LogLevel.NORMAL):
        """
        Initialize the logger.
        
        Args:
            level: Logging level (SILENT, NORMAL, VERBOSE)
        """
        self.level = level
        self._indent_level = 0
        self._start_times = {}
        
        # Icons for different message types
        self.icons = {
            'start': '🚀',
            'step': '📊',
            'substep': '├──',
            'success': '✅',
            'warning': '⚠️',
            'error': '❌',
            'info': 'ℹ️ ',
            'feature': '🔧',
            'model': '🎯',
            'data': '📈',
            'save': '💾',
            'load': '📂',
            'time': '⏱️'
        }
    
    def set_level(self, level: LogLevel):
        """Set the logging level."""
        self.level = level
    
    def _should_log(self, min_level: LogLevel) -> bool:
        """Check if message should be logged based on current level."""
        return self.level >= min_level
    
    def _get_timestamp(self) -> str:
        """Get formatted timestamp."""
        return datetime.now().strftime("%H:%M:%S")
    
    def _get_indent(self) -> str:
        """Get current indentation string."""
        if self._indent_level == 0:
            return ""
        elif self._indent_level == 1:
            return "├── "
        else:
            return "│   " * (self._indent_level - 1) + "├── "
    
    def _print(self, message: str, icon: str = "", timestamp: bool = True):
        """Internal print method with formatting."""
        parts = []
        
        if icon:
            parts.append(icon)
        
        if timestamp and self.level >= LogLevel.NORMAL:
            parts.append(f"[{self._get_timestamp()}]")
        
        indent = self._get_indent()
        if indent:
            parts.insert(-1 if timestamp else 0, indent.rstrip())
        
        if parts:
            formatted_message = " ".join(parts) + " " + message
        else:
            formatted_message = message
        
        print(formatted_message)
    
    def header(self, title: str, level: LogLevel = LogLevel.NORMAL):
        """
        Print a main header for major pipeline sections.
        
        Args:
            title: Header title
            level: Minimum log level required
        """
        if not self._should_log(level):
            return
        
        if level == LogLevel.NORMAL:
            print(f"\n{self.icons['start']} [{self._get_timestamp()}] {title.upper()}")
        else:
            separator = "=" * 60
            print(f"\n{separator}")
            print(f"{self.icons['start']} {title.upper()}")
            print(separator)
    
    def step(self, message: str, step_num: Optional[int] = None, level: LogLevel = LogLevel.NORMAL):
        """
        Print a main step message.
        
        Args:
            message: Step message
            step_num: Optional step number
            level: Minimum log level required
        """
        if not self._should_log(level):
            return
        
        if step_num:
            formatted_message = f"STEP {step_num}: {message}"
        else:
            formatted_message = message
        
        self._print(formatted_message, self.icons['step'])
    
    def substep(self, message: str, level: LogLevel = LogLevel.VERBOSE):
        """
        Print a substep message (verbose only by default).
        
        Args:
            message: Substep message
            level: Minimum log level required
        """
        if not self._should_log(level):
            return
        
        self._print(message, timestamp=False)
    
    def success(self, message: str, level: LogLevel = LogLevel.NORMAL):
        """Print a success message."""
        if not self._should_log(level):
            return
        
        self._print(message, self.icons['success'], timestamp=False)
    
    def warning(self, message: str, level: LogLevel = LogLevel.NORMAL):
        """Print a warning message."""
        if not self._should_log(level):
            return
        
        self._print(message, self.icons['warning'], timestamp=False)
    
    def error(self, message: str, level: LogLevel = LogLevel.NORMAL):
        """Print an error message."""
        if not self._should_log(level):
            return
        
        self._print(message, self.icons['error'], timestamp=False)
    
    def info(self, message: str, level: LogLevel = LogLevel.VERBOSE):
        """Print an info message (verbose by default)."""
        if not self._should_log(level):
            return
        
        self._print(message, self.icons['info'], timestamp=False)
    
    def data_info(self, message: str, level: LogLevel = LogLevel.NORMAL):
        """Print data-related information."""
        if not self._should_log(level):
            return
        
        self._print(message, self.icons['data'], timestamp=False)
    
    def model_info(self, message: str, level: LogLevel = LogLevel.NORMAL):
        """Print model-related information."""
        if not self._should_log(level):
            return
        
        self._print(message, self.icons['model'], timestamp=False)
    
    def feature_info(self, message: str, level: LogLevel = LogLevel.NORMAL):
        """Print feature engineering information."""
        if not self._should_log(level):
            return
        
        self._print(message, self.icons['feature'], timestamp=False)
    
    def results_summary(self, results: Dict[str, Any], level: LogLevel = LogLevel.NORMAL):
        """
        Print a formatted results summary.
        
        Args:
            results: Dictionary of results to display
            level: Minimum log level required
        """
        if not self._should_log(level):
            return
        
        print(f"\n{self.icons['model']} [{self._get_timestamp()}] RESULTS SUMMARY")
        
        with self.indent():
            for key, value in results.items():
                if isinstance(value, float):
                    self._print(f"{key}: {value:.3f}", timestamp=False)
                else:
                    self._print(f"{key}: {value}", timestamp=False)
    
    def dataframe_info(self, df, name: str = "DataFrame", level: LogLevel = LogLevel.NORMAL):
        """
        Print DataFrame information in a professional format.
        
        Args:
            df: pandas DataFrame
            name: Name to display for the DataFrame
            level: Minimum log level required
        """
        if not self._should_log(level):
            return
        
        self.data_info(f"{name} Information:")
        
        with self.indent():
            self.substep(f"Shape: {df.shape}", LogLevel.NORMAL)
            self.substep(f"Columns: {list(df.columns)}", LogLevel.VERBOSE)
            
            missing_count = df.isnull().sum().sum()
            if missing_count > 0:
                self.substep(f"Missing values: {missing_count}", LogLevel.NORMAL)
            else:
                self.substep("Missing values: None", LogLevel.VERBOSE)
            
            if 'pm2_5' in df.columns:
                pm25_range = f"[{df['pm2_5'].min():.1f}, {df['pm2_5'].max():.1f}]"
                self.substep(f"PM2.5 range: {pm25_range}", LogLevel.NORMAL)
    
    @contextmanager
    def indent(self):
        """Context manager for indented logging."""
        self._indent_level += 1
        try:
            yield
        finally:
            self._indent_level -= 1
    
    @contextmanager
    def timer(self, operation_name: str, level: LogLevel = LogLevel.NORMAL):
        """
        Context manager for timing operations.
        
        Args:
            operation_name: Name of the operation being timed
            level: Minimum log level required
        """
        if not self._should_log(level):
            yield
            return
        
        start_time = time.time()
        self.substep(f"Starting {operation_name}...", level)
        
        try:
            yield
        finally:
            elapsed = time.time() - start_time
            elapsed_str = self._format_time_elapsed(elapsed)
            self.success(f"{operation_name} completed in {elapsed_str}", level)
    
    def _format_time_elapsed(self, elapsed: float) -> str:
        """Format elapsed time in a readable way."""
        if elapsed < 60:
            return f"{elapsed:.1f}s"
        else:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            return f"{minutes}m {seconds:.1f}s"
    
    def pipeline_complete(self, total_time: Optional[float] = None):
        """Print pipeline completion message."""
        if not self._should_log(LogLevel.NORMAL):
            return
        
        message = "Pipeline completed successfully!"
        if total_time:
            time_str = self._format_time_elapsed(total_time)
            message += f" (Total time: {time_str})"
        
        print(f"\n🎉 [{self._get_timestamp()}] {message}")


# Global logger instance
_logger = PipelineLogger()


def get_logger() -> PipelineLogger:
    """Get the global logger instance."""
    return _logger


def set_log_level(level: LogLevel):
    """Set the global logging level."""
    _logger.set_level(level)


def log_level_from_string(level_str: str) -> LogLevel:
    """Convert string to LogLevel enum."""
    level_map = {
        'silent': LogLevel.SILENT,
        'normal': LogLevel.NORMAL,
        'verbose': LogLevel.VERBOSE
    }
    
    level_str = level_str.lower()
    if level_str not in level_map:
        raise ValueError(f"Invalid log level: {level_str}. Available: {list(level_map.keys())}")
    
    return level_map[level_str]


# Convenience functions for backward compatibility
def print_step_header(step_number: int, step_name: str):
    """Backward compatibility function."""
    _logger.step(step_name, step_number)


def print_func_header(step_name: str):
    """Backward compatibility function."""
    _logger.substep(step_name)


def print_results_summary(results_dict: Dict[str, Any]):
    """Backward compatibility function."""
    _logger.results_summary(results_dict)


def print_dataframe_info(df, name: str = "DataFrame"):
    """Backward compatibility function."""
    _logger.dataframe_info(df, name)
