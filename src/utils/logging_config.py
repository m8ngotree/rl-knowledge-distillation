"""
Logging configuration for RL Knowledge Distillation
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        # Add color to levelname
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"
        
        return super().format(record)


def setup_logging(
    name: str = "rl_distillation",
    level: str = "INFO",
    log_dir: Optional[Path] = None,
    console_output: bool = True,
    file_output: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup comprehensive logging configuration
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (None for ./logs)
        console_output: Whether to output to console
        file_output: Whether to output to files
        max_bytes: Maximum bytes per log file before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    
    # Input validation
    if not isinstance(name, str):
        raise TypeError(f"name must be a string, got {type(name)}")
    if level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
        raise ValueError(f"Invalid logging level: {level}")
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s | %(name)s | %(levelname)s | %(module)s:%(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = ColoredFormatter(
        fmt='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    # File handlers
    if file_output:
        if log_dir is None:
            log_dir = Path("./logs")
        
        log_dir = Path(log_dir)
        log_dir.mkdir(exist_ok=True)
        
        # Main log file (rotating)
        main_log_file = log_dir / f"{name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)
        
        # Error log file (errors only)
        error_log_file = log_dir / f"{name}_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        logger.addHandler(error_handler)
    
    # Prevent duplicate logs from propagating to root logger
    logger.propagate = False
    
    return logger


def setup_experiment_logging(experiment_name: str, save_dir: Path) -> logging.Logger:
    """
    Setup logging specifically for experiments
    
    Args:
        experiment_name: Name of the experiment
        save_dir: Directory to save experiment logs
        
    Returns:
        Configured experiment logger
    """
    
    # Create experiment-specific log directory
    log_dir = save_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Setup logger with experiment name
    logger_name = f"experiment_{experiment_name}"
    logger = setup_logging(
        name=logger_name,
        level="DEBUG",
        log_dir=log_dir,
        console_output=True,
        file_output=True
    )
    
    # Log experiment start
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with default configuration
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    
    # Check if logger already exists and is configured
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        # Setup default configuration if not already configured
        setup_logging(name=name, level="INFO")
        logger = logging.getLogger(name)
    
    return logger


# Module-level logger for this logging module
_logger = logging.getLogger(__name__)


class LoggingMixin:
    """Mixin class to add logging capabilities to any class"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = get_logger(self.__class__.__module__)
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        return self._logger
    
    def log_info(self, message: str, **kwargs) -> None:
        """Log info message with context"""
        self._logger.info(f"[{self.__class__.__name__}] {message}", **kwargs)
    
    def log_warning(self, message: str, **kwargs) -> None:
        """Log warning message with context"""
        self._logger.warning(f"[{self.__class__.__name__}] {message}", **kwargs)
    
    def log_error(self, message: str, **kwargs) -> None:
        """Log error message with context"""
        self._logger.error(f"[{self.__class__.__name__}] {message}", **kwargs)
    
    def log_debug(self, message: str, **kwargs) -> None:
        """Log debug message with context"""
        self._logger.debug(f"[{self.__class__.__name__}] {message}", **kwargs)


# Performance logging utilities
class PerformanceLogger:
    """Context manager for logging performance metrics"""
    
    def __init__(self, logger: logging.Logger, operation: str, level: str = "INFO"):
        self.logger = logger
        self.operation = operation
        self.level = getattr(logging, level)
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        self.logger.log(self.level, f"Starting {self.operation}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        duration = time.time() - self.start_time
        
        if exc_type is None:
            self.logger.log(self.level, f"Completed {self.operation} in {duration:.2f}s")
        else:
            self.logger.error(f"Failed {self.operation} after {duration:.2f}s: {exc_val}")


def log_performance(operation: str, level: str = "INFO"):
    """Decorator for logging function performance"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            with PerformanceLogger(logger, f"{func.__name__}({operation})", level):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# GPU memory logging utility
def log_gpu_memory(logger: logging.Logger, context: str = "") -> None:
    """Log current GPU memory usage"""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
            max_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # GB
            
            logger.info(
                f"GPU Memory {context}: "
                f"Allocated={allocated:.2f}GB, "
                f"Reserved={reserved:.2f}GB, "
                f"Max={max_allocated:.2f}GB"
            )
        else:
            logger.debug("CUDA not available, skipping GPU memory logging")
    except ImportError:
        logger.debug("PyTorch not available, skipping GPU memory logging")
    except Exception as e:
        logger.warning(f"Failed to log GPU memory: {e}")


# Initialize default logging
setup_logging()