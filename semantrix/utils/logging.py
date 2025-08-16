"""
Structured logging system for Semantrix.

This module provides a comprehensive logging system with structured logging,
correlation IDs for tracing, and configurable log rotation and retention.
"""
import json
import logging
import logging.handlers
import os
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

# Context variable for correlation ID
correlation_id: ContextVar[Optional[str]] = ContextVar('correlation_id', default=None)


class StructuredFormatter(logging.Formatter):
    """
    Structured JSON formatter for logs.
    
    This formatter outputs logs in JSON format with consistent structure
    including timestamp, level, correlation ID, and structured data.
    """
    
    def __init__(self, include_correlation_id: bool = True):
        """
        Initialize the structured formatter.
        
        Args:
            include_correlation_id: Whether to include correlation ID in logs
        """
        super().__init__()
        self.include_correlation_id = include_correlation_id
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as structured JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON formatted log string
        """
        # Base log structure
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add correlation ID if available and enabled
        if self.include_correlation_id:
            current_correlation_id = correlation_id.get()
            if current_correlation_id:
                log_entry["correlation_id"] = current_correlation_id
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add extra fields if present
        if hasattr(record, 'extra_fields'):
            log_entry.update(record.extra_fields)
        
        # Add any additional attributes
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 
                          'exc_text', 'stack_info', 'extra_fields']:
                if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False)


class CorrelationIdFilter(logging.Filter):
    """
    Log filter that adds correlation ID to log records.
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """
        Add correlation ID to log record.
        
        Args:
            record: Log record to process
            
        Returns:
            True to include the record, False to exclude
        """
        current_correlation_id = correlation_id.get()
        if current_correlation_id:
            record.correlation_id = current_correlation_id
        return True


class MetricsLogger:
    """
    Logger for metrics and performance data.
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize metrics logger.
        
        Args:
            logger: Base logger instance
        """
        self.logger = logger
    
    def log_operation_start(self, operation: str, **kwargs):
        """
        Log the start of an operation.
        
        Args:
            operation: Name of the operation
            **kwargs: Additional operation metadata
        """
        self.logger.info(
            "Operation started",
            extra={
                'extra_fields': {
                    'event_type': 'operation_start',
                    'operation': operation,
                    **kwargs
                }
            }
        )
    
    def log_operation_end(self, operation: str, duration: float, success: bool = True, **kwargs):
        """
        Log the end of an operation.
        
        Args:
            operation: Name of the operation
            duration: Duration in seconds
            success: Whether the operation was successful
            **kwargs: Additional operation metadata
        """
        self.logger.info(
            "Operation completed",
            extra={
                'extra_fields': {
                    'event_type': 'operation_end',
                    'operation': operation,
                    'duration_seconds': duration,
                    'success': success,
                    **kwargs
                }
            }
        )
    
    def log_cache_hit(self, cache_type: str, key: str, **kwargs):
        """
        Log a cache hit.
        
        Args:
            cache_type: Type of cache (e.g., 'memory', 'redis')
            key: Cache key
            **kwargs: Additional metadata
        """
        self.logger.debug(
            "Cache hit",
            extra={
                'extra_fields': {
                    'event_type': 'cache_hit',
                    'cache_type': cache_type,
                    'cache_key': key,
                    **kwargs
                }
            }
        )
    
    def log_cache_miss(self, cache_type: str, key: str, **kwargs):
        """
        Log a cache miss.
        
        Args:
            cache_type: Type of cache (e.g., 'memory', 'redis')
            key: Cache key
            **kwargs: Additional metadata
        """
        self.logger.debug(
            "Cache miss",
            extra={
                'extra_fields': {
                    'event_type': 'cache_miss',
                    'cache_type': cache_type,
                    'cache_key': key,
                    **kwargs
                }
            }
        )
    
    def log_error_rate(self, operation: str, error_count: int, total_count: int, **kwargs):
        """
        Log error rate for an operation.
        
        Args:
            operation: Name of the operation
            error_count: Number of errors
            total_count: Total number of operations
            **kwargs: Additional metadata
        """
        error_rate = error_count / total_count if total_count > 0 else 0
        self.logger.warning(
            "Error rate threshold exceeded",
            extra={
                'extra_fields': {
                    'event_type': 'error_rate',
                    'operation': operation,
                    'error_count': error_count,
                    'total_count': total_count,
                    'error_rate': error_rate,
                    **kwargs
                }
            }
        )


class LogManager:
    """
    Centralized log manager for Semantrix.
    
    This class manages loggers, handlers, and configuration for the entire
    application with support for structured logging and correlation IDs.
    """
    
    def __init__(self, 
                 log_level: str = "INFO",
                 log_format: str = "json",
                 log_file: Optional[str] = None,
                 max_bytes: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5,
                 include_correlation_id: bool = True):
        """
        Initialize the log manager.
        
        Args:
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_format: Log format ('json' or 'text')
            log_file: Path to log file (optional)
            max_bytes: Maximum size of log file before rotation
            backup_count: Number of backup files to keep
            include_correlation_id: Whether to include correlation IDs
        """
        self.log_level = getattr(logging, log_level.upper())
        self.log_format = log_format
        self.log_file = log_file
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.include_correlation_id = include_correlation_id
        
        # Configure root logger
        self._configure_root_logger()
        
        # Create main logger
        self.logger = logging.getLogger("semantrix")
        self.logger.setLevel(self.log_level)
        
        # Create metrics logger
        self.metrics = MetricsLogger(self.logger)
    
    def _configure_root_logger(self):
        """Configure the root logger with handlers."""
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Create formatter
        if self.log_format == "json":
            formatter = StructuredFormatter(self.include_correlation_id)
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        
        if self.include_correlation_id:
            console_handler.addFilter(CorrelationIdFilter())
        
        root_logger.addHandler(console_handler)
        
        # File handler (if specified)
        if self.log_file:
            # Ensure log directory exists
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                self.log_file,
                maxBytes=self.max_bytes,
                backupCount=self.backup_count
            )
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            
            if self.include_correlation_id:
                file_handler.addFilter(CorrelationIdFilter())
            
            root_logger.addHandler(file_handler)
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a logger with the specified name.
        
        Args:
            name: Logger name
            
        Returns:
            Configured logger instance
        """
        return logging.getLogger(name)
    
    def set_correlation_id(self, correlation_id_value: str):
        """
        Set the correlation ID for the current context.
        
        Args:
            correlation_id_value: Correlation ID value
        """
        correlation_id.set(correlation_id_value)
    
    def get_correlation_id(self) -> Optional[str]:
        """
        Get the current correlation ID.
        
        Returns:
            Current correlation ID or None
        """
        return correlation_id.get()
    
    def clear_correlation_id(self):
        """Clear the current correlation ID."""
        correlation_id.set(None)


# Global log manager instance
_log_manager: Optional[LogManager] = None


def initialize_logging(
    log_level: str = "INFO",
    log_format: str = "json",
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    include_correlation_id: bool = True
) -> LogManager:
    """
    Initialize the global logging system.
    
    Args:
        log_level: Logging level
        log_format: Log format ('json' or 'text')
        log_file: Path to log file
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        include_correlation_id: Whether to include correlation IDs
        
    Returns:
        Configured log manager instance
    """
    global _log_manager
    
    if _log_manager is None:
        _log_manager = LogManager(
            log_level=log_level,
            log_format=log_format,
            log_file=log_file,
            max_bytes=max_bytes,
            backup_count=backup_count,
            include_correlation_id=include_correlation_id
        )
    
    return _log_manager


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    if _log_manager is None:
        # Initialize with defaults if not already initialized
        initialize_logging()
    
    return _log_manager.get_logger(name)


def get_metrics_logger() -> MetricsLogger:
    """
    Get the metrics logger.
    
    Returns:
        Metrics logger instance
    """
    if _log_manager is None:
        initialize_logging()
    
    return _log_manager.metrics


def set_correlation_id(correlation_id_value: str):
    """
    Set the correlation ID for the current context.
    
    Args:
        correlation_id_value: Correlation ID value
    """
    if _log_manager is None:
        initialize_logging()
    
    _log_manager.set_correlation_id(correlation_id_value)


def get_correlation_id() -> Optional[str]:
    """
    Get the current correlation ID.
    
    Returns:
        Current correlation ID or None
    """
    if _log_manager is None:
        return None
    
    return _log_manager.get_correlation_id()


def clear_correlation_id():
    """Clear the current correlation ID."""
    if _log_manager is not None:
        _log_manager.clear_correlation_id()


class CorrelationIdContext:
    """
    Context manager for correlation ID management.
    """
    
    def __init__(self, correlation_id_value: Optional[str] = None):
        """
        Initialize correlation ID context.
        
        Args:
            correlation_id_value: Correlation ID value (auto-generated if None)
        """
        self.correlation_id_value = correlation_id_value or str(uuid.uuid4())
        self.previous_correlation_id: Optional[str] = None
    
    def __enter__(self):
        """Enter the correlation ID context."""
        self.previous_correlation_id = get_correlation_id()
        set_correlation_id(self.correlation_id_value)
        return self.correlation_id_value
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the correlation ID context."""
        if self.previous_correlation_id is not None:
            set_correlation_id(self.previous_correlation_id)
        else:
            clear_correlation_id()


def with_correlation_id(correlation_id_value: Optional[str] = None):
    """
    Create a correlation ID context.
    
    Args:
        correlation_id_value: Correlation ID value (auto-generated if None)
        
    Returns:
        CorrelationIdContext instance
    """
    return CorrelationIdContext(correlation_id_value)
