"""
Logging configuration utilities for Semantrix.

This module provides pre-configured logging setups for different environments
and use cases.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional

from .logging import initialize_logging, ThreadSafeLogManager


class LoggingPresets:
    """Pre-configured logging setups for different environments."""
    
    @staticmethod
    def development(
        log_file: Optional[str] = None,
        enable_async: bool = False
    ) -> ThreadSafeLogManager:
        """
        Development environment logging configuration.
        
        Args:
            log_file: Optional log file path
            enable_async: Whether to enable async handlers
            
        Returns:
            Configured log manager
        """
        return initialize_logging(
            log_level="DEBUG",
            log_format="json",
            log_file=log_file or "logs/semantrix_dev.log",
            max_bytes=5 * 1024 * 1024,  # 5MB
            backup_count=3,
            include_correlation_id=True,
            enable_async_handler=enable_async
        )
    
    @staticmethod
    def production(
        log_file: Optional[str] = None,
        enable_async: bool = True
    ) -> ThreadSafeLogManager:
        """
        Production environment logging configuration.
        
        Args:
            log_file: Optional log file path
            enable_async: Whether to enable async handlers
            
        Returns:
            Configured log manager
        """
        return initialize_logging(
            log_level="INFO",
            log_format="json",
            log_file=log_file or "logs/semantrix_prod.log",
            max_bytes=50 * 1024 * 1024,  # 50MB
            backup_count=10,
            include_correlation_id=True,
            enable_async_handler=enable_async
        )
    
    @staticmethod
    def testing(
        log_file: Optional[str] = None,
        enable_async: bool = False
    ) -> ThreadSafeLogManager:
        """
        Testing environment logging configuration.
        
        Args:
            log_file: Optional log file path
            enable_async: Whether to enable async handlers
            
        Returns:
            Configured log manager
        """
        return initialize_logging(
            log_level="WARNING",
            log_format="text",
            log_file=log_file or "logs/semantrix_test.log",
            max_bytes=1 * 1024 * 1024,  # 1MB
            backup_count=1,
            include_correlation_id=False,
            enable_async_handler=enable_async
        )
    
    @staticmethod
    def docker(
        log_file: Optional[str] = None,
        enable_async: bool = True
    ) -> ThreadSafeLogManager:
        """
        Docker container logging configuration.
        
        Args:
            log_file: Optional log file path
            enable_async: Whether to enable async handlers
            
        Returns:
            Configured log manager
        """
        return initialize_logging(
            log_level="INFO",
            log_format="json",
            log_file=log_file or "/var/log/semantrix/app.log",
            max_bytes=100 * 1024 * 1024,  # 100MB
            backup_count=5,
            include_correlation_id=True,
            enable_async_handler=enable_async
        )


def configure_from_environment() -> ThreadSafeLogManager:
    """
    Configure logging based on environment variables.
    
    Environment variables:
    - SEMANTRIX_LOG_LEVEL: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    - SEMANTRIX_LOG_FORMAT: Log format (json, text)
    - SEMANTRIX_LOG_FILE: Log file path
    - SEMANTRIX_LOG_MAX_BYTES: Max file size in bytes
    - SEMANTRIX_LOG_BACKUP_COUNT: Number of backup files
    - SEMANTRIX_LOG_ENABLE_ASYNC: Enable async handlers (true/false)
    - SEMANTRIX_LOG_INCLUDE_CORRELATION_ID: Include correlation IDs (true/false)
    
    Returns:
        Configured log manager
    """
    # Get environment variables with defaults
    log_level = os.getenv("SEMANTRIX_LOG_LEVEL", "INFO")
    log_format = os.getenv("SEMANTRIX_LOG_FORMAT", "json")
    log_file = os.getenv("SEMANTRIX_LOG_FILE")
    max_bytes = int(os.getenv("SEMANTRIX_LOG_MAX_BYTES", "10485760"))  # 10MB default
    backup_count = int(os.getenv("SEMANTRIX_LOG_BACKUP_COUNT", "5"))
    enable_async = os.getenv("SEMANTRIX_LOG_ENABLE_ASYNC", "false").lower() == "true"
    include_correlation_id = os.getenv("SEMANTRIX_LOG_INCLUDE_CORRELATION_ID", "true").lower() == "true"
    
    return initialize_logging(
        log_level=log_level,
        log_format=log_format,
        log_file=log_file,
        max_bytes=max_bytes,
        backup_count=backup_count,
        include_correlation_id=include_correlation_id,
        enable_async_handler=enable_async
    )


def configure_for_service(
    service_name: str,
    environment: str = "development",
    log_dir: Optional[str] = None,
    **kwargs
) -> ThreadSafeLogManager:
    """
    Configure logging for a specific service.
    
    Args:
        service_name: Name of the service
        environment: Environment (development, production, testing)
        log_dir: Log directory path
        **kwargs: Additional configuration options
        
    Returns:
        Configured log manager
    """
    # Determine log file path
    if log_dir is None:
        log_dir = os.getenv("SEMANTRIX_LOG_DIR", "logs")
    
    log_file = Path(log_dir) / f"{service_name}_{environment}.log"
    
    # Use appropriate preset based on environment
    if environment == "production":
        return LoggingPresets.production(str(log_file), **kwargs)
    elif environment == "testing":
        return LoggingPresets.testing(str(log_file), **kwargs)
    elif environment == "docker":
        return LoggingPresets.docker(str(log_file), **kwargs)
    else:
        return LoggingPresets.development(str(log_file), **kwargs)


def get_logging_config() -> Dict[str, Any]:
    """
    Get current logging configuration.
    
    Returns:
        Dictionary with current logging configuration
    """
    from .logging import _log_manager
    
    if _log_manager is None:
        return {"status": "not_initialized"}
    
    return {
        "status": "initialized",
        "log_level": _log_manager.log_level,
        "log_format": _log_manager.log_format,
        "log_file": _log_manager.log_file,
        "max_bytes": _log_manager.max_bytes,
        "backup_count": _log_manager.backup_count,
        "include_correlation_id": _log_manager.include_correlation_id,
        "enable_async_handler": _log_manager.enable_async_handler,
        "initialized": _log_manager._initialized
    }
