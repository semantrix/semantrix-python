"""
Tests for the improved Semantrix logging system.
"""

import asyncio
import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from semantrix.utils.logging import (
    initialize_logging,
    get_logger,
    get_adapter,
    with_correlation_id,
    get_metrics_logger,
    set_correlation_id,
    get_correlation_id,
    clear_correlation_id
)
from semantrix.utils.logging_config import (
    LoggingPresets,
    configure_from_environment,
    get_logging_config
)


class TestLoggingSystem:
    """Test the logging system functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = Path(self.temp_dir) / "test.log"
    
    def teardown_method(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_basic_logging_initialization(self):
        """Test basic logging initialization."""
        # Initialize logging
        log_manager = initialize_logging(
            log_level="DEBUG",
            log_format="json",
            log_file=str(self.log_file),
            include_correlation_id=True
        )
        
        assert log_manager is not None
        assert log_manager.log_level == 10  # DEBUG level
        
        # Get a logger and log a message
        logger = get_logger("test.basic")
        logger.info("Test message", extra={"test_field": "test_value"})
        
        # Check that log file was created and contains the message
        assert self.log_file.exists()
        
        with open(self.log_file, 'r') as f:
            log_line = f.readline().strip()
            log_data = json.loads(log_line)
            
            assert log_data["level"] == "INFO"
            assert log_data["logger"] == "test.basic"
            assert log_data["message"] == "Test message"
            assert log_data["test_field"] == "test_value"
    
    def test_correlation_id_functionality(self):
        """Test correlation ID functionality."""
        initialize_logging(log_level="DEBUG", log_format="json")
        
        # Test setting and getting correlation ID
        test_id = "test-correlation-123"
        set_correlation_id(test_id)
        assert get_correlation_id() == test_id
        
        # Test clearing correlation ID
        clear_correlation_id()
        assert get_correlation_id() is None
    
    def test_correlation_id_context_manager(self):
        """Test correlation ID context manager."""
        initialize_logging(log_level="DEBUG", log_format="json")
        
        # Set initial correlation ID
        set_correlation_id("initial-id")
        
        # Use context manager
        with with_correlation_id("context-id") as correlation_id:
            assert correlation_id == "context-id"
            assert get_correlation_id() == "context-id"
        
        # Check that original ID is restored
        assert get_correlation_id() == "initial-id"
    
    def test_logger_adapter(self):
        """Test logger adapter functionality."""
        initialize_logging(log_level="DEBUG", log_format="json")
        
        # Create adapter with correlation ID and extra fields
        adapter = get_adapter(
            "test.adapter",
            correlation_id="adapter-id",
            service="test_service",
            version="1.0.0"
        )
        
        # Log a message
        adapter.info("Adapter test message")
        
        # Test binding additional context
        user_adapter = adapter.bind(user_id="user-123")
        user_adapter.info("User message")
    
    def test_metrics_logger(self):
        """Test metrics logger functionality."""
        initialize_logging(log_level="DEBUG", log_format="json")
        
        metrics_logger = get_metrics_logger()
        
        # Test operation logging
        metrics_logger.log_operation_start("test_operation", operation_type="test")
        metrics_logger.log_operation_end("test_operation", 0.5, True, result_count=10)
        
        # Test cache logging
        metrics_logger.log_cache_hit("memory", "test_key", hit_type="exact")
        metrics_logger.log_cache_miss("redis", "missing_key", miss_reason="not_found")
        
        # Test error rate logging
        metrics_logger.log_error_rate("test_operation", 5, 100, threshold=0.05)
    
    def test_logging_presets(self):
        """Test logging presets."""
        # Test development preset
        dev_manager = LoggingPresets.development(enable_async=False)
        assert dev_manager.log_level == 10  # DEBUG
        assert dev_manager.log_format == "json"
        
        # Test production preset
        prod_manager = LoggingPresets.production(enable_async=False)
        assert prod_manager.log_level == 20  # INFO
        assert prod_manager.log_format == "json"
        
        # Test testing preset
        test_manager = LoggingPresets.testing(enable_async=False)
        assert test_manager.log_level == 30  # WARNING
        assert test_manager.log_format == "text"
    
    def test_environment_configuration(self):
        """Test environment-based configuration."""
        import os
        
        # Set environment variables
        os.environ["SEMANTRIX_LOG_LEVEL"] = "ERROR"
        os.environ["SEMANTRIX_LOG_FORMAT"] = "text"
        os.environ["SEMANTRIX_LOG_ENABLE_ASYNC"] = "false"
        
        # Configure from environment
        log_manager = configure_from_environment()
        
        assert log_manager.log_level == 40  # ERROR
        assert log_manager.log_format == "text"
        assert not log_manager.enable_async_handler
    
    def test_thread_safety(self):
        """Test thread safety of logging system."""
        initialize_logging(log_level="DEBUG", log_format="json")
        
        results = []
        errors = []
        
        def worker(thread_id):
            try:
                logger = get_logger(f"test.thread.{thread_id}")
                set_correlation_id(f"thread-{thread_id}")
                
                for i in range(10):
                    logger.info(f"Message {i} from thread {thread_id}")
                    time.sleep(0.001)  # Small delay
                
                results.append(f"thread-{thread_id}-completed")
            except Exception as e:
                errors.append(f"thread-{thread_id}-error: {e}")
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check that all threads completed successfully
        assert len(results) == 5
        assert len(errors) == 0
    
    def test_async_logging(self):
        """Test async logging functionality."""
        log_manager = initialize_logging(
            log_level="DEBUG",
            log_format="json",
            enable_async_handler=True
        )
        
        logger = get_logger("test.async")
        
        async def async_worker():
            logger.info("Async message")
            await asyncio.sleep(0.01)
            logger.info("Another async message")
        
        # Run async function
        asyncio.run(async_worker())
    
    def test_logging_config(self):
        """Test logging configuration retrieval."""
        initialize_logging(
            log_level="WARNING",
            log_format="text",
            log_file="test.log",
            include_correlation_id=False
        )
        
        config = get_logging_config()
        
        assert config["status"] == "initialized"
        assert config["log_level"] == 30  # WARNING
        assert config["log_format"] == "text"
        assert config["log_file"] == "test.log"
        assert not config["include_correlation_id"]
    
    def test_error_handling(self):
        """Test error handling in logging system."""
        # Test with invalid log level
        with pytest.raises(AttributeError):
            initialize_logging(log_level="INVALID_LEVEL")
        
        # Test with invalid log format
        log_manager = initialize_logging(log_format="invalid_format")
        # Should fall back to text format
        assert log_manager.log_format == "text"
    
    def test_structured_logging_with_exceptions(self):
        """Test structured logging with exceptions."""
        initialize_logging(log_level="DEBUG", log_format="json", log_file=str(self.log_file))
        
        logger = get_logger("test.exceptions")
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            logger.error("Caught exception", exc_info=True)
        
        # Check that exception info is in the log
        with open(self.log_file, 'r') as f:
            log_line = f.readline().strip()
            log_data = json.loads(log_line)
            
            assert log_data["level"] == "ERROR"
            assert "exception" in log_data
            assert log_data["exception"]["type"] == "ValueError"
            assert "Test exception" in log_data["exception"]["message"]
            assert "traceback" in log_data["exception"]


if __name__ == "__main__":
    pytest.main([__file__])
