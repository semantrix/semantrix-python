#!/usr/bin/env python3
"""
Demonstration of Semantrix's improved logging system.

This example shows how to use structured logging with correlation IDs,
metrics logging, and different configuration presets.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from semantrix import Semantrix
from semantrix.utils.logging import (
    initialize_logging,
    get_logger,
    get_adapter,
    with_correlation_id,
    get_metrics_logger,
    set_correlation_id,
    clear_correlation_id
)
from semantrix.utils.logging_config import (
    LoggingPresets,
    configure_from_environment,
    configure_for_service
)


async def demo_basic_logging():
    """Demonstrate basic logging functionality."""
    print("🔧 Setting up basic logging...")
    
    # Initialize logging with development preset
    LoggingPresets.development(enable_async=True)
    
    # Get loggers
    logger = get_logger("semantrix.demo")
    metrics_logger = get_metrics_logger()
    
    logger.info("Basic logging demo started")
    
    # Demonstrate different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Demonstrate structured logging with extra fields
    logger.info("User action completed", extra={
        "user_id": "user-123",
        "action": "login",
        "duration_ms": 150,
        "success": True
    })
    
    # Demonstrate metrics logging
    metrics_logger.log_operation_start("demo_operation", operation_type="test")
    await asyncio.sleep(0.1)  # Simulate work
    metrics_logger.log_operation_end("demo_operation", 0.1, True, result_count=5)
    
    metrics_logger.log_cache_hit("memory", "test_key", hit_type="exact")
    metrics_logger.log_cache_miss("redis", "missing_key", miss_reason="not_found")
    
    logger.info("Basic logging demo completed")


async def demo_correlation_ids():
    """Demonstrate correlation ID functionality."""
    print("\n🔗 Setting up correlation ID demo...")
    
    logger = get_logger("semantrix.correlation")
    
    # Set a correlation ID for the entire request
    request_id = "req-12345"
    set_correlation_id(request_id)
    
    logger.info("Request processing started", extra={"user_id": "user-456"})
    
    # Simulate nested operations
    await process_user_data("user-456")
    await validate_request("req-12345")
    
    logger.info("Request processing completed")
    
    # Clear correlation ID
    clear_correlation_id()


async def process_user_data(user_id: str):
    """Simulate processing user data with correlation ID."""
    logger = get_logger("semantrix.user_processor")
    
    # The correlation ID is automatically inherited from the parent context
    logger.info("Processing user data", extra={"user_id": user_id})
    
    # Use context manager for sub-operations
    with with_correlation_id(f"user-{user_id}-proc"):
        logger.info("User data processing started")
        await asyncio.sleep(0.05)  # Simulate work
        logger.info("User data processing completed")


async def validate_request(request_id: str):
    """Simulate request validation with correlation ID."""
    logger = get_logger("semantrix.validator")
    
    logger.info("Validating request", extra={"request_id": request_id})
    
    # Simulate validation steps
    validation_steps = ["auth", "permissions", "rate_limit"]
    for step in validation_steps:
        with with_correlation_id(f"{request_id}-{step}"):
            logger.info(f"Validation step: {step}")
            await asyncio.sleep(0.02)  # Simulate work
            logger.info(f"Validation step completed: {step}")


async def demo_logger_adapters():
    """Demonstrate logger adapters with bound context."""
    print("\n📝 Setting up logger adapter demo...")
    
    # Create a logger adapter with correlation ID and extra fields
    adapter = get_adapter(
        "semantrix.adapter_demo",
        correlation_id="adapter-demo-123",
        service="demo_service",
        version="1.0.0"
    )
    
    adapter.info("Logger adapter demo started")
    
    # Create a new adapter with additional context
    user_adapter = adapter.bind(user_id="user-789", session_id="sess-456")
    user_adapter.info("User session started")
    
    # Simulate user actions
    actions = ["view_profile", "update_settings", "logout"]
    for action in actions:
        action_adapter = user_adapter.bind(action=action, timestamp=asyncio.get_event_loop().time())
        action_adapter.info(f"User performed action: {action}")
        await asyncio.sleep(0.01)
    
    adapter.info("Logger adapter demo completed")


async def demo_semantrix_integration():
    """Demonstrate logging integration with Semantrix."""
    print("\n🤖 Setting up Semantrix integration demo...")
    
    # Configure logging for the service
    configure_for_service("semantrix_demo", "development", enable_async=True)
    
    # Initialize Semantrix with logging enabled but WAL disabled
    semantrix = Semantrix(
        similarity_threshold=0.8,
        enable_wal=False,  # Disable WAL to avoid the error
        enable_2pc=False,  # Disable 2PC as well
        enable_logging=True,
        logging_config={
            "log_level": "DEBUG",
            "log_format": "json",
            "include_correlation_id": True,
            "enable_async_handler": True
        }
    )
    
    await semantrix.initialize()
    
    # Simulate cache operations
    test_prompts = [
        "What is the capital of France?",
        "How do I make a cake?",
        "What is machine learning?"
    ]
    
    for i, prompt in enumerate(test_prompts):
        with with_correlation_id(f"cache-op-{i}"):
            logger = get_logger("semantrix.cache_demo")
            logger.info("Processing cache request", extra={
                "prompt_length": len(prompt),
                "request_index": i
            })
            
            # Simulate cache operations
            await semantrix.set(prompt, f"Response to: {prompt}")
            result = await semantrix.get(prompt)
            
            logger.info("Cache operation completed", extra={
                "found": result is not None,
                "result_length": len(result) if result else 0
            })
    
    await semantrix.shutdown()


async def demo_environment_configuration():
    """Demonstrate environment-based configuration."""
    print("\n🌍 Setting up environment configuration demo...")
    
    # Set environment variables for configuration
    os.environ["SEMANTRIX_LOG_LEVEL"] = "DEBUG"
    os.environ["SEMANTRIX_LOG_FORMAT"] = "json"
    os.environ["SEMANTRIX_LOG_ENABLE_ASYNC"] = "true"
    os.environ["SEMANTRIX_LOG_INCLUDE_CORRELATION_ID"] = "true"
    
    # Configure from environment
    configure_from_environment()
    
    logger = get_logger("semantrix.env_demo")
    logger.info("Environment-based configuration demo started")
    
    # Demonstrate different presets
    presets = [
        ("development", LoggingPresets.development),
        ("production", LoggingPresets.production),
        ("testing", LoggingPresets.testing),
        ("docker", LoggingPresets.docker)
    ]
    
    for name, preset_func in presets:
        logger.info(f"Testing {name} preset")
        # Note: In a real application, you'd only use one preset
        # This is just for demonstration
    
    logger.info("Environment configuration demo completed")


async def main():
    """Run all logging demonstrations."""
    print("🚀 Semantrix Logging System Demo")
    print("=" * 50)
    
    try:
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        # Run all demos
        await demo_basic_logging()
        await demo_correlation_ids()
        await demo_logger_adapters()
        await demo_semantrix_integration()
        await demo_environment_configuration()
        
        print("\n✅ All logging demos completed successfully!")
        print("\n📁 Check the 'logs' directory for log files:")
        print("   - semantrix_dev.log (development logs)")
        print("   - semantrix_demo_development.log (service logs)")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
