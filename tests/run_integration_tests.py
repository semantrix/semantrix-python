"""
Integration Test Runner for Semantrix Cache Stores

This script runs integration tests for various cache store implementations.
By default, it runs tests using local services (DynamoDB Local, Redis, etc.).
To test against actual cloud services, set the appropriate environment variables.

Environment Variables:
    RUN_INTEGRATION_TESTS: Set to "true" to enable integration tests
    
    # For DynamoDB
    DYNAMODB_ENDPOINT_URL: Endpoint for DynamoDB Local (default: http://localhost:8000)
    
    # For ElastiCache/Redis
    ELASTICACHE_ENDPOINT: Redis endpoint (default: localhost:6379)
    ELASTICACHE_PASSWORD: Redis password if required
    
    # For Google Memorystore
    GOOGLE_CLOUD_PROJECT: GCP project ID
    GOOGLE_CLOUD_REGION: GCP region (default: us-central1)
    GOOGLE_MEMORYSTORE_INSTANCE: Memorystore instance ID (default: semantrix-test)
"""

import os
import sys
import asyncio
import argparse
import subprocess
from typing import List, Optional

# Test modules to run
TEST_MODULES = [
    "test_cache_store_dynamodb",
    "test_cache_store_elasticache",
    "test_cache_store_google_memorystore"
]

# Service ports for local testing
SERVICE_PORTS = {
    "dynamodb": 8000,
    "redis": 6379
}


def check_dependencies():
    """Check if required dependencies are installed."""
    required = ["pytest", "pytest-asyncio"]
    missing = []
    
    for package in required:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"Error: Missing required dependencies: {', '.join(missing)}")
        print("Please install them using:")
        print(f"    pip install {' '.join(missing)}")
        sys.exit(1)


def start_local_services(services: List[str]):
    """Start local services for testing."""
    processes = {}
    
    if "dynamodb" in services:
        print("Starting DynamoDB Local...")
        try:
            # Check if DynamoDB Local is already running
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex(('localhost', SERVICE_PORTS["dynamDB"]))
            if result != 0:
                processes["dynamodb"] = subprocess.Popen(
                    ["java", "-Djava.library.path=./DynamoDBLocal_lib", "-jar", "DynamoDBLocal.jar", "-sharedDb", "-port", str(SERVICE_PORTS["dynamodb"])],
                    cwd=".dynamodb",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                # Give it a moment to start
                import time
                time.sleep(2)
        except Exception as e:
            print(f"Warning: Could not start DynamoDB Local: {e}")
            print("Make sure you have Java installed and DynamoDB Local set up in the .dynamodb directory.")
    
    if "redis" in services:
        print("Starting Redis...")
        try:
            processes["redis"] = subprocess.Popen(
                ["redis-server", "--port", str(SERVICE_PORTS["redis"])],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            # Give it a moment to start
            import time
            time.sleep(1)
        except Exception as e:
            print(f"Warning: Could not start Redis: {e}")
            print("Make sure Redis is installed and available in your PATH.")
    
    return processes


def stop_local_services(processes):
    """Stop local services."""
    for name, process in processes.items():
        try:
            print(f"Stopping {name}...")
            process.terminate()
            process.wait(timeout=5)
        except Exception as e:
            print(f"Error stopping {name}: {e}")
            try:
                process.kill()
            except:
                pass


async def run_tests(modules: List[str], services: List[str]):
    """Run integration tests."""
    # Set environment variables for tests
    os.environ["RUN_INTEGRATION_TESTS"] = "true"
    
    # Set default endpoints for local services
    if "dynamodb" in services:
        os.environ["DYNAMODB_ENDPOINT_URL"] = f"http://localhost:{SERVICE_PORTS['dynamodb']}"
    
    if "redis" in services:
        os.environ["ELASTICACHE_ENDPOINT"] = f"localhost:{SERVICE_PORTS['redis']}"
    
    # Run tests
    import pytest
    
    args = [
        "-v",
        "--asyncio-mode=auto",
        "--log-level=INFO",
        "-m", "integration"
    ]
    
    # Add test modules
    for module in modules:
        args.append(f"tests/{module}.py")
    
    # Run pytest
    return await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: pytest.main(args)
    )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run Semantrix cache store integration tests")
    parser.add_argument(
        "--services", 
        nargs="+", 
        choices=["all", "dynamodb", "redis"], 
        default=["all"],
        help="Services to test against"
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        choices=TEST_MODULES + ["all"],
        default=["all"],
        help="Test modules to run"
    )
    parser.add_argument(
        "--no-local",
        action="store_true",
        help="Don't start local services, assume they're already running"
    )
    
    args = parser.parse_args()
    
    # Expand 'all' in services
    if "all" in args.services:
        args.services = ["dynamodb", "redis"]
    
    # Expand 'all' in modules
    if "all" in args.modules:
        args.modules = TEST_MODULES
    
    # Check dependencies
    check_dependencies()
    
    # Start local services if needed
    processes = {}
    if not args.no_local and args.services:
        processes = start_local_services(args.services)
    
    try:
        # Run tests
        exit_code = asyncio.run(run_tests(args.modules, args.services))
        sys.exit(exit_code)
    finally:
        # Stop local services
        if processes:
            stop_local_services(processes)


if __name__ == "__main__":
    main()
