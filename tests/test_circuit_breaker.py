"""
Tests for circuit breaker functionality.
"""
import asyncio
import time
import unittest
from unittest import mock

from semantrix.utils.circuit_breaker import (
    CircuitBreaker,
    CircuitState,
    circuit_breaker,
)
from semantrix.exceptions import OperationError


class TestCircuitBreaker(unittest.TestCase):
    """Test cases for CircuitBreaker class."""

    def setUp(self):
        """Set up test fixtures."""
        self.breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1.0,
            expected_exception=ValueError,
            name="test_breaker"
        )

    def test_initial_state(self):
        """Test that circuit breaker starts in CLOSED state."""
        self.assertEqual(self.breaker.state, CircuitState.CLOSED)
        self.assertEqual(self.breaker.failure_count, 0)

    def test_successful_operation(self):
        """Test successful operation doesn't affect circuit state."""
        def success_func():
            return "success"

        result = self.breaker._call_sync(success_func)
        self.assertEqual(result, "success")
        self.assertEqual(self.breaker.state, CircuitState.CLOSED)
        self.assertEqual(self.breaker.failure_count, 0)

    def test_failure_below_threshold(self):
        """Test that failures below threshold don't open circuit."""
        def failing_func():
            raise ValueError("test error")

        # Fail twice (below threshold of 3)
        for _ in range(2):
            with self.assertRaises(ValueError):
                self.breaker._call_sync(failing_func)

        self.assertEqual(self.breaker.state, CircuitState.CLOSED)
        self.assertEqual(self.breaker.failure_count, 2)

    def test_circuit_opens_at_threshold(self):
        """Test that circuit opens when failure threshold is reached."""
        def failing_func():
            raise ValueError("test error")

        # Fail exactly at threshold
        for _ in range(3):
            with self.assertRaises(ValueError):
                self.breaker._call_sync(failing_func)

        self.assertEqual(self.breaker.state, CircuitState.OPEN)
        self.assertEqual(self.breaker.failure_count, 3)

    def test_circuit_blocks_calls_when_open(self):
        """Test that circuit blocks calls when in OPEN state."""
        # First, open the circuit
        def failing_func():
            raise ValueError("test error")

        for _ in range(3):
            with self.assertRaises(ValueError):
                self.breaker._call_sync(failing_func)

        # Now try to call a successful function
        def success_func():
            return "success"

        with self.assertRaises(OperationError) as cm:
            self.breaker._call_sync(success_func)

        self.assertIn("Circuit test_breaker is OPEN", str(cm.exception))

    def test_circuit_transitions_to_half_open(self):
        """Test that circuit transitions to HALF_OPEN after timeout."""
        # Open the circuit
        def failing_func():
            raise ValueError("test error")

        for _ in range(3):
            with self.assertRaises(ValueError):
                self.breaker._call_sync(failing_func)

        # Wait for recovery timeout
        time.sleep(1.1)

        # Next call should transition to HALF_OPEN
        def success_func():
            return "success"

        result = self.breaker._call_sync(success_func)
        self.assertEqual(result, "success")
        self.assertEqual(self.breaker.state, CircuitState.CLOSED)

    def test_half_open_failure_reopens_circuit(self):
        """Test that failure in HALF_OPEN state reopens circuit."""
        # Open the circuit
        def failing_func():
            raise ValueError("test error")

        for _ in range(3):
            with self.assertRaises(ValueError):
                self.breaker._call_sync(failing_func)

        # Wait for recovery timeout
        time.sleep(1.1)

        # Fail in HALF_OPEN state
        with self.assertRaises(ValueError):
            self.breaker._call_sync(failing_func)

        self.assertEqual(self.breaker.state, CircuitState.OPEN)

    def test_reset_functionality(self):
        """Test that reset returns circuit to CLOSED state."""
        # Open the circuit
        def failing_func():
            raise ValueError("test error")

        for _ in range(3):
            with self.assertRaises(ValueError):
                self.breaker._call_sync(failing_func)

        # Reset the circuit
        self.breaker.reset()

        self.assertEqual(self.breaker.state, CircuitState.CLOSED)
        self.assertEqual(self.breaker.failure_count, 0)

        # Should work normally after reset
        def success_func():
            return "success"

        result = self.breaker._call_sync(success_func)
        self.assertEqual(result, "success")

    def test_unexpected_exception_doesnt_affect_circuit(self):
        """Test that unexpected exceptions don't affect circuit state."""
        def unexpected_error_func():
            raise RuntimeError("unexpected error")

        with self.assertRaises(RuntimeError):
            self.breaker._call_sync(unexpected_error_func)

        self.assertEqual(self.breaker.state, CircuitState.CLOSED)
        self.assertEqual(self.breaker.failure_count, 0)


class TestCircuitBreakerAsync(unittest.IsolatedAsyncioTestCase):
    """Test cases for async CircuitBreaker functionality."""

    async def asyncSetUp(self):
        """Set up test fixtures."""
        self.breaker = CircuitBreaker(
            failure_threshold=2,
            recovery_timeout=0.1,
            expected_exception=ValueError,
            name="test_async_breaker"
        )

    async def test_async_successful_operation(self):
        """Test successful async operation."""
        async def success_func():
            await asyncio.sleep(0.01)
            return "success"

        result = await self.breaker._call_async(success_func)
        self.assertEqual(result, "success")
        self.assertEqual(self.breaker.state, CircuitState.CLOSED)

    async def test_async_failure_opens_circuit(self):
        """Test that async failures open circuit."""
        async def failing_func():
            await asyncio.sleep(0.01)
            raise ValueError("test error")

        # Fail twice to open circuit
        for _ in range(2):
            with self.assertRaises(ValueError):
                await self.breaker._call_async(failing_func)

        self.assertEqual(self.breaker.state, CircuitState.OPEN)

    async def test_async_circuit_blocks_calls_when_open(self):
        """Test that async circuit blocks calls when open."""
        # Open the circuit
        async def failing_func():
            await asyncio.sleep(0.01)
            raise ValueError("test error")

        for _ in range(2):
            with self.assertRaises(ValueError):
                await self.breaker._call_async(failing_func)

        # Try to call successful function
        async def success_func():
            await asyncio.sleep(0.01)
            return "success"

        with self.assertRaises(OperationError) as cm:
            await self.breaker._call_async(success_func)

        self.assertIn("Circuit test_async_breaker is OPEN", str(cm.exception))


class TestCircuitBreakerDecorator(unittest.TestCase):
    """Test cases for circuit_breaker decorator."""

    def test_sync_decorator(self):
        """Test circuit breaker decorator with sync function."""
        call_count = 0

        @circuit_breaker(failure_threshold=2, expected_exception=ValueError)
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise ValueError("test error")
            return "success"

        # First two calls should fail
        with self.assertRaises(ValueError):
            test_func()
        with self.assertRaises(ValueError):
            test_func()

        # Third call should be blocked
        with self.assertRaises(OperationError):
            test_func()

    async def test_async_decorator(self):
        """Test circuit breaker decorator with async function."""
        call_count = 0

        @circuit_breaker(failure_threshold=2, expected_exception=ValueError)
        async def test_func():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.01)
            if call_count <= 2:
                raise ValueError("test error")
            return "success"

        # First two calls should fail
        with self.assertRaises(ValueError):
            await test_func()
        with self.assertRaises(ValueError):
            await test_func()

        # Third call should be blocked
        with self.assertRaises(OperationError):
            await test_func()


if __name__ == "__main__":
    unittest.main()
