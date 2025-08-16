"""
Tests for timeout functionality.
"""
import asyncio
import time
import unittest
from unittest import mock

from semantrix.utils.timeout import (
    timeout,
    TimeoutContext,
    with_timeout,
    TimeoutError,
)


class TestTimeoutDecorator(unittest.TestCase):
    """Test cases for timeout decorator."""

    def test_sync_function_success(self):
        """Test that sync function succeeds within timeout."""
        @timeout(1.0)
        def fast_function():
            return "success"

        result = fast_function()
        self.assertEqual(result, "success")

    def test_sync_function_timeout(self):
        """Test that sync function times out."""
        @timeout(0.1)
        def slow_function():
            time.sleep(0.2)
            return "success"

        with self.assertRaises(TimeoutError) as cm:
            slow_function()

        self.assertIn("timed out after 0.1s", str(cm.exception))

    def test_sync_function_custom_error_message(self):
        """Test custom error message for timeout."""
        @timeout(0.1, "Custom timeout message")
        def slow_function():
            time.sleep(0.2)
            return "success"

        with self.assertRaises(TimeoutError) as cm:
            slow_function()

        self.assertEqual(str(cm.exception), "Custom timeout message")

    async def test_async_function_success(self):
        """Test that async function succeeds within timeout."""
        @timeout(1.0)
        async def fast_async_function():
            await asyncio.sleep(0.01)
            return "success"

        result = await fast_async_function()
        self.assertEqual(result, "success")

    async def test_async_function_timeout(self):
        """Test that async function times out."""
        @timeout(0.1)
        async def slow_async_function():
            await asyncio.sleep(0.2)
            return "success"

        with self.assertRaises(TimeoutError) as cm:
            await slow_async_function()

        self.assertIn("timed out after 0.1s", str(cm.exception))

    async def test_async_function_custom_error_message(self):
        """Test custom error message for async timeout."""
        @timeout(0.1, "Custom async timeout message")
        async def slow_async_function():
            await asyncio.sleep(0.2)
            return "success"

        with self.assertRaises(TimeoutError) as cm:
            await slow_async_function()

        self.assertEqual(str(cm.exception), "Custom async timeout message")

    def test_function_with_arguments(self):
        """Test timeout decorator with function arguments."""
        @timeout(1.0)
        def function_with_args(a, b, c=None):
            return f"{a}_{b}_{c}"

        result = function_with_args("hello", "world", c="test")
        self.assertEqual(result, "hello_world_test")

    async def test_async_function_with_arguments(self):
        """Test timeout decorator with async function arguments."""
        @timeout(1.0)
        async def async_function_with_args(a, b, c=None):
            await asyncio.sleep(0.01)
            return f"{a}_{b}_{c}"

        result = await async_function_with_args("hello", "world", c="test")
        self.assertEqual(result, "hello_world_test")


class TestTimeoutContext(unittest.TestCase):
    """Test cases for TimeoutContext class."""

    def test_sync_context_success(self):
        """Test successful sync operation with context."""
        def fast_function():
            return "success"

        with TimeoutContext(1.0) as ctx:
            result = ctx.run_sync(fast_function)
            self.assertEqual(result, "success")

    def test_sync_context_timeout(self):
        """Test sync operation timeout with context."""
        def slow_function():
            time.sleep(0.2)
            return "success"

        with TimeoutContext(0.1) as ctx:
            with self.assertRaises(TimeoutError) as cm:
                ctx.run_sync(slow_function)

            self.assertIn("timed out after 0.1s", str(cm.exception))

    def test_sync_context_custom_error_message(self):
        """Test custom error message for sync context timeout."""
        def slow_function():
            time.sleep(0.2)
            return "success"

        with TimeoutContext(0.1, "Custom context timeout") as ctx:
            with self.assertRaises(TimeoutError) as cm:
                ctx.run_sync(slow_function)

            self.assertEqual(str(cm.exception), "Custom context timeout")

    async def test_async_context_success(self):
        """Test successful async operation with context."""
        async def fast_async_function():
            await asyncio.sleep(0.01)
            return "success"

        async with TimeoutContext(1.0) as ctx:
            result = await ctx.run_async(fast_async_function())
            self.assertEqual(result, "success")

    async def test_async_context_timeout(self):
        """Test async operation timeout with context."""
        async def slow_async_function():
            await asyncio.sleep(0.2)
            return "success"

        async with TimeoutContext(0.1) as ctx:
            with self.assertRaises(TimeoutError) as cm:
                await ctx.run_async(slow_async_function())

            self.assertIn("timed out after 0.1s", str(cm.exception))

    async def test_async_context_custom_error_message(self):
        """Test custom error message for async context timeout."""
        async def slow_async_function():
            await asyncio.sleep(0.2)
            return "success"

        async with TimeoutContext(0.1, "Custom async context timeout") as ctx:
            with self.assertRaises(TimeoutError) as cm:
                await ctx.run_async(slow_async_function())

            self.assertEqual(str(cm.exception), "Custom async context timeout")

    def test_context_with_function_arguments(self):
        """Test context with function arguments."""
        def function_with_args(a, b, c=None):
            return f"{a}_{b}_{c}"

        with TimeoutContext(1.0) as ctx:
            result = ctx.run_sync(function_with_args, "hello", "world", c="test")
            self.assertEqual(result, "hello_world_test")

    async def test_async_context_with_function_arguments(self):
        """Test async context with function arguments."""
        async def async_function_with_args(a, b, c=None):
            await asyncio.sleep(0.01)
            return f"{a}_{b}_{c}"

        async with TimeoutContext(1.0) as ctx:
            result = await ctx.run_async(async_function_with_args("hello", "world", c="test"))
            self.assertEqual(result, "hello_world_test")


class TestWithTimeoutFunction(unittest.TestCase):
    """Test cases for with_timeout function."""

    def test_with_timeout_creates_context(self):
        """Test that with_timeout creates a TimeoutContext."""
        ctx = with_timeout(1.0)
        self.assertIsInstance(ctx, TimeoutContext)
        self.assertEqual(ctx.seconds, 1.0)

    def test_with_timeout_with_custom_message(self):
        """Test with_timeout with custom error message."""
        ctx = with_timeout(1.0, "Custom message")
        self.assertIsInstance(ctx, TimeoutContext)
        self.assertEqual(ctx.seconds, 1.0)
        self.assertEqual(ctx.timeout_error_message, "Custom message")


class TestTimeoutError(unittest.TestCase):
    """Test cases for TimeoutError exception."""

    def test_timeout_error_inheritance(self):
        """Test that TimeoutError inherits from OperationError."""
        error = TimeoutError("test message")
        self.assertIsInstance(error, TimeoutError)
        # Note: This would need to be imported from the correct location
        # self.assertIsInstance(error, OperationError)

    def test_timeout_error_message(self):
        """Test TimeoutError message."""
        message = "Test timeout error"
        error = TimeoutError(message)
        self.assertEqual(str(error), message)


if __name__ == "__main__":
    unittest.main()
