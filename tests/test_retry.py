"""Tests for the retry decorator with exponential backoff."""
import asyncio
import random
import time
from typing import Any, Optional, Type
import unittest
from unittest import mock

import pytest

from semantrix.utils.retry import retry


class TestRetryDecorator(unittest.IsolatedAsyncioTestCase):
    """Test cases for the retry decorator."""

    async def test_async_success_on_first_attempt(self) -> None:
        """Test that a successful async function works on first attempt."""
        call_count = 0

        @retry(max_retries=3)
        async def test_func() -> int:
            nonlocal call_count
            call_count += 1
            return 42

        result = await test_func()
        self.assertEqual(result, 42)
        self.assertEqual(call_count, 1)

    async def test_sync_success_on_retry(self) -> None:
        """Test that a sync function succeeds after retries."""
        call_count = 0

        @retry(ValueError, max_retries=3)
        def test_func() -> int:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return 42

        result = test_func()
        self.assertEqual(result, 42)
        self.assertEqual(call_count, 3)

    async def test_async_failure_after_max_retries(self) -> None:
        """Test that an async function raises after max retries."""
        call_count = 0
        max_retries = 2

        @retry(ValueError, max_retries=max_retries)
        async def test_func() -> None:
            nonlocal call_count
            call_count += 1
            raise ValueError("Permanent failure")

        with self.assertRaises(ValueError):
            await test_func()

        self.assertEqual(call_count, max_retries + 1)

    async def test_exponential_backoff(self) -> None:
        """Test that the delay between retries increases exponentially."""
        call_count = 0
        initial_delay = 0.1
        backoff_factor = 2.0
        max_retries = 3
        expected_delays = [
            initial_delay * (backoff_factor ** i) for i in range(max_retries)
        ]

        @retry(
            ValueError,
            max_retries=max_retries,
            initial_delay=initial_delay,
            backoff_factor=backoff_factor,
            jitter=0,  # Disable jitter for predictable test
        )
        async def test_func() -> None:
            nonlocal call_count
            call_count += 1
            if call_count <= max_retries:
                raise ValueError("Temporary failure")

        with mock.patch("asyncio.sleep") as mock_sleep:
            try:
                await test_func()
            except ValueError:
                pass  # Expected

            # Check sleep was called with increasing delays
            self.assertEqual(mock_sleep.call_count, max_retries)
            for i, call in enumerate(mock_sleep.call_args_list):
                expected = min(expected_delays[i], 30.0)  # Respect max_delay
                self.assertAlmostEqual(call[0][0], expected, places=2)

    async def test_jitter(self) -> None:
        """Test that jitter adds randomness to the delay."""
        jitter = 0.2
        delays = set()

        @retry(
            ValueError,
            max_retries=5,
            initial_delay=1.0,
            jitter=jitter,
        )
        async def test_func() -> None:
            raise ValueError("Temporary failure")

        with mock.patch("random.uniform") as mock_uniform, mock.patch(
            "asyncio.sleep"
        ) as mock_sleep:
            # Mock random.uniform to return fixed values for testing
            mock_uniform.side_effect = [1.1, 0.9, 1.15, 0.85, 1.2]

            try:
                await test_func()
            except ValueError:
                pass  # Expected

            # Check that random.uniform was called with correct parameters
            for call in mock_uniform.call_args_list:
                self.assertAlmostEqual(call[0][0], 1 - jitter)
                self.assertAlmostEqual(call[0][1], 1 + jitter)

            # Check that sleep was called with jittered values
            self.assertEqual(mock_sleep.call_count, 5)
            self.assertAlmostEqual(mock_sleep.call_args_list[0][0][0], 1.1, places=2)
            self.assertAlmostEqual(mock_sleep.call_args_list[1][0][0], 1.8, places=2)  # 2.0 * 0.9
            self.assertAlmostEqual(mock_sleep.call_args_list[2][0][0], 4.6, places=2)  # 4.0 * 1.15

    async def test_specific_exception_handling(self) -> None:
        """Test that only specified exceptions are caught and retried."""
        call_count = 0

        @retry(ValueError, max_retries=2)
        async def test_func() -> None:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("This will be retried")
            else:
                raise TypeError("This should not be retried")

        with self.assertRaises(TypeError):
            await test_func()

        self.assertEqual(call_count, 2)  # First attempt + one retry

    def test_sync_function_support(self) -> None:
        """Test that the decorator works with synchronous functions."""
        call_count = 0

        @retry(ValueError, max_retries=2)
        def sync_func() -> int:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Temporary failure")
            return 42

        with mock.patch("time.sleep") as mock_sleep:
            result = sync_func()
            self.assertEqual(result, 42)
            self.assertEqual(call_count, 3)
            self.assertEqual(mock_sleep.call_count, 2)

    async def test_max_delay_respected(self) -> None:
        """Test that the delay doesn't exceed max_delay."""
        max_delay = 2.0
        call_count = 0

        @retry(
            ValueError,
            max_retries=3,
            initial_delay=1.5,
            backoff_factor=2.0,
            max_delay=max_delay,
            jitter=0,  # Disable jitter for predictable test
        )
        async def test_func() -> None:
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise ValueError("Temporary failure")

        with mock.patch("asyncio.sleep") as mock_sleep:
            try:
                await test_func()
            except ValueError:
                pass  # Expected

            # Check sleep was called with delays not exceeding max_delay
            self.assertEqual(mock_sleep.call_count, 3)
            for call in mock_sleep.call_args_list:
                self.assertLessEqual(call[0][0], max_delay)

    async def test_no_retry_on_unexpected_exception(self) -> None:
        """Test that unexpected exceptions are not retried."""
        call_count = 0

        @retry(ValueError, max_retries=3)
        async def test_func() -> None:
            nonlocal call_count
            call_count += 1
            raise TypeError("Unexpected error")

        with self.assertRaises(TypeError):
            await test_func()

        self.assertEqual(call_count, 1)  # No retries for unexpected exception


if __name__ == "__main__":
    unittest.main()
