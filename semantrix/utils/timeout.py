"""
Timeout utilities for Semantrix operations.

This module provides timeout functionality to prevent operations from hanging
indefinitely and to ensure responsive behavior.
"""
import asyncio
import functools
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Awaitable, Callable, Optional, TypeVar, Union, cast

from semantrix.exceptions import OperationError
from typing_extensions import ParamSpec, TypeAlias

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")
FuncT: TypeAlias = Union[
    Callable[P, Awaitable[R]],  # Async function
    Callable[P, R],  # Sync function
]


class TimeoutError(OperationError):
    """Raised when an operation times out."""
    pass


def timeout(
    seconds: float,
    timeout_error_message: Optional[str] = None,
) -> Callable[[FuncT[P, R]], FuncT[P, R]]:
    """
    Decorator that adds timeout functionality to functions.
    
    Args:
        seconds: Timeout duration in seconds
        timeout_error_message: Custom error message for timeout
    
    Returns:
        Decorated function with timeout protection
    """
    def decorator(func: FuncT[P, R]) -> FuncT[P, R]:
        is_async = asyncio.iscoroutinefunction(func)
        
        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                try:
                    return await asyncio.wait_for(
                        func(*args, **kwargs),
                        timeout=seconds
                    )
                except asyncio.TimeoutError:
                    message = timeout_error_message or f"Operation {func.__name__} timed out after {seconds}s"
                    raise TimeoutError(message)
            
            return cast(FuncT[P, R], async_wrapper)
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(func, *args, **kwargs)
                    try:
                        return future.result(timeout=seconds)
                    except FutureTimeoutError:
                        message = timeout_error_message or f"Operation {func.__name__} timed out after {seconds}s"
                        raise TimeoutError(message)
            
            return cast(FuncT[P, R], sync_wrapper)
    
    return decorator


class TimeoutContext:
    """
    Context manager for timeout operations.
    
    This provides a more flexible way to apply timeouts to code blocks.
    """
    
    def __init__(self, seconds: float, timeout_error_message: Optional[str] = None):
        """
        Initialize timeout context.
        
        Args:
            seconds: Timeout duration in seconds
            timeout_error_message: Custom error message for timeout
        """
        self.seconds = seconds
        self.timeout_error_message = timeout_error_message
        self._task: Optional[asyncio.Task] = None
        self._thread_executor: Optional[ThreadPoolExecutor] = None
        self._future: Optional[Any] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    def __enter__(self):
        """Sync context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit."""
        if self._future:
            self._future.cancel()
    
    async def run_async(self, coro: Awaitable[R]) -> R:
        """
        Run an async operation with timeout.
        
        Args:
            coro: Coroutine to run
            
        Returns:
            Result of the coroutine
            
        Raises:
            TimeoutError: If the operation times out
        """
        try:
            return await asyncio.wait_for(coro, timeout=self.seconds)
        except asyncio.TimeoutError:
            message = self.timeout_error_message or f"Async operation timed out after {self.seconds}s"
            raise TimeoutError(message)
    
    def run_sync(self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        """
        Run a sync operation with timeout.
        
        Args:
            func: Function to run
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Result of the function
            
        Raises:
            TimeoutError: If the operation times out
        """
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(func, *args, **kwargs)
            try:
                return future.result(timeout=self.seconds)
            except FutureTimeoutError:
                message = self.timeout_error_message or f"Sync operation timed out after {self.seconds}s"
                raise TimeoutError(message)


def with_timeout(seconds: float, timeout_error_message: Optional[str] = None) -> TimeoutContext:
    """
    Create a timeout context for use with 'with' statements.
    
    Args:
        seconds: Timeout duration in seconds
        timeout_error_message: Custom error message for timeout
        
    Returns:
        TimeoutContext instance
    """
    return TimeoutContext(seconds, timeout_error_message)
