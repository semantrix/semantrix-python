"""
Circuit Breaker pattern implementation for Semantrix.

This module provides a circuit breaker that can prevent cascading failures
by temporarily stopping execution of operations that are likely to fail.
"""
import asyncio
import functools
import time
from enum import Enum
from typing import Any, Awaitable, Callable, Optional, Type, TypeVar, Union, cast

from semantrix.exceptions import OperationError
from typing_extensions import ParamSpec, TypeAlias
from semantrix.utils.logging import get_logger

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")
FuncT: TypeAlias = Union[
    Callable[P, Awaitable[R]],  # Async function
    Callable[P, R],  # Sync function
]

logger = get_logger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Circuit is open, calls fail fast
    HALF_OPEN = "half_open"  # Testing if service is back


class CircuitBreaker:
    """
    Circuit breaker implementation with configurable failure thresholds.
    
    The circuit breaker has three states:
    - CLOSED: Normal operation, calls pass through
    - OPEN: Circuit is open, calls fail fast
    - HALF_OPEN: Testing if service is back to normal
    """
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Type[Exception] = Exception,
        name: str = "default",
    ):
        """
        Initialize the circuit breaker.
        
        Args:
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Time to wait before trying to close circuit
            expected_exception: Exception type that indicates failure
            name: Name for logging and identification
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.name = name
        
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        self._lock = asyncio.Lock()
    
    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        return self._state
    
    @property
    def failure_count(self) -> int:
        """Get current failure count."""
        return self._failure_count
    
    async def _call_async(self, func: Callable[P, Awaitable[R]], *args: P.args, **kwargs: P.kwargs) -> R:
        """Execute async function with circuit breaker logic."""
        async with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    logger.info(f"Circuit {self.name}: Attempting to close circuit")
                    self._state = CircuitState.HALF_OPEN
                else:
                    raise OperationError(f"Circuit {self.name} is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except self.expected_exception as e:
            await self._on_failure()
            raise
    
    def _call_sync(self, func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
        """Execute sync function with circuit breaker logic."""
        if self._state == CircuitState.OPEN:
            if time.time() - self._last_failure_time >= self.recovery_timeout:
                logger.info(f"Circuit {self.name}: Attempting to close circuit")
                self._state = CircuitState.HALF_OPEN
            else:
                raise OperationError(f"Circuit {self.name} is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success_sync()
            return result
        except self.expected_exception as e:
            self._on_failure_sync()
            raise
    
    async def _on_success(self):
        """Handle successful operation."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                logger.info(f"Circuit {self.name}: Success, closing circuit")
                self._state = CircuitState.CLOSED
                self._failure_count = 0
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0
    
    def _on_success_sync(self):
        """Handle successful operation (sync version)."""
        if self._state == CircuitState.HALF_OPEN:
            logger.info(f"Circuit {self.name}: Success, closing circuit")
            self._state = CircuitState.CLOSED
            self._failure_count = 0
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0
    
    async def _on_failure(self):
        """Handle failed operation."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.time()
            
            if self._state == CircuitState.CLOSED and self._failure_count >= self.failure_threshold:
                logger.warning(f"Circuit {self.name}: Opening circuit after {self._failure_count} failures")
                self._state = CircuitState.OPEN
            elif self._state == CircuitState.HALF_OPEN:
                logger.warning(f"Circuit {self.name}: Re-opening circuit after failure")
                self._state = CircuitState.OPEN
    
    def _on_failure_sync(self):
        """Handle failed operation (sync version)."""
        self._failure_count += 1
        self._last_failure_time = time.time()
        
        if self._state == CircuitState.CLOSED and self._failure_count >= self.failure_threshold:
            logger.warning(f"Circuit {self.name}: Opening circuit after {self._failure_count} failures")
            self._state = CircuitState.OPEN
        elif self._state == CircuitState.HALF_OPEN:
            logger.warning(f"Circuit {self.name}: Re-opening circuit after failure")
            self._state = CircuitState.OPEN
    
    def reset(self):
        """Reset the circuit breaker to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._last_failure_time = 0.0
        logger.info(f"Circuit {self.name}: Reset to closed state")


def circuit_breaker(
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    expected_exception: Type[Exception] = Exception,
    name: Optional[str] = None,
) -> Callable[[FuncT[P, R]], FuncT[P, R]]:
    """
    Decorator that applies circuit breaker pattern to functions.
    
    Args:
        failure_threshold: Number of failures before opening circuit
        recovery_timeout: Time to wait before trying to close circuit
        expected_exception: Exception type that indicates failure
        name: Name for the circuit breaker (defaults to function name)
    
    Returns:
        Decorated function with circuit breaker protection
    """
    def decorator(func: FuncT[P, R]) -> FuncT[P, R]:
        circuit_name = name or func.__name__
        breaker = CircuitBreaker(
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
            expected_exception=expected_exception,
            name=circuit_name,
        )
        
        is_async = asyncio.iscoroutinefunction(func)
        
        if is_async:
            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                return await breaker._call_async(func, *args, **kwargs)
            return cast(FuncT[P, R], async_wrapper)
        else:
            @functools.wraps(func)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                return breaker._call_sync(func, *args, **kwargs)
            return cast(FuncT[P, R], sync_wrapper)
    
    return decorator
