"""
Retry decorator with exponential backoff for Semantrix operations.

This module provides a retry decorator that implements exponential backoff with jitter
to handle transient failures in distributed systems.
"""
import asyncio
import functools
import logging
import random
import time
from typing import Any, Awaitable, Callable, Optional, Type, TypeVar, Union, cast

from typing_extensions import ParamSpec, TypeAlias

T = TypeVar("T")
P = ParamSpec("P")
R = TypeVar("R")
FuncT: TypeAlias = Union[
    Callable[P, Awaitable[R]],  # Async function
    Callable[P, R],  # Sync function
]

logger = logging.getLogger(__name__)


def retry(
    *exceptions: Type[BaseException],
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: float = 0.1,
) -> Callable[[FuncT[P, R]], FuncT[P, R]]:
    """
    Decorator that retries the wrapped function with exponential backoff.

    Args:
        exceptions: Exception types to catch and retry on.
        max_retries: Maximum number of retry attempts.
        initial_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.
        backoff_factor: Multiplier for the delay between retries.
        jitter: Random jitter factor to avoid thundering herd problem.

    Returns:
        A decorator that can be applied to functions or coroutines.
    """
    if not exceptions:
        exceptions = (Exception,)

    def decorator(func: FuncT[P, R]) -> FuncT[P, R]:
        is_async = asyncio.iscoroutinefunction(func)

        if is_async:
            # Handle async functions
            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                last_exception: Optional[Exception] = None
                delay = initial_delay

                for attempt in range(max_retries + 1):
                    try:
                        return await cast(Awaitable[R], func(*args, **kwargs))
                    except exceptions as e:
                        last_exception = e
                        if attempt == max_retries:
                            break

                        # Calculate next delay with jitter
                        jitter_amount = random.uniform(1 - jitter, 1 + jitter)
                        current_delay = min(delay * jitter_amount, max_delay)
                        
                        logger.warning(
                            "Attempt %s/%s failed: %s. Retrying in %.2fs...",
                            attempt + 1,
                            max_retries,
                            str(e),
                            current_delay,
                            exc_info=True,
                        )

                        await asyncio.sleep(current_delay)
                        delay = min(delay * backoff_factor, max_delay)

                # If we've exhausted all retries, raise the last exception
                logger.error(
                    "All %s attempts failed. Last error: %s",
                    max_retries + 1,
                    str(last_exception),
                    exc_info=True,
                )
                raise last_exception  # type: ignore

            return cast(FuncT[P, R], async_wrapper)

        else:
            # Handle sync functions
            @functools.wraps(func)
            def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                last_exception: Optional[Exception] = None
                delay = initial_delay

                for attempt in range(max_retries + 1):
                    try:
                        return cast(R, func(*args, **kwargs))
                    except exceptions as e:
                        last_exception = e
                        if attempt == max_retries:
                            break

                        # Calculate next delay with jitter
                        jitter_amount = random.uniform(1 - jitter, 1 + jitter)
                        current_delay = min(delay * jitter_amount, max_delay)
                        
                        logger.warning(
                            "Attempt %s/%s failed: %s. Retrying in %.2fs...",
                            attempt + 1,
                            max_retries,
                            str(e),
                            current_delay,
                            exc_info=True,
                        )

                        time.sleep(current_delay)
                        delay = min(delay * backoff_factor, max_delay)

                # If we've exhausted all retries, raise the last exception
                logger.error(
                    "All %s attempts failed. Last error: %s",
                    max_retries + 1,
                    str(last_exception),
                    exc_info=True,
                )
                raise last_exception  # type: ignore

            return cast(FuncT[P, R], sync_wrapper)

    return decorator
