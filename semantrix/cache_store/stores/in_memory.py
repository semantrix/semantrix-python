"""
In-memory cache store implementation for Semantrix.

Provides a fast, ephemeral cache using Python's OrderedDict with advanced features:
- Background eviction
- Memory pressure detection
- TTL support
- Monitoring and metrics
"""

import asyncio
import logging
import random
import time
import weakref
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, Deque

# Make psutil optional for memory pressure detection
try:
    import psutil  # type: ignore[import]
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    PSUTIL_AVAILABLE = False

from semantrix.cache_store.base import BaseCacheStore, EvictionPolicy
from semantrix.cache_store.strategies import LRUEvictionStrategy, TTLStrategy
from semantrix.cache_store.eviction_policies import StrategyBasedEvictionPolicy, NoOpEvictionPolicy

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class CacheMetrics:
    """Metrics collector for cache operations."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    ttl_evictions: int = 0
    policy_evictions: int = 0
    last_eviction_time: float = 0
    avg_eviction_duration: float = 0
    memory_high_watermark: float = 0

class InMemoryStore(BaseCacheStore):
    """
    In-memory cache store with advanced features.

    Features:
    - Background eviction
    - Memory pressure detection
    - TTL support
    - Monitoring and metrics

    Args:
        max_size: Maximum number of items to store
        eviction_policy: Custom eviction policy (defaults to LRU strategy)
        enable_ttl: Enable time-to-live functionality
        ttl_seconds: TTL duration in seconds
        eviction_interval: How often to run eviction (seconds)
        enable_memory_pressure: Enable memory pressure detection
        memory_pressure_threshold: Memory usage threshold (0-1) to trigger aggressive eviction
        batch_size: Maximum items to evict in one batch
        sample_size: Number of items to sample for TTL checks
    """
    def __init__(
        self,
        max_size: int = 1000,
        eviction_policy: Optional[EvictionPolicy] = None,
        enable_ttl: bool = True,
        ttl_seconds: int = 3600,
        eviction_interval: float = 60.0,
        enable_memory_pressure: bool = True,
        memory_pressure_threshold: float = 0.8,
        batch_size: int = 100,
        sample_size: int = 1000,
    ):
        self.max_size = max_size
        self.eviction_policy = eviction_policy or StrategyBasedEvictionPolicy(LRUEvictionStrategy())
        self.enable_ttl = enable_ttl
        self.ttl_seconds = ttl_seconds
        self.eviction_interval = eviction_interval
        self.enable_memory_pressure = enable_memory_pressure
        self.memory_pressure_threshold = memory_pressure_threshold
        self.batch_size = batch_size
        self.sample_size = sample_size
        
        self._cache: Dict[str, Any] = OrderedDict()
        self._lock = asyncio.Lock()
        self._eviction_task: Optional[asyncio.Task] = None
        self._running = False
        self._closed = False
        
        # Metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.memory_high_watermark = 0.0
        
        # Use weakref.finalize for cleanup on garbage collection
        self._finalizer = weakref.finalize(
            self,
            self._cleanup,
            self._eviction_task,
            self._lock
        )
        
        # Start background eviction task if enabled
        if eviction_interval > 0:
            self._start_eviction_task()

    async def _check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        if not PSUTIL_AVAILABLE or not self.enable_memory_pressure:
            return False
            
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            self.memory_high_watermark = max(
                self.memory_high_watermark,
                memory_info.rss / (1024 * 1024)  # Convert to MB
            )
            return memory_info.rss > self.memory_pressure_threshold * psutil.virtual_memory().total
        except Exception as e:
            logger.warning(f"Failed to check memory pressure: {e}")
            return False

    async def _sample_items(self, sample_size: int) -> List[Tuple[str, Any]]:
        """Get a random sample of cache items."""
        if len(self._cache) <= sample_size:
            return list(self._cache.items())
        return random.sample(list(self._cache.items()), sample_size)

    async def _evict_in_batches(self, keys: List[str]) -> int:
        """Evict keys in batches to prevent long blocking."""
        evicted = 0
        for i in range(0, len(keys), self.batch_size):
            batch = keys[i:i + self.batch_size]
            for key in batch:
                if key in self._cache:
                    self._cache.pop(key, None)
                    evicted += 1
            # Small sleep to prevent starving other tasks
            await asyncio.sleep(0.001)
        return evicted

    async def _evict_expired_items(self) -> int:
        """Evict expired items using sampling."""
        if not self.enable_ttl or not self._cache:
            return 0

        start_time = time.time()
        sample = await self._sample_items(min(self.sample_size, len(self._cache)))
        current_time = time.time()
        
        expired = [
            k for k, v in sample
            if isinstance(v, dict) and 
               current_time - v.get('timestamp', 0) > self.ttl_seconds
        ]
        
        evicted = await self._evict_in_batches(expired)
        
        duration = (time.time() - start_time) * 1000  # ms
        logger.debug(f"TTL eviction: checked {len(sample)} items, evicted {evicted} in {duration:.2f}ms")
        
        self.evictions += evicted
        return evicted

    async def _run_eviction_cycle(self) -> int:
        """Run one eviction cycle, returns number of evicted items."""
        start_time = time.time()
        total_evicted = 0
        under_pressure = await self._check_memory_pressure()
        
        try:
            # Always check TTL if enabled
            if self.enable_ttl:
                total_evicted += await self._evict_expired_items()

            # Check if we need policy-based eviction
            needs_eviction = (
                len(self._cache) > self.max_size or 
                under_pressure
            )

            if needs_eviction:
                evicted = await self.eviction_policy.apply(
                    self._cache, 
                    max(0, self.max_size - total_evicted)
                )
                total_evicted += evicted
                self.evictions += evicted

            # Update metrics
            if total_evicted > 0:
                duration = (time.time() - start_time) * 1000  # ms
                self.memory_high_watermark = max(self.memory_high_watermark, duration)
                logger.info(
                    f"Eviction cycle: evicted {total_evicted} items "
                    f"in {duration:.2f}ms (pressure: {under_pressure})"
                )

        except Exception as e:
            logger.error(f"Error during eviction cycle: {e}", exc_info=True)
        
        return total_evicted

    async def _eviction_loop(self):
        """Main eviction loop."""
        while getattr(self, '_running', True):
            try:
                start_time = time.time()
                await self._run_eviction_cycle()
                
                # Adaptive sleep based on last run time
                elapsed = time.time() - start_time
                sleep_time = max(1.0, self.eviction_interval - elapsed)
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in eviction loop: {e}", exc_info=True)
                await asyncio.sleep(min(60, self.eviction_interval * 2))  # Backoff

    def _start_eviction_task(self):
        """Start background eviction tasks."""
        self._running = True
        self._eviction_task = asyncio.create_task(self._eviction_loop())
        logger.info("Started background eviction task")

    async def close(self):
        """Stop the background eviction task and clean up resources."""
        if self._closed:
            return
            
        self._running = False
        
        # Cancel the eviction task if it exists
        if self._eviction_task and not self._eviction_task.done():
            self._eviction_task.cancel()
            try:
                await self._eviction_task
            except asyncio.CancelledError:
                pass
            except Exception:
                # Log any other exceptions during task cancellation
                import logging
                logging.exception("Error during eviction task cleanup")
                
        self._closed = True
        
    def _cleanup(self, task: Optional[asyncio.Task], lock: asyncio.Lock):
        """Cleanup function that runs when the instance is garbage collected."""
        if not self._closed:
            # If we're in an event loop, schedule cleanup
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    if task and not task.done():
                        task.cancel()
                    # If we can't get the lock immediately, schedule the cleanup
                    if not lock.locked():
                        asyncio.create_task(self.close())
            except (RuntimeError, AttributeError):
                # If there's no event loop or it's closed, just cancel the task
                if task and not task.done():
                    task.cancel()
            except Exception:
                # Don't let cleanup exceptions propagate
                pass

    async def get_exact(self, prompt: str) -> Optional[str]:
        if prompt in self._cache:
            value = self._cache[prompt]
            # Handle TTL if enabled
            if self.enable_ttl and isinstance(value, dict):
                if time.time() - value['timestamp'] > self.ttl_seconds:
                    del self._cache[prompt]
                    self.misses += 1
                    return None
                response = value['response']
            else:
                response = value
            self._cache.move_to_end(prompt)
            self.hits += 1
            return response
        self.misses += 1
        return None

    async def add(self, prompt: str, response: str) -> None:
        if self.enable_ttl:
            self._cache[prompt] = {
                'response': response,
                'timestamp': time.time()
            }
        else:
            self._cache[prompt] = response
        self._cache.move_to_end(prompt)
        
        # Trigger eviction if we're over size limit
        if len(self._cache) > self.max_size * 1.1:  # 10% over limit
            await self._run_eviction_cycle()

    async def enforce_limits(self, resource_limits: Any) -> None:
        """Enforce cache limits asynchronously."""
        evicted_count = await self._run_eviction_cycle()
        if evicted_count > 0:
            logger.info(f"Enforced limits: evicted {evicted_count} items")

    async def clear(self) -> None:
        self._cache.clear()

    async def size(self) -> int:
        return len(self._cache)

    def get_eviction_policy(self) -> EvictionPolicy:
        return self.eviction_policy

    def get_stats(self) -> dict:
        """Get detailed cache statistics and metrics."""
        stats = {
            'size': self.size(),
            'max_size': self.max_size,
            'eviction_policy': type(self.eviction_policy).__name__,
            'enable_ttl': self.enable_ttl,
            'evicted_count': getattr(self.eviction_policy, 'evicted_count', 0),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': (
                self.hits / (self.hits + self.misses)
                if (self.hits + self.misses) > 0 else 0
            ),
            'total_evictions': self.evictions,
            'memory_high_watermark': self.memory_high_watermark,
            'background_task_running': hasattr(self, '_running') and self._running
        }
        
        # Add memory info if available
        try:
            mem = psutil.virtual_memory()
            stats.update({
                'memory_used_percent': mem.percent,
                'memory_available_gb': mem.available / (1024 ** 3),
                'memory_total_gb': mem.total / (1024 ** 3)
            })
        except Exception as e:
            logger.warning(f"Could not get memory info: {e}")
            
        return stats