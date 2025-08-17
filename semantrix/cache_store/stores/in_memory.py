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

from semantrix.cache_store.base import BaseCacheStore, EvictionPolicy, DeletionMode
from semantrix.cache_store.strategies import LRUEvictionStrategy, TTLStrategy
from semantrix.cache_store.eviction_policies import StrategyBasedEvictionPolicy, NoOpEvictionPolicy
from semantrix.exceptions import CacheOperationError

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class CacheMetrics:
    """
    Metrics collector for cache operations and performance.
    
    Attributes:
        hits: Number of successful cache lookups
        misses: Number of failed cache lookups
        evictions: Total number of items evicted
        ttl_evictions: Number of items evicted due to TTL expiration
        policy_evictions: Number of items evicted due to cache policy
        last_eviction_time: Timestamp of last eviction cycle
        avg_eviction_duration: Moving average of eviction cycle durations (ms)
        memory_high_watermark: Maximum memory usage in MB
        total_operations: Total number of cache operations performed
        hit_rate: Cache hit rate (hits / total_operations)
        eviction_rate: Number of evictions per minute
        current_size: Current number of items in cache
        max_size: Maximum allowed items in cache
        memory_pressure_level: Current memory pressure level (0-3)
        last_metrics_update: Timestamp of last metrics update
    """
    # Basic metrics
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    ttl_evictions: int = 0
    policy_evictions: int = 0
    
    # Timing metrics
    last_eviction_time: float = 0
    last_metrics_update: float = 0
    
    # Performance metrics
    avg_eviction_duration: float = 0
    memory_high_watermark: float = 0
    total_operations: int = 0
    hit_rate: float = 0.0
    eviction_rate: float = 0.0
    
    # Cache state
    current_size: int = 0
    max_size: int = 0
    memory_pressure_level: int = 0  # 0-3 (NONE to HIGH)
    
    def to_dict(self) -> dict:
        """Convert metrics to a dictionary for easy serialization."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'ttl_evictions': self.ttl_evictions,
            'policy_evictions': self.policy_evictions,
            'last_eviction_time': self.last_eviction_time,
            'avg_eviction_duration': self.avg_eviction_duration,
            'memory_high_watermark_mb': self.memory_high_watermark,
            'total_operations': self.total_operations,
            'hit_rate': self.hit_rate,
            'eviction_rate': self.eviction_rate,
            'current_size': self.current_size,
            'max_size': self.max_size,
            'memory_pressure_level': self.memory_pressure_level,
            'last_metrics_update': self.last_metrics_update,
        }
    
    def __str__(self) -> str:
        """Return a human-readable string representation of metrics."""
        return (
            f"CacheMetrics(hits={self.hits}, misses={self.misses}, "
            f"hit_rate={self.hit_rate:.1%}, size={self.current_size}/{self.max_size}, "
            f"evictions={self.evictions}, memory={self.memory_high_watermark:.1f}MB, "
            f"pressure={'NONE' if self.memory_pressure_level == 0 else 'LOW' if self.memory_pressure_level == 1 else 'MEDIUM' if self.memory_pressure_level == 2 else 'HIGH'})"
        )

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
        self.enable_memory_pressure = enable_memory_pressure and PSUTIL_AVAILABLE
        self.memory_pressure_threshold = memory_pressure_threshold
        self.batch_size = batch_size
        self.sample_size = sample_size
        
        self._cache: Dict[str, Any] = OrderedDict()
        self._lock = asyncio.Lock()
        self._eviction_task: Optional[asyncio.Task] = None
        self._running = False
        self._closed = False
        
        # Track vector IDs for deletion
        self._prompt_to_vector_ids: Dict[str, List[str]] = {}
        
        # Initialize metrics and memory tracking
        self.memory_high_watermark = 0.0
        self._metrics = CacheMetrics()
        self._metrics.max_size = max_size
        self._metrics.last_metrics_update = time.time()
        self._metrics.memory_high_watermark = 0.0
        self._metrics_lock = asyncio.Lock()
        self._last_metrics_update = 0
        self._last_eviction_count = 0
        
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
            
        # Start metrics update task
        self._start_metrics_task()
    
    async def _update_metrics(self) -> None:
        """Update metrics with current cache state."""
        current_time = time.time()
        
        async with self._metrics_lock:
            # Update basic metrics
            self._metrics.current_size = len(self._cache)
            self._metrics.memory_pressure_level = await self._check_memory_pressure()
            
            # Calculate hit rate
            total_ops = self._metrics.hits + self._metrics.misses
            self._metrics.hit_rate = self._metrics.hits / total_ops if total_ops > 0 else 0.0
            
            # Calculate eviction rate (per minute)
            time_since_last = current_time - self._last_metrics_update
            if time_since_last > 0:
                evictions_since_last = self._metrics.evictions - self._last_eviction_count
                self._metrics.eviction_rate = (evictions_since_last / time_since_last) * 60
                self._last_eviction_count = self._metrics.evictions
            
            self._last_metrics_update = current_time
            self._metrics.last_metrics_update = current_time
    
    def _start_metrics_task(self) -> None:
        """Start the background metrics update task."""
        if not hasattr(self, '_metrics_task') or self._metrics_task.done():
            self._metrics_task = asyncio.create_task(self._metrics_loop())
    
    async def _metrics_loop(self) -> None:
        """Background task that updates metrics at regular intervals."""
        while self._running:
            try:
                await self._update_metrics()
                await asyncio.sleep(5)  # Update every 5 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in metrics loop: {e}", exc_info=True)
                # Don't raise here, just log and continue
                await asyncio.sleep(30)  # Back off on errors
    
    async def get_metrics(self) -> CacheMetrics:
        """
        Get a snapshot of the current cache metrics.
        
        Returns:
            CacheMetrics: A snapshot of the current metrics
        """
        await self._update_metrics()  # Ensure metrics are up to date
        return self._metrics
    
    def get_metrics_sync(self) -> CacheMetrics:
        """
        Get a snapshot of the current cache metrics (synchronous version).
        
        Note: This may return slightly stale data as it doesn't force an update.
        
        Returns:
            CacheMetrics: A snapshot of the current metrics
        """
        return self._metrics
    
    async def _background_eviction_loop(self) -> None:
        """Background task that runs eviction at regular intervals."""
        while self._running:
            try:
                # Run one eviction cycle
                evicted = await self._run_eviction_cycle()
                
                # Log if we're under memory pressure
                if await self._check_memory_pressure():
                    logger.warning(
                        f"Memory pressure detected during eviction cycle. "
                        f"Evicted {evicted} items."
                    )
                
                # Wait for the next cycle, but allow for early wakeup on shutdown
                for _ in range(int(self.eviction_interval * 10)):
                    if not self._running:
                        return
                    await asyncio.sleep(0.1)
                    
            except asyncio.CancelledError:
                logger.debug("Background eviction task was cancelled")
                return
            except Exception as e:
                logger.error(f"Error in background eviction task: {e}", exc_info=True)
                # Re-raise as a cache operation error to allow for monitoring
                # but don't let it kill the task.
                await asyncio.sleep(min(60, self.eviction_interval))  # Back off on errors
    
    def _start_eviction_task(self) -> None:
        """Start the background eviction task if not already running."""
        if self._eviction_task is None or self._eviction_task.done():
            self._running = True
            self._eviction_task = asyncio.create_task(
                self._background_eviction_loop(),
                name=f"InMemoryStore-eviction-{id(self)}"
            )
            logger.debug("Started background eviction task")
    
    def _stop_eviction_task(self) -> None:
        """Stop the background eviction task."""
        if self._eviction_task is not None and not self._eviction_task.done():
            self._running = False
            self._eviction_task.cancel()
            logger.debug("Stopped background eviction task")
    
    async def close(self) -> None:
        """Clean up resources."""
        if self._closed:
            return
            
        self._stop_eviction_task()
        self._closed = True
        logger.debug("InMemoryStore closed")
    
    def __del__(self) -> None:
        """Ensure resources are cleaned up on garbage collection."""
        if not self._closed:
            logger.warning(
                "InMemoryStore was not properly closed. "
                "Make sure to call close() or use async with."
            )
            self._stop_eviction_task()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    class MemoryPressureLevel:
        """Enumeration of memory pressure levels."""
        NONE = 0
        LOW = 1
        MEDIUM = 2
        HIGH = 3
    
    async def _get_memory_usage(self) -> tuple[float, float]:
        """
        Get current memory usage metrics.
        
        Returns:
            Tuple of (process_rss_mb, system_available_percent)
        """
        if not PSUTIL_AVAILABLE:
            return 0.0, 0.0
            
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            rss_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
            
            # Update high watermark
            self.memory_high_watermark = max(self.memory_high_watermark, rss_mb)
            
            # Get system memory info
            system_mem = psutil.virtual_memory()
            available_percent = (system_mem.available / system_mem.total) * 100
            
            return rss_mb, available_percent
            
        except Exception as e:
            logger.warning(f"Failed to get memory usage: {e}")
            raise CacheOperationError("Failed to get memory usage", original_exception=e) from e
    
    async def _check_memory_pressure(self) -> int:
        """
        Check current memory pressure level.
        
        Returns:
            int: The current memory pressure level (0-3)
        """
        if not self.enable_memory_pressure:
            return self.MemoryPressureLevel.NONE
            
        try:
            if not PSUTIL_AVAILABLE:
                # Fallback to cache size based pressure if psutil not available
                cache_size = len(self._cache)
                if cache_size > self.max_size * 0.9:
                    return self.MemoryPressureLevel.HIGH
                elif cache_size > self.max_size * 0.7:
                    return self.MemoryPressureLevel.MEDIUM
                elif cache_size > self.max_size * 0.5:
                    return self.MemoryPressureLevel.LOW
                return self.MemoryPressureLevel.NONE
                
            rss_mb, available_percent = await self._get_memory_usage()
            
            # Update memory high watermark
            if rss_mb > self.memory_high_watermark:
                self.memory_high_watermark = rss_mb
                self._metrics.memory_high_watermark = rss_mb
            
            # Check system-wide memory pressure first
            if available_percent < 5:  # Less than 5% available
                return self.MemoryPressureLevel.HIGH
            elif available_percent < 15:
                return self.MemoryPressureLevel.MEDIUM
            elif available_percent < 25:
                return self.MemoryPressureLevel.LOW
                
            # Check process-specific memory usage
            process = psutil.Process()
            process_mem_percent = process.memory_percent()
            
            if process_mem_percent > 90:
                return self.MemoryPressureLevel.HIGH
            elif process_mem_percent > 70:
                return self.MemoryPressureLevel.MEDIUM
            elif process_mem_percent > 50:
                return self.MemoryPressureLevel.LOW
                
            return self.MemoryPressureLevel.NONE
            
        except Exception as e:
            logger.warning(f"Failed to check memory pressure: {e}")
            # Fallback to cache size based pressure on error
            cache_size = len(self._cache)
            if cache_size > self.max_size * 0.9:
                return self.MemoryPressureLevel.HIGH
            elif cache_size > self.max_size * 0.7:
                return self.MemoryPressureLevel.MEDIUM
            elif cache_size > self.max_size * 0.5:
                return self.MemoryPressureLevel.LOW
            return self.MemoryPressureLevel.NONE
    
    async def _get_eviction_batch_size(self, pressure_level: int) -> int:
        """
        Determine how many items to evict based on memory pressure.
        
        Args:
            pressure_level: Current memory pressure level
            
        Returns:
            Number of items to evict in this cycle
        """
        if pressure_level == self.MemoryPressureLevel.HIGH:
            return max(100, len(self._cache) // 10)  # Be aggressive
        elif pressure_level == self.MemoryPressureLevel.MEDIUM:
            return max(50, len(self._cache) // 20)
        elif pressure_level == self.MemoryPressureLevel.LOW:
            return max(10, len(self._cache) // 50)
        return 0  # No eviction needed

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
        """
        Evict expired items using adaptive sampling and batching.
        
        This method uses several optimizations:
        1. Adaptive sampling based on cache size and TTL distribution
        2. Batched eviction to prevent long blocking
        3. Early termination if no expired items found in sample
        4. Dynamic adjustment of sample size based on hit rate
        """
        if not self.enable_ttl or not self._cache:
            return 0

        start_time = time.time()
        total_evicted = 0
        
        # Calculate adaptive sample size based on cache size and TTL distribution
        sample_size = min(
            max(100, len(self._cache) // 100),  # Between 100 and 1% of cache size
            self.sample_size
        )
        
        # Get a sample of items to check for expiration
        sample = await self._sample_items(min(sample_size, len(self._cache)))
        current_time = time.time()
        
        # Find expired items in the sample
        expired = []
        for key, value in sample:
            if not isinstance(value, dict):
                continue
                
            # Support both 'timestamp' and 'expires_at' formats
            if 'expires_at' in value:
                is_expired = current_time > value['expires_at']
            else:
                is_expired = (current_time - value.get('timestamp', 0)) > self.ttl_seconds
                
            if is_expired:
                expired.append(key)
        
        # If we found expired items in the sample, do a full scan with batching
        if expired:
            # Evict the expired items we found in the sample
            evicted = await self._evict_in_batches(expired)
            total_evicted += evicted
            
            # Estimate total expired items based on sample
            expired_ratio = len(expired) / len(sample)
            estimated_total_expired = int(len(self._cache) * expired_ratio)
            
            # If we have a significant number of expired items, do a full scan
            if estimated_total_expired > len(expired) * 2:  # More than double what we found in sample
                logger.debug(
                    f"High TTL expiration rate detected ({(expired_ratio*100):.1f}%). "
                    f"Performing full scan..."
                )
                
                # Process in batches to prevent long blocking
                batch_size = min(1000, max(100, len(self._cache) // 10))
                items = list(self._cache.items())
                
                for i in range(0, len(items), batch_size):
                    batch = items[i:i + batch_size]
                    batch_expired = []
                    
                    for key, value in batch:
                        if not isinstance(value, dict):
                            continue
                            
                        if 'expires_at' in value:
                            is_expired = current_time > value['expires_at']
                        else:
                            is_expired = (current_time - value.get('timestamp', 0)) > self.ttl_seconds
                            
                        if is_expired:
                            batch_expired.append(key)
                    
                    # Evict this batch of expired items
                    if batch_expired:
                        evicted = await self._evict_in_batches(batch_expired)
                        total_evicted += evicted
                        
                        # Small sleep to prevent starving other tasks
                        await asyncio.sleep(0.001)
        
        try:
            # Update metrics
            duration = (time.time() - start_time) * 1000  # ms
            if total_evicted > 0:
                logger.debug(
                    f"TTL eviction: checked {len(sample)} items, evicted {total_evicted} "
                    f"in {duration:.2f}ms"
                )
                
                # Update TTL sampling rate if we found many expired items
                expiration_rate = len(expired) / len(sample) if sample else 0
                if expiration_rate > 0.1:  # More than 10% expired in sample
                    self.sample_size = min(
                        self.sample_size * 2,  # Double sample size
                        10000  # But cap at 10k
                    )
                    logger.debug(f"Increased TTL sample size to {self.sample_size}")
                
                # Get current memory pressure level
                pressure_level = await self._check_memory_pressure()
                
                # Map memory pressure level to string name
                pressure_level_names = {
                    0: 'NONE',
                    1: 'LOW',
                    2: 'MEDIUM',
                    3: 'HIGH'
                }
                pressure_name = pressure_level_names.get(pressure_level, 'UNKNOWN')
                
                logger.info(
                    f"Eviction cycle: evicted {total_evicted} items "
                    f"in {duration:.2f}ms (pressure: {pressure_name})"
                )

                # Check if we need policy-based eviction
                needs_eviction = (
                    len(self._cache) > self.max_size or 
                    pressure_level > self.MemoryPressureLevel.NONE
                )

                if needs_eviction:
                    # Calculate how many items to evict based on pressure
                    if pressure_level > self.MemoryPressureLevel.NONE:
                        batch_size = await self._get_eviction_batch_size(pressure_level)
                        target_size = max(0, len(self._cache) - batch_size)
                    else:
                        target_size = self.max_size
                    
                    # Ensure we don't go below minimum cache size
                    target_size = max(0, min(target_size, self.max_size))
                    
                    if target_size < len(self._cache):
                        evicted = await self.eviction_policy.apply(
                            self._cache, 
                            max(0, len(self._cache) - target_size)
                        )
                        total_evicted += evicted
                        
                        if evicted > 0:
                            logger.info(
                                f"Evicted {evicted} items due to memory pressure "
                                f"(new size: {len(self._cache)}/{self.max_size})"
                            )

                # Update metrics
                if total_evicted > 0:
                    duration = (time.time() - start_time) * 1000  # ms
                    async with self._metrics_lock:
                        self._metrics.last_eviction_time = time.time()
                        self._metrics.evictions += total_evicted
                        self._metrics.policy_evictions += total_evicted
                        self._metrics.avg_eviction_duration = (
                            (self._metrics.avg_eviction_duration * 0.9) + (duration * 0.1)
                        )
                        self._metrics.memory_high_watermark = max(
                            self._metrics.memory_high_watermark, 
                            self._metrics.current_size
                        )
        
        except Exception as e:
            logger.error(f"Error during eviction cycle: {e}", exc_info=True)
        
        return total_evicted

    async def _run_eviction_cycle(self) -> int:
        """
        Run a single eviction cycle.
        
        Returns:
            int: Number of items evicted in this cycle
        """
        total_evicted = 0
        
        # Check memory pressure level
        pressure_level = await self._check_memory_pressure()
        
        # Evict expired items if TTL is enabled
        if self.enable_ttl:
            total_evicted += await self._evict_expired_items()
        
        # Check if we need to evict more items due to memory pressure or size limits
        needs_eviction = (
            len(self._cache) > self.max_size or 
            pressure_level > self.MemoryPressureLevel.NONE
        )
        
        if needs_eviction:
            # Calculate how many items to evict based on pressure
            if pressure_level > self.MemoryPressureLevel.NONE:
                batch_size = await self._get_eviction_batch_size(pressure_level)
                target_size = max(0, len(self._cache) - batch_size)
            else:
                target_size = self.max_size
            
            # Ensure we don't go below minimum cache size
            target_size = max(0, min(target_size, self.max_size))
            
            if target_size < len(self._cache):
                evicted = await self.eviction_policy.apply(
                    self._cache, 
                    max(0, len(self._cache) - target_size)
                )
                total_evicted += evicted
                
                if evicted > 0:
                    logger.info(
                        f"Evicted {evicted} items due to memory pressure "
                        f"(new size: {len(self._cache)}/{self.max_size})"
                    )
        
        # Update metrics
        if total_evicted > 0:
            async with self._metrics_lock:
                self._metrics.evictions += total_evicted
                self._metrics.policy_evictions += total_evicted
                self._metrics.last_eviction_time = time.time()
        
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
            
        self._stop_eviction_task()
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

    async def delete(self, key: str, mode: DeletionMode = DeletionMode.DIRECT) -> bool:
        """
        Delete an item from the cache.
        
        Args:
            key: Cache key to delete
            mode: Deletion mode (direct, semantic, or tombstone)
            
        Returns:
            bool: True if the key existed and was deleted, False otherwise
        """
        if mode == DeletionMode.TOMBSTONE:
            return await self.tombstone(key)
        
        if key in self._cache:
            async with self._metrics_lock:
                del self._cache[key]
                # Also remove vector ID mapping
                self._prompt_to_vector_ids.pop(key, None)
                self._metrics.current_size = len(self._cache)
            return True
        return False
    

    
    async def tombstone(self, key: str) -> bool:
        """
        Mark a key as deleted (tombstoning) without removing it from storage.
        
        Args:
            key: The key to tombstone
            
        Returns:
            bool: True if the key was found and tombstoned, False otherwise
        """
        if key in self._cache:
            value = self._cache[key]
            if isinstance(value, dict):
                # Add tombstone flag to existing dict
                value['tombstoned'] = True
                value['tombstoned_at'] = time.time()
            else:
                # Convert to dict format with tombstone flag
                self._cache[key] = {
                    'response': value,
                    'tombstoned': True,
                    'tombstoned_at': time.time(),
                    'timestamp': time.time()
                }
            return True
        return False
    
    async def is_tombstoned(self, key: str) -> bool:
        """
        Check if a key is tombstoned.
        
        Args:
            key: The key to check
            
        Returns:
            bool: True if the key is tombstoned, False otherwise
        """
        if key in self._cache:
            value = self._cache[key]
            if isinstance(value, dict):
                return value.get('tombstoned', False)
        return False
    
    async def get_vector_ids(self, prompt: str) -> List[str]:
        """
        Get vector IDs associated with a prompt.
        
        Args:
            prompt: The prompt to get vector IDs for
            
        Returns:
            List[str]: List of vector IDs associated with the prompt
        """
        return self._prompt_to_vector_ids.get(prompt, [])
    
    async def purge_tombstones(self) -> int:
        """
        Permanently remove all tombstoned keys from storage.
        
        Returns:
            int: Number of tombstoned keys that were purged
        """
        purged_count = 0
        keys_to_purge = []
        
        # Find all tombstoned keys
        for key, value in self._cache.items():
            if isinstance(value, dict) and value.get('tombstoned', False):
                keys_to_purge.append(key)
        
        # Remove tombstoned keys
        for key in keys_to_purge:
            del self._cache[key]
            # Also remove vector ID mapping
            self._prompt_to_vector_ids.pop(key, None)
            purged_count += 1
        
        # Update metrics
        if purged_count > 0:
            async with self._metrics_lock:
                self._metrics.current_size = len(self._cache)
        
        return purged_count

    async def get_exact(self, prompt: str) -> Optional[str]:
        """
        Get a cached response if it exists and is not expired.

        Args:
            prompt: The prompt to look up in the cache.

        Returns:
            The cached response if found and not expired, None otherwise.
        """
        async with self._metrics_lock:
            if prompt in self._cache:
                value = self._cache[prompt]
                if isinstance(value, dict):
                    # Check if item is tombstoned
                    if value.get('tombstoned', False):
                        self._metrics.misses += 1
                        return None
                    
                    # Check for TTL expiration
                    if 'expires_at' in value and time.time() > value['expires_at']:
                        del self._cache[prompt]
                        self._metrics.misses += 1
                        self._metrics.ttl_evictions += 1
                        return None
                    
                    # Move to end (most recently used)
                    self._cache.move_to_end(prompt)
                    self._metrics.hits += 1
                    return value.get('response')
                
                # For backward compatibility with non-dict values
                self._cache.move_to_end(prompt)
                self._metrics.hits += 1
                return value
            
            self._metrics.misses += 1
            return None

    async def get(self, key: str) -> Optional[Any]:
        """
        Get an item from the cache.
        
        Updates hit/miss metrics automatically.
        """
        async with self._metrics_lock:
            if key in self._cache:
                value = self._cache[key]
                if isinstance(value, dict):
                    # Check if item is tombstoned
                    if value.get('tombstoned', False):
                        self._metrics.misses += 1
                        return None
                    
                    # Check for TTL expiration
                    if 'expires_at' in value and time.time() > value['expires_at']:
                        del self._cache[key]
                        self._metrics.misses += 1
                        self._metrics.ttl_evictions += 1
                        return None
                    
                    # Move to end (most recently used)
                    self._cache.move_to_end(key)
                    self._metrics.hits += 1
                    return value.get('response')
                
                # For backward compatibility with non-dict values
                self._cache.move_to_end(key)
                self._metrics.hits += 1
                return value
            
            self._metrics.misses += 1
            return None

    async def add(self, prompt: str, response: str, vector_ids: Optional[List[str]] = None) -> None:
        if self.enable_ttl:
            self._cache[prompt] = {
                'response': response,
                'timestamp': time.time(),
                'expires_at': time.time() + self.ttl_seconds
            }
        else:
            self._cache[prompt] = response
        self._cache.move_to_end(prompt)
        
        # Track vector IDs if provided
        if vector_ids:
            self._prompt_to_vector_ids[prompt] = vector_ids
        
        # Trigger eviction if we're over size limit
        if len(self._cache) > self.max_size * 1.1:  # 10% over limit
            await self._run_eviction_cycle()

    async def enforce_limits(self, resource_limits: Any) -> None:
        """Enforce cache limits asynchronously."""
        evicted_count = await self._run_eviction_cycle()
        if evicted_count > 0:
            logger.info(f"Enforced limits: evicted {evicted_count} items")

    async def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.

        Args:
            key: The key to delete
            
        Returns:
            bool: True if the key was found and deleted, False otherwise
        """
        if key in self._cache:
            del self._cache[key]
            # Update metrics
            await self._update_metrics()
            return True
        return False

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
            # This is a non-critical error, so we just log it.
            
        return stats