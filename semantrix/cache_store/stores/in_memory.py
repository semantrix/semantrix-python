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
import psutil
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Deque

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
        memory_pressure_threshold: Memory usage threshold (0-1) to trigger aggressive eviction
        max_batch_size: Maximum items to evict in one batch
        ttl_sample_size: Number of items to sample for TTL checks
    """
    def __init__(self, 
                 max_size: int = 10_000,
                 eviction_policy: Optional[EvictionPolicy] = None,
                 enable_ttl: bool = False,
                 ttl_seconds: int = 3600,
                 eviction_interval: float = 60.0,
                 memory_pressure_threshold: float = 0.8,
                 max_batch_size: int = 1000,
                 ttl_sample_size: int = 1000):
        self.cache = OrderedDict()
        self.max_size = max_size
        
        # Set eviction policy
        if eviction_policy is not None:
            self.eviction_policy = eviction_policy
        elif enable_ttl:
            self.eviction_policy = StrategyBasedEvictionPolicy(TTLStrategy(ttl_seconds))
        else:
            self.eviction_policy = StrategyBasedEvictionPolicy(LRUEvictionStrategy())
        self.enable_ttl = enable_ttl
        
        # Initialize metrics
        self.metrics = CacheMetrics()
        
        # Initialize memory pressure detection
        self.memory_pressure_threshold = memory_pressure_threshold
        self.memory_high_watermark = 0
        
        # Initialize background eviction
        self.eviction_interval = eviction_interval
        self.max_batch_size = max_batch_size
        self.ttl_sample_size = ttl_sample_size
        self.eviction_task = None
        
        # Start background eviction
        self.start_eviction_task()

    async def _check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        try:
            mem = psutil.virtual_memory()
            current_usage = mem.percent / 100
            self.memory_high_watermark = max(self.memory_high_watermark, current_usage)
            return current_usage > self.memory_pressure_threshold
        except Exception as e:
            logger.warning(f"Failed to check memory pressure: {e}")
            return False

    async def _sample_items(self, sample_size: int) -> List[Tuple[str, Any]]:
        """Get a random sample of cache items."""
        if len(self.cache) <= sample_size:
            return list(self.cache.items())
        return random.sample(list(self.cache.items()), sample_size)

    async def _evict_in_batches(self, keys: List[str]) -> int:
        """Evict keys in batches to prevent long blocking."""
        evicted = 0
        for i in range(0, len(keys), self.max_batch_size):
            batch = keys[i:i + self.max_batch_size]
            for key in batch:
                if key in self.cache:
                    self.cache.pop(key, None)
                    evicted += 1
            # Small sleep to prevent starving other tasks
            await asyncio.sleep(0.001)
        return evicted

    async def _evict_expired_items(self) -> int:
        """Evict expired items using sampling."""
        if not self.enable_ttl or not self.cache:
            return 0

        start_time = time.time()
        sample = await self._sample_items(min(self.ttl_sample_size, len(self.cache)))
        current_time = time.time()
        
        expired = [
            k for k, v in sample
            if isinstance(v, dict) and 
               current_time - v.get('timestamp', 0) > getattr(
                   getattr(self.eviction_policy, 'strategy', None), 
                   'ttl_seconds', 
                   3600
               )
        ]
        
        evicted = await self._evict_in_batches(expired)
        
        duration = (time.time() - start_time) * 1000  # ms
        logger.debug(f"TTL eviction: checked {len(sample)} items, evicted {evicted} in {duration:.2f}ms")
        
        self.metrics.ttl_evictions += evicted
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
                len(self.cache) > self.max_size or 
                under_pressure
            )

            if needs_eviction:
                evicted = await self.eviction_policy.apply(
                    self.cache, 
                    max(0, self.max_size - total_evicted)
                )
                total_evicted += evicted
                self.metrics.policy_evictions += evicted

            # Update metrics
            if total_evicted > 0:
                duration = (time.time() - start_time) * 1000  # ms
                self.metrics.last_eviction_time = time.time()
                self.metrics.avg_eviction_duration = (
                    0.8 * self.metrics.avg_eviction_duration + 
                    0.2 * duration
                )
                self.metrics.evictions += total_evicted
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

    def start_eviction_task(self):
        """Start background eviction tasks."""
        if not hasattr(self, '_running') or not self._running:
            self._running = True
            self.eviction_task = asyncio.create_task(self._eviction_loop())
            logger.info("Started background eviction task")

    async def stop_eviction_task(self):
        """Stop background tasks."""
        self._running = False
        if self.eviction_task and not self.eviction_task.done():
            self.eviction_task.cancel()
            try:
                await self.eviction_task
            except asyncio.CancelledError:
                pass
            logger.info("Stopped background eviction task")

    def get_exact(self, prompt: str) -> Optional[str]:
        if prompt in self.cache:
            value = self.cache[prompt]
            # Handle TTL if enabled
            if self.enable_ttl and isinstance(value, dict):
                ttl_seconds = getattr(
                    getattr(self.eviction_policy, 'strategy', None),
                    'ttl_seconds',
                    3600
                )
                if time.time() - value['timestamp'] > ttl_seconds:
                    del self.cache[prompt]
                    self.metrics.misses += 1
                    return None
                response = value['response']
            else:
                response = value
            self.cache.move_to_end(prompt)
            self.metrics.hits += 1
            return response
        self.metrics.misses += 1
        return None

    def add(self, prompt: str, response: str) -> None:
        if self.enable_ttl:
            self.cache[prompt] = {
                'response': response,
                'timestamp': time.time()
            }
        else:
            self.cache[prompt] = response
        self.cache.move_to_end(prompt)
        
        # Trigger eviction if we're over size limit
        if len(self.cache) > self.max_size * 1.1:  # 10% over limit
            asyncio.create_task(self._run_eviction_cycle())

    async def enforce_limits(self, resource_limits: Any) -> None:
        """Enforce cache limits asynchronously."""
        evicted_count = await self._run_eviction_cycle()
        if evicted_count > 0:
            logger.info(f"Enforced limits: evicted {evicted_count} items")

    def clear(self) -> None:
        self.cache.clear()

    def size(self) -> int:
        return len(self.cache)

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
            'hits': self.metrics.hits,
            'misses': self.metrics.misses,
            'hit_rate': (
                self.metrics.hits / (self.metrics.hits + self.metrics.misses)
                if (self.metrics.hits + self.metrics.misses) > 0 else 0
            ),
            'total_evictions': self.metrics.evictions,
            'ttl_evictions': self.metrics.ttl_evictions,
            'policy_evictions': self.metrics.policy_evictions,
            'last_eviction_time': datetime.fromtimestamp(
                self.metrics.last_eviction_time
            ).isoformat() if self.metrics.last_eviction_time > 0 else None,
            'avg_eviction_duration_ms': self.metrics.avg_eviction_duration,
            'memory_high_watermark': self.metrics.memory_high_watermark,
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
        
    async def __aenter__(self):
        self.start_eviction_task()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop_eviction_task()
        
    def __del__(self):
        """Ensure background tasks are cleaned up."""
        if hasattr(self, '_running') and self._running and self.eviction_task:
            self.eviction_task.cancel()