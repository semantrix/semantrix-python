"""
In-memory cache store implementation for Semantrix.

Provides a fast, ephemeral cache using Python's OrderedDict.
"""

import time
from typing import Optional, Any
from collections import OrderedDict
from semantrix.cache_store.base import BaseCacheStore, EvictionPolicy
from semantrix.cache_store.strategies import LRUEvictionStrategy, TTLStrategy
from semantrix.cache_store.eviction_policies import StrategyBasedEvictionPolicy, NoOpEvictionPolicy

class InMemoryStore(BaseCacheStore):
    """
    In-memory cache store with eviction policies.

    Args:
        max_size: Maximum number of items to store
        eviction_policy: Custom eviction policy (defaults to LRU strategy)
        enable_ttl: Enable time-to-live functionality
        ttl_seconds: TTL duration in seconds
    """
    def __init__(self, 
                 max_size: int = 10_000,
                 eviction_policy: Optional[EvictionPolicy] = None,
                 enable_ttl: bool = False,
                 ttl_seconds: int = 3600):
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

    def get_exact(self, prompt: str) -> Optional[str]:
        if prompt in self.cache:
            value = self.cache[prompt]
            # Handle TTL if enabled
            if self.enable_ttl and isinstance(value, dict):
                if hasattr(self.eviction_policy, 'strategy'):
                    ttl_seconds = getattr(self.eviction_policy.strategy, 'ttl_seconds', 3600)
                else:
                    ttl_seconds = 3600
                if time.time() - value['timestamp'] > ttl_seconds:
                    del self.cache[prompt]
                    return None
                response = value['response']
            else:
                response = value
            self.cache.move_to_end(prompt)
            return response
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

    def enforce_limits(self, resource_limits: Any) -> None:
        evicted_count = self.eviction_policy.apply(self.cache, self.max_size)
        if evicted_count > 0:
            print(f"Evicted {evicted_count} items from cache")

    def clear(self) -> None:
        self.cache.clear()

    def size(self) -> int:
        return len(self.cache)

    def get_eviction_policy(self) -> EvictionPolicy:
        return self.eviction_policy

    def get_stats(self) -> dict:
        return {
            'size': self.size(),
            'max_size': self.max_size,
            'eviction_policy': type(self.eviction_policy).__name__,
            'enable_ttl': self.enable_ttl,
            'evicted_count': getattr(self.eviction_policy, 'evicted_count', 0)
        } 