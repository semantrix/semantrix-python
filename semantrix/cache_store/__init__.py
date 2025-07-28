"""
Cache store package for Semantrix.

This package contains all cache store base classes, strategies, policies, and store implementations.
"""

# Base classes
from .base import BaseCacheStore, EvictionStrategy, EvictionPolicy

# Strategies
from .strategies import (
    NoOpEvictionStrategy,
    LRUEvictionStrategy,
    FIFOEvictionStrategy,
    TTLStrategy,
    LFUEvictionStrategy
)

# Eviction Policies
from .eviction_policies import (
    StrategyBasedEvictionPolicy,
    NoOpEvictionPolicy,
    AdaptiveEvictionPolicy
)

# Store Implementations (from stores subpackage)
from .stores import InMemoryStore, RedisCacheStore, MemcachedCacheStore

__all__ = [
    # Base classes
    "BaseCacheStore",
    "EvictionStrategy",
    "EvictionPolicy",
    # Strategies
    "NoOpEvictionStrategy",
    "LRUEvictionStrategy",
    "FIFOEvictionStrategy",
    "TTLStrategy",
    "LFUEvictionStrategy",
    # Eviction Policies
    "StrategyBasedEvictionPolicy",
    "NoOpEvictionPolicy",
    "AdaptiveEvictionPolicy",
    # Stores
    "InMemoryStore",
    "RedisCacheStore",
    "MemcachedCacheStore"
]



