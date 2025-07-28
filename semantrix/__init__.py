"""
Semantrix - Semantic Caching Library
====================================

A high-performance semantic caching library for AI applications.
"""

from .core.cache import Semantrix
from .client import SemantrixClient
from .utils.resource_limits import ResourceLimits
from .utils.redis_helpers import create_redis_cache_store
from .cache_store.redis_store import RedisCacheStore
from .cache_store.memcached_store import MemcachedCacheStore

# Export base classes for custom implementations
from .embedding.embedding import BaseEmbedder, Embedder
from .vector_store.vector_store import BaseVectorStore, FAISSVectorStore
from .cache_store import (
    BaseCacheStore, 
    InMemoryStore, 
    EvictionStrategy,
    EvictionPolicy,
    NoOpEvictionStrategy,
    LRUEvictionStrategy,
    FIFOEvictionStrategy,
    TTLStrategy,
    LFUEvictionStrategy,
    StrategyBasedEvictionPolicy,
    NoOpEvictionPolicy,
    AdaptiveEvictionPolicy
)

# Export explain models
from .models.explain import ExplainResult, CacheMatch, create_explain_result

__version__ = "0.1.0"
__all__ = [
    "Semantrix",
    "SemantrixClient", 
    "ResourceLimits",
    "BaseEmbedder",
    "Embedder",
    "BaseVectorStore",
    "FAISSVectorStore",
    "BaseCacheStore",
    "InMemoryStore",
    "EvictionStrategy",
    "EvictionPolicy",
    "NoOpEvictionStrategy",
    "LRUEvictionStrategy",
    "FIFOEvictionStrategy",
    "TTLStrategy",
    "LFUEvictionStrategy",
    "StrategyBasedEvictionPolicy",
    "NoOpEvictionPolicy",
    "AdaptiveEvictionPolicy",
    "ExplainResult",
    "CacheMatch",
    "create_explain_result",
    "create_redis_cache_store",
    "RedisCacheStore",
    "MemcachedCacheStore"
] 