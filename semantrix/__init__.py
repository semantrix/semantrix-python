"""
Semantrix - Semantic Caching Library
====================================

A high-performance semantic caching library for AI applications.
"""

__version__ = "0.1.0"

from .core.cache import Semantrix
from .client import SemantrixClient
from .utils.resource_limits import ResourceLimits
from .utils.redis_helpers import create_redis_cache_store
from .cache_store.stores.redis import RedisCacheStore
from .cache_store.stores.memcached import MemcachedCacheStore

# Export base classes for custom implementations
from .embedding.base import BaseEmbedder
from .embedding.embedders.sentence_transformer_embedder import SentenceTransformerEmbedder as Embedder
from .vector_store.base import BaseVectorStore
from .vector_store.stores.faiss import FAISSVectorStore
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