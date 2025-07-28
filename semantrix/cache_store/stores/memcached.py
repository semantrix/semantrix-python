"""
Memcached cache store implementation for Semantrix.

Provides a fast, distributed, ephemeral cache using Memcached.
"""

from typing import Optional, Any
from semantrix.cache_store.base import BaseCacheStore, EvictionPolicy, NoOpEvictionPolicy
import json
import time

class MemcachedCacheStore(BaseCacheStore):
    """
    Memcached-based cache store implementation.

    Args:
        memcached_client: An instance of pymemcache.client.base.Client or compatible.
        key_prefix: Prefix for cache keys (default: 'semantrix:')
        eviction_policy: Eviction policy (defaults to NoOpEvictionPolicy since Memcached handles eviction)
    """
    def __init__(self, 
                 memcached_client: Any,  # Accepts pymemcache client or compatible
                 key_prefix: str = "semantrix:",
                 eviction_policy: Optional[EvictionPolicy] = None):
        self.client = memcached_client
        self.key_prefix = key_prefix
        self.eviction_policy = eviction_policy or NoOpEvictionPolicy()

    def _get_key(self, prompt: str) -> str:
        return f"{self.key_prefix}{hash(prompt)}"

    def get_exact(self, prompt: str) -> Optional[str]:
        key = self._get_key(prompt)
        value = self.client.get(key)
        if value is not None:
            try:
                data = json.loads(value.decode('utf-8'))
                return data.get('response')
            except (json.JSONDecodeError, KeyError, AttributeError):
                return value.decode('utf-8') if hasattr(value, 'decode') else value
        return None

    def add(self, prompt: str, response: str) -> None:
        key = self._get_key(prompt)
        data = {
            'response': response,
            'timestamp': time.time(),
            'prompt_hash': hash(prompt)
        }
        self.client.set(key, json.dumps(data))

    def enforce_limits(self, resource_limits: Any) -> None:
        # Memcached handles eviction automatically
        pass

    def clear(self) -> None:
        # Memcached does not support key pattern deletion natively
        # This will flush all keys (be careful in shared environments)
        self.client.flush_all()

    def size(self) -> int:
        # Memcached does not provide a direct way to count keys
        # This is a limitation; return None or -1 to indicate unknown
        return -1

    def get_eviction_policy(self) -> EvictionPolicy:
        return self.eviction_policy

    def get_stats(self) -> dict:
        stats = self.client.stats()
        return {
            'stats': stats,
            'eviction_policy': type(self.eviction_policy).__name__,
            'key_prefix': self.key_prefix
        } 