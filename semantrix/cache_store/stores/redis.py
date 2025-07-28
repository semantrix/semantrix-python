"""
Redis cache store implementation for Semantrix.

Provides a distributed, persistent cache using Redis.
"""

import json
import time
from typing import Optional, Any, Dict
from semantrix.cache_store.base import BaseCacheStore, EvictionPolicy, NoOpEvictionPolicy

class RedisCacheStore(BaseCacheStore):
    """
    Redis-based cache store implementation.

    Args:
        redis_client: An instance of redis.Redis, redis.cluster.RedisCluster, or any compatible client.
        key_prefix: Prefix for cache keys (default: 'semantrix:')
        eviction_policy: Eviction policy (defaults to NoOpEvictionPolicy since Redis handles eviction)
    """
    def __init__(self, 
                 redis_client: Any,  # Accepts redis.Redis or compatible
                 key_prefix: str = "semantrix:",
                 eviction_policy: Optional[EvictionPolicy] = None):
        self.redis = redis_client
        self.key_prefix = key_prefix
        self.eviction_policy = eviction_policy or NoOpEvictionPolicy()

    def _get_key(self, prompt: str) -> str:
        return f"{self.key_prefix}{hash(prompt)}"

    def get_exact(self, prompt: str) -> Optional[str]:
        key = self._get_key(prompt)
        value = self.redis.get(key)
        if value is not None:
            try:
                data = json.loads(value.decode('utf-8'))
                return data.get('response')
            except (json.JSONDecodeError, KeyError, AttributeError):
                return value.decode('utf-8')
        return None

    def add(self, prompt: str, response: str) -> None:
        key = self._get_key(prompt)
        data = {
            'response': response,
            'timestamp': time.time(),
            'prompt_hash': hash(prompt)
        }
        self.redis.set(key, json.dumps(data))

    def enforce_limits(self, resource_limits: Any) -> None:
        # Redis handles eviction automatically
        pass

    def clear(self) -> None:
        pattern = f"{self.key_prefix}*"
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)

    def size(self) -> int:
        pattern = f"{self.key_prefix}*"
        keys = self.redis.keys(pattern)
        return len(keys)

    def get_eviction_policy(self) -> EvictionPolicy:
        return self.eviction_policy

    def get_stats(self) -> dict:
        pattern = f"{self.key_prefix}*"
        keys = self.redis.keys(pattern)
        return {
            'size': len(keys),
            'redis_info': self.redis.info(),
            'eviction_policy': type(self.eviction_policy).__name__,
            'key_prefix': self.key_prefix
        } 