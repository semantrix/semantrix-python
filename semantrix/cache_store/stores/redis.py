"""
Redis cache store implementation for Semantrix.

Provides a distributed, persistent cache using Redis with async support.
"""

import json
import time
import logging
from typing import Optional, Any, Dict, Union

from redis.exceptions import RedisError

from semantrix.cache_store.base import BaseCacheStore, EvictionPolicy, NoOpEvictionPolicy
from semantrix.exceptions import CacheOperationError, ValidationError

# Configure logging
logger = logging.getLogger(__name__)

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

    async def get_exact(self, prompt: str) -> Optional[str]:
        """
        Get a cached response from Redis asynchronously.
        
        Args:
            prompt: The prompt to look up in the cache.
            
        Returns:
            The cached response if found, None otherwise.
        """
        key = self._get_key(prompt)
        try:
            if hasattr(self.redis, 'get') and callable(getattr(self.redis, 'get')):
                # Handle both sync and async Redis clients
                if hasattr(self.redis, 'execute_command'):  # aioredis
                    value = await self.redis.get(key)
                else:  # redis-py
                    value = self.redis.get(key)
                    if hasattr(value, 'decode'):
                        value = value.decode('utf-8')
                
                if value is not None:
                    try:
                        data = json.loads(value)
                        return data.get('response')
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to decode JSON from cache key {key}: {e}")
                        raise ValidationError(f"Corrupt data in cache for key {key}.", original_exception=e) from e
        except RedisError as e:
            logger.error(f"Redis error getting value for key {key}: {e}")
            raise CacheOperationError(f"Failed to get value from Redis for key {key}", original_exception=e) from e
        return None

    async def add(self, prompt: str, response: str) -> None:
        """
        Add a response to the Redis cache asynchronously.
        
        Args:
            prompt: The prompt to cache.
            response: The response to cache.
        """
        key = self._get_key(prompt)
        data = {
            'response': response,
            'timestamp': time.time(),
            'prompt_hash': hash(prompt)
        }
        try:
            if hasattr(self.redis, 'set'):
                if hasattr(self.redis, 'execute_command'):  # aioredis
                    await self.redis.set(key, json.dumps(data))
                else:  # redis-py
                    self.redis.set(key, json.dumps(data))
        except RedisError as e:
            logger.error(f"Redis error adding value for key {key}: {e}")
            raise CacheOperationError(f"Failed to add value to Redis for key {key}", original_exception=e) from e

    async def enforce_limits(self, resource_limits: Any) -> None:
        """
        Enforce cache size limits asynchronously.
        
        Note: Redis handles its own eviction based on maxmemory-policy.
        This is a no-op since Redis manages its own eviction.
        """
        pass
        
    async def delete(self, key: str) -> bool:
        """
        Delete a key from the Redis cache asynchronously.
        
        Args:
            key: The key to delete
            
        Returns:
            bool: True if the key was found and deleted, False otherwise
        """
        cache_key = self._get_key(key)
        try:
            if hasattr(self.redis, 'delete'):
                if hasattr(self.redis, 'execute_command'):  # aioredis
                    deleted = await self.redis.delete(cache_key)
                else:  # redis-py
                    deleted = self.redis.delete(cache_key)
                return bool(deleted > 0)
            return False
        except RedisError as e:
            logger.error(f"Redis error deleting key {cache_key}: {e}")
            raise CacheOperationError(f"Failed to delete key from Redis: {cache_key}", original_exception=e) from e

    async def clear(self) -> None:
        """Clear all cached items from Redis asynchronously."""
        try:
            # Get all keys matching the prefix
            if hasattr(self.redis, 'scan_iter'):
                keys = []
                for key in self.redis.scan_iter(f"{self.key_prefix}*"):
                    if isinstance(key, bytes):
                        key = key.decode('utf-8')
                    keys.append(key)
                
                # Delete in batches
                if keys:
                    if hasattr(self.redis, 'delete'):
                        if hasattr(self.redis, 'execute_command'):  # aioredis
                            await self.redis.delete(*keys)
                        else:  # redis-py
                            self.redis.delete(*keys)
        except RedisError as e:
            logger.error(f"Redis error clearing cache: {e}")
            raise CacheOperationError("Failed to clear Redis cache", original_exception=e) from e

    async def size(self) -> int:
        """Get the number of cached items asynchronously."""
        try:
            if hasattr(self.redis, 'scan_iter'):
                return sum(1 for _ in self.redis.scan_iter(f"{self.key_prefix}*"))
            return 0
        except RedisError as e:
            logger.error(f"Redis error getting cache size: {e}")
            raise CacheOperationError("Failed to get cache size from Redis", original_exception=e) from e

    def get_eviction_policy(self) -> EvictionPolicy:
        return self.eviction_policy

    async def get_stats(self) -> dict:
        """Get cache statistics asynchronously."""
        stats = {
            'backend': 'redis',
            'key_prefix': self.key_prefix,
            'eviction_policy': type(self.eviction_policy).__name__,
            'connected': False
        }
        
        try:
            if hasattr(self.redis, 'ping'):
                if hasattr(self.redis, 'execute_command'):  # aioredis
                    stats['connected'] = await self.redis.ping()
                else:  # redis-py
                    stats['connected'] = self.redis.ping()
        except RedisError as e:
            logger.error(f"Error getting Redis stats: {e}")
            raise CacheOperationError("Failed to get Redis stats", original_exception=e) from e
            
        return stats