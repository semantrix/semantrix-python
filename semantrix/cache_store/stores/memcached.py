"""
Memcached cache store implementation for Semantrix.

Provides a fast, distributed, ephemeral cache using Memcached with async support.
"""

import json
import time
import logging
from typing import Optional, Any, Dict, Union

from semantrix.cache_store.base import BaseCacheStore, EvictionPolicy, NoOpEvictionPolicy
from semantrix.exceptions import CacheOperationError

# Configure logging
logger = logging.getLogger(__name__)

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

    async def get_exact(self, prompt: str) -> Optional[str]:
        """
        Get a cached response from Memcached asynchronously.
        
        Args:
            prompt: The prompt to look up in the cache.
            
        Returns:
            The cached response if found, None otherwise.
        """
        key = self._get_key(prompt)
        try:
            if hasattr(self.client, 'get'):
                # Handle both sync and async Memcached clients
                if hasattr(self.client, 'get_multi'):  # aiomcache
                    value = await self.client.get(key.encode('utf-8'))
                else:  # pymemcache
                    value = self.client.get(key)
                
                if value is not None:
                    try:
                        if isinstance(value, bytes):
                            value = value.decode('utf-8')
                        if isinstance(value, str):
                            data = json.loads(value)
                            return data.get('response')
                        return str(value)
                    except (json.JSONDecodeError, KeyError, AttributeError) as e:
                        logger.warning(f"Error decoding cached value: {e}")
                        return str(value)
        except Exception as e:
            logger.error(f"Error getting value from Memcached: {e}")
            raise CacheOperationError("Failed to get item from Memcached", original_exception=e) from e

    async def add(self, prompt: str, response: str) -> None:
        """
        Add a response to the Memcached cache asynchronously.
        
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
            if hasattr(self.client, 'set'):
                value = json.dumps(data)
                if hasattr(self.client, 'get_multi'):  # aiomcache
                    await self.client.set(key.encode('utf-8'), value.encode('utf-8'))
                else:  # pymemcache
                    self.client.set(key, value)
        except Exception as e:
            logger.error(f"Error adding value to Memcached: {e}")
            raise CacheOperationError("Failed to add item to Memcached", original_exception=e) from e

    async def enforce_limits(self, resource_limits: Any) -> None:
        """
        Enforce cache size limits asynchronously.
        
        Note: Memcached handles its own eviction based on LRU.
        This is a no-op since Memcached manages its own eviction.
        """
        pass
        
    async def delete(self, key: str) -> bool:
        """
        Delete a key from the Memcached cache asynchronously.
        
        Args:
            key: The key to delete
            
        Returns:
            bool: True if the key was found and deleted, False otherwise
        """
        cache_key = self._get_key(key)
        try:
            if hasattr(self.client, 'delete'):
                if hasattr(self.client, 'get_multi'):  # aiomcache
                    # aiomcache delete returns True if key was deleted, False if not found
                    return await self.client.delete(cache_key.encode('utf-8'))
                else:  # pymemcache
                    # pymemcache delete returns True if key was deleted, False if not found
                    return self.client.delete(cache_key)
            return False
        except Exception as e:
            logger.error(f"Error deleting key from Memcached: {e}")
            raise CacheOperationError(f"Failed to delete key from Memcached", original_exception=e) from e

    async def clear(self) -> None:
        """Clear all cached items from Memcached asynchronously."""
        try:
            if hasattr(self.client, 'flush_all'):
                if hasattr(self.client, 'get_multi'):  # aiomcache
                    await self.client.flush_all()
                else:  # pymemcache
                    self.client.flush_all()
        except Exception as e:
            logger.error(f"Error clearing Memcached cache: {e}")
            raise CacheOperationError("Failed to clear Memcached cache", original_exception=e) from e

    async def size(self) -> int:
        """Get the number of cached items asynchronously."""
        try:
            if hasattr(self.client, 'stats'):
                if hasattr(self.client, 'get_multi'):  # aiomcache
                    stats = await self.client.stats()
                    return int(stats.get(b'curr_items', 0))
                else:  # pymemcache
                    stats = self.client.stats()
                    return int(stats.get('curr_items', 0))
            return 0
        except Exception as e:
            logger.error(f"Error getting cache size: {e}")
            raise CacheOperationError("Failed to get Memcached cache size", original_exception=e) from e

    def get_eviction_policy(self) -> EvictionPolicy:
        return self.eviction_policy

    async def get_stats(self) -> dict:
        """Get cache statistics asynchronously."""
        stats = {
            'backend': 'memcached',
            'key_prefix': self.key_prefix,
            'eviction_policy': type(self.eviction_policy).__name__,
            'connected': False,
            'size': 0,
            'memcached_stats': {}
        }
        
        try:
            if hasattr(self.client, 'stats'):
                if hasattr(self.client, 'get_multi'):  # aiomcache
                    raw_stats = await self.client.stats()
                    stats['connected'] = bool(raw_stats)
                    stats['size'] = int(raw_stats.get(b'curr_items', 0))
                    stats['memcached_stats'] = {k.decode('utf-8'): v.decode('utf-8') 
                                             for k, v in raw_stats.items()}
                else:  # pymemcache
                    raw_stats = self.client.stats()
                    stats['connected'] = bool(raw_stats)
                    stats['size'] = int(raw_stats.get('curr_items', 0))
                    stats['memcached_stats'] = dict(raw_stats)
        except Exception as e:
            logger.error(f"Error getting Memcached stats: {e}")
            raise CacheOperationError("Failed to get Memcached stats", original_exception=e) from e
            
        return stats