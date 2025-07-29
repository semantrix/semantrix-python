"""
Eviction strategies for cache stores.
"""

import time
from typing import Dict, Any
from .base import EvictionStrategy

class NoOpEvictionStrategy(EvictionStrategy):
    """
    No-operation eviction strategy for distributed cache stores.
    
    This strategy never performs any evictions, assuming that eviction is handled
    by an external system (like Redis with its own eviction policies).
    """
    
    async def should_evict(self, cache_size: int, max_size: int) -> bool:
        """
        Never evict - let the external system handle it.
        
        Args:
            cache_size: Current number of items in the cache (unused)
            max_size: Maximum allowed items in the cache (unused)
            
        Returns:
            bool: Always returns False (never evict)
        """
        return False
    
    async def get_eviction_candidates(self, cache: Dict[str, Any], max_size: int) -> list[str]:
        """
        Return empty list - no eviction candidates.
        
        Args:
            cache: The cache dictionary (unused)
            max_size: Maximum allowed items in the cache (unused)
            
        Returns:
            list[str]: Empty list (no candidates for eviction)
        """
        return []

class LRUEvictionStrategy(EvictionStrategy):
    """
    Least Recently Used eviction strategy.
    
    This strategy evicts the least recently accessed items first when the cache
    exceeds its maximum size. It assumes the cache is an OrderedDict where the
    order reflects access time (oldest items first).
    """
    
    async def should_evict(self, cache_size: int, max_size: int) -> bool:
        """
        Determine if eviction should occur based on cache size.
        
        Args:
            cache_size: Current number of items in the cache
            max_size: Maximum allowed items in the cache
            
        Returns:
            bool: True if cache size exceeds max_size, False otherwise
        """
        return cache_size > max_size
    
    async def get_eviction_candidates(self, cache: Dict[str, Any], max_size: int) -> list[str]:
        """
        Get the oldest items for eviction.
        
        Args:
            cache: The cache dictionary (should be an OrderedDict)
            max_size: Maximum allowed items in the cache
            
        Returns:
            list[str]: List of keys to evict, oldest first
        """
        if len(cache) <= max_size:
            return []
        
        # For OrderedDict, oldest items are at the beginning
        items_to_remove = len(cache) - max_size
        return list(cache.keys())[:items_to_remove]

class FIFOEvictionStrategy(EvictionStrategy):
    """
    First In First Out eviction strategy.
    
    This strategy evicts the oldest items first when the cache exceeds its maximum size.
    For OrderedDict, this behaves the same as LRU.
    """
    
    async def should_evict(self, cache_size: int, max_size: int) -> bool:
        """
        Determine if eviction should occur based on cache size.
        
        Args:
            cache_size: Current number of items in the cache
            max_size: Maximum allowed items in the cache
            
        Returns:
            bool: True if cache size exceeds max_size, False otherwise
        """
        return cache_size > max_size
    
    async def get_eviction_candidates(self, cache: Dict[str, Any], max_size: int) -> list[str]:
        """
        Get the oldest items for eviction.
        
        Args:
            cache: The cache dictionary (should be an OrderedDict)
            max_size: Maximum allowed items in the cache
            
        Returns:
            list[str]: List of keys to evict, oldest first
        """
        if len(cache) <= max_size:
            return []
        
        items_to_remove = len(cache) - max_size
        return list(cache.keys())[:items_to_remove]

class TTLStrategy(EvictionStrategy):
    """
    Time To Live eviction strategy.
    
    This strategy evicts items that have exceeded their TTL (Time To Live).
    If the cache is still over its size limit after removing expired items,
    it will remove the oldest items to reach the size limit.
    
    Args:
        ttl_seconds: Time in seconds after which an item is considered expired
    """
    
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds
    
    async def should_evict(self, cache_size: int, max_size: int) -> bool:
        """
        Always check for expired items or if cache is over limit.
        
        Args:
            cache_size: Current number of items in the cache
            max_size: Maximum allowed items in the cache
            
        Returns:
            bool: Always returns True to ensure we check for expired items
        """
        return True
    
    async def get_eviction_candidates(self, cache: Dict[str, Any], max_size: int) -> list[str]:
        """
        Get expired items and oldest items if still over limit.
        
        Args:
            cache: The cache dictionary containing items with 'timestamp' field
            max_size: Maximum allowed items in the cache
            
        Returns:
            list[str]: List of keys to evict (expired items first, then oldest)
        """
        current_time = time.time()
        expired_keys = []
        
        # Find expired items
        for key, value in cache.items():
            if isinstance(value, dict) and 'timestamp' in value:
                if current_time - value['timestamp'] > self.ttl_seconds:
                    expired_keys.append(key)
        
        # If still over limit after removing expired items
        remaining_size = len(cache) - len(expired_keys)
        if remaining_size > max_size:
            # Remove oldest non-expired items
            non_expired_keys = [k for k in cache.keys() if k not in expired_keys]
            items_to_remove = remaining_size - max_size
            expired_keys.extend(non_expired_keys[:items_to_remove])
        
        return expired_keys

class LFUEvictionStrategy(EvictionStrategy):
    """
    Least Frequently Used eviction strategy.
    
    This strategy evicts the least frequently accessed items first when the cache
    exceeds its maximum size. It assumes each cache value is a dictionary with an
    'access_count' field that tracks how many times the item has been accessed.
    """
    
    async def should_evict(self, cache_size: int, max_size: int) -> bool:
        """
        Determine if eviction should occur based on cache size.
        
        Args:
            cache_size: Current number of items in the cache
            max_size: Maximum allowed items in the cache
            
        Returns:
            bool: True if cache size exceeds max_size, False otherwise
        """
        return cache_size > max_size
    
    async def get_eviction_candidates(self, cache: Dict[str, Any], max_size: int) -> list[str]:
        """
        Get the least frequently used items for eviction.
        
        Args:
            cache: The cache dictionary where values should have 'access_count'
            max_size: Maximum allowed items in the cache
            
        Returns:
            list[str]: List of keys to evict, least frequently used first
            
        Note:
            This implementation assumes cache values are dictionaries with an optional
            'access_count' field. Items without this field are treated as having a count of 0.
        """
        if len(cache) <= max_size:
            return []
        
        # Calculate how many items we need to remove
        items_to_remove = len(cache) - max_size
        
        # Sort by access count (least frequent first)
        sorted_keys = sorted(
            cache.keys(),
            key=lambda k: cache[k].get('access_count', 0) if isinstance(cache[k], dict) else 0
        )
        
        return sorted_keys[:items_to_remove] 