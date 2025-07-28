"""
Eviction strategies for cache stores.
"""

import time
from typing import Dict, Any
from .base import EvictionStrategy

class NoOpEvictionStrategy(EvictionStrategy):
    """No-operation eviction strategy for distributed cache stores."""
    
    def should_evict(self, cache_size: int, max_size: int) -> bool:
        """Never evict - let the external system handle it."""
        return False
    
    def get_eviction_candidates(self, cache: Dict[str, Any], max_size: int) -> list[str]:
        """Return empty list - no eviction candidates."""
        return []

class LRUEvictionStrategy(EvictionStrategy):
    """Least Recently Used eviction strategy."""
    
    def should_evict(self, cache_size: int, max_size: int) -> bool:
        """Evict when cache size exceeds max size."""
        return cache_size > max_size
    
    def get_eviction_candidates(self, cache: Dict[str, Any], max_size: int) -> list[str]:
        """Get oldest items for eviction."""
        if len(cache) <= max_size:
            return []
        
        # For OrderedDict, oldest items are at the beginning
        items_to_remove = len(cache) - max_size
        return list(cache.keys())[:items_to_remove]

class FIFOEvictionStrategy(EvictionStrategy):
    """First In First Out eviction strategy."""
    
    def should_evict(self, cache_size: int, max_size: int) -> bool:
        """Evict when cache size exceeds max size."""
        return cache_size > max_size
    
    def get_eviction_candidates(self, cache: Dict[str, Any], max_size: int) -> list[str]:
        """Get oldest items for eviction (same as LRU for OrderedDict)."""
        if len(cache) <= max_size:
            return []
        
        items_to_remove = len(cache) - max_size
        return list(cache.keys())[:items_to_remove]

class TTLStrategy(EvictionStrategy):
    """Time To Live eviction strategy."""
    
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl_seconds = ttl_seconds
    
    def should_evict(self, cache_size: int, max_size: int) -> bool:
        """Always check for expired items."""
        return True
    
    def get_eviction_candidates(self, cache: Dict[str, Any], max_size: int) -> list[str]:
        """Get expired items and oldest items if over limit."""
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
    """Least Frequently Used eviction strategy."""
    
    def should_evict(self, cache_size: int, max_size: int) -> bool:
        """Evict when cache size exceeds max size."""
        return cache_size > max_size
    
    def get_eviction_candidates(self, cache: Dict[str, Any], max_size: int) -> list[str]:
        """Get least frequently used items for eviction."""
        if len(cache) <= max_size:
            return []
        
        # For simplicity, assume cache values have 'access_count' field
        # In a real implementation, this would track access frequency
        items_to_remove = len(cache) - max_size
        
        # Sort by access count (least frequent first)
        sorted_keys = sorted(
            cache.keys(),
            key=lambda k: cache[k].get('access_count', 0) if isinstance(cache[k], dict) else 0
        )
        
        return sorted_keys[:items_to_remove] 