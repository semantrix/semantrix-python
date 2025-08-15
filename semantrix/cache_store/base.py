"""
Base classes for cache stores and eviction strategies.
"""

from abc import ABC, abstractmethod
from typing import Optional, Any, Dict

class EvictionStrategy(ABC):
    """Abstract base class for cache eviction strategies."""
    
    @abstractmethod
    async def should_evict(self, cache_size: int, max_size: int) -> bool:
        """
        Determine if eviction should occur.
        
        Args:
            cache_size: Current size of the cache
            max_size: Maximum allowed size of the cache
            
        Returns:
            bool: True if eviction should occur, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_eviction_candidates(self, cache: Dict[str, Any], max_size: int) -> list[str]:
        """
        Get list of keys to evict.
        
        Args:
            cache: The cache dictionary
            max_size: Maximum allowed size of the cache
            
        Returns:
            list[str]: List of keys to evict
        """
        pass

class EvictionPolicy(ABC):
    """Abstract base class for eviction policies."""
    
    @abstractmethod
    async def apply(self, cache: Dict[str, Any], max_size: int) -> int:
        """
        Apply the eviction policy and return number of items evicted.
        
        Args:
            cache: The cache dictionary to evict from
            max_size: Maximum allowed size
            
        Returns:
            int: Number of items evicted
        """
        pass

class BaseCacheStore(ABC):
    """Abstract base class for cache storage with eviction policies."""
    
    @abstractmethod
    async def get_exact(self, prompt: str) -> Optional[str]:
        """
        Get a cached response if it exists and is not expired.

        Args:
            prompt: The prompt to look up in the cache.

        Returns:
            The cached response if found and not expired, None otherwise.
        """
        pass
    
    @abstractmethod
    async def add(self, prompt: str, response: str) -> None:
        """
        Add a response to the cache.

        Args:
            prompt: The prompt to cache.
            response: The response to cache.
        """
        pass
    
    @abstractmethod
    async def enforce_limits(self, resource_limits: Any) -> None:
        """
        Enforce cache size limits asynchronously.

        Args:
            resource_limits: The resource limits to enforce.

        Note:
            This method should be called whenever the cache is modified to ensure
            that it doesn't exceed the specified resource limits.
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        Delete a key from the cache.

        Args:
            key: The key to delete
            
        Returns:
            bool: True if the key was found and deleted, False otherwise
        """
        pass
        
    @abstractmethod
    async def clear(self) -> None:
        """Clear all cached items asynchronously."""
        pass


class NoOpEvictionPolicy(EvictionPolicy):
    """
    A no-operation eviction policy that never evicts any items.
    
    This is useful when you want to disable eviction entirely.
    """
    
    async def apply(self, cache: Dict[str, Any], max_size: int) -> int:
        """
        Apply the eviction policy (no-op implementation).
        
        Args:
            cache: The cache dictionary (unused)
            max_size: Maximum allowed size (unused)
            
        Returns:
            int: Always returns 0 (no items evicted)
        """
        return 0
    
    @abstractmethod
    async def size(self) -> int:
        """Get the number of cached items asynchronously."""
        pass
    
    @abstractmethod
    def get_eviction_policy(self) -> EvictionPolicy:
        """Get the eviction policy for this cache store."""
        pass 