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
    def get_exact(self, prompt: str) -> Optional[str]:
        """
        Get an exact match for a prompt.
        
        Args:
            prompt: The prompt to search for
            
        Returns:
            The cached response if found, None otherwise
        """
        pass
    
    @abstractmethod
    def add(self, prompt: str, response: str) -> None:
        """
        Add a prompt-response pair to the cache.
        
        Args:
            prompt: The prompt
            response: The response to cache
        """
        pass
    
    @abstractmethod
    async def enforce_limits(self, resource_limits: Any) -> None:
        """
        Enforce resource limits by evicting items if necessary.
        
        Args:
            resource_limits: The resource limits object containing cache limits
            
        Note:
            This method should be called whenever the cache is modified to ensure
            that it doesn't exceed the specified resource limits.
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all cached items."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get the number of cached items."""
        pass
    
    @abstractmethod
    def get_eviction_policy(self) -> EvictionPolicy:
        """Get the eviction policy for this cache store."""
        pass 