"""
Eviction Policies for cache stores.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from .base import EvictionStrategy, EvictionPolicy
from .strategies import NoOpEvictionStrategy

class StrategyBasedEvictionPolicy(EvictionPolicy):
    """Eviction policy that uses a strategy pattern."""
    
    def __init__(self, strategy: EvictionStrategy):
        """
        Initialize with an eviction strategy.
        
        Args:
            strategy: The eviction strategy to use
        """
        self.strategy = strategy
        self.evicted_count = 0
    
    @property
    def strategy(self) -> EvictionStrategy:
        """Get the current strategy."""
        return self._strategy
    
    @strategy.setter
    def strategy(self, value: EvictionStrategy):
        """Set the strategy."""
        self._strategy = value
    
    def apply(self, cache: Dict[str, Any], max_size: int) -> int:
        """Apply the eviction policy using the strategy."""
        if not self.strategy.should_evict(len(cache), max_size):
            return 0
        
        candidates = self.strategy.get_eviction_candidates(cache, max_size)
        evicted_count = 0
        
        for key in candidates:
            if key in cache:
                del cache[key]
                evicted_count += 1
        
        self.evicted_count += evicted_count
        return evicted_count

class NoOpEvictionPolicy(EvictionPolicy):
    """No-operation eviction policy for distributed caches."""
    
    def apply(self, cache: Dict[str, Any], max_size: int) -> int:
        """No-op - return 0 evicted items."""
        return 0

class AdaptiveEvictionPolicy(EvictionPolicy):
    """Adaptive eviction policy that can switch strategies."""
    
    def __init__(self, default_strategy: EvictionStrategy):
        """
        Initialize with a default strategy.
        
        Args:
            default_strategy: The default eviction strategy
        """
        self.current_strategy = default_strategy
        self.strategy_history = []
        self.evicted_count = 0
    
    def switch_strategy(self, new_strategy: EvictionStrategy) -> None:
        """Switch to a new eviction strategy."""
        self.strategy_history.append(self.current_strategy)
        self.current_strategy = new_strategy
    
    def apply(self, cache: Dict[str, Any], max_size: int) -> int:
        """Apply the eviction policy using current strategy."""
        if not self.current_strategy.should_evict(len(cache), max_size):
            return 0
        
        candidates = self.current_strategy.get_eviction_candidates(cache, max_size)
        evicted_count = 0
        
        for key in candidates:
            if key in cache:
                del cache[key]
                evicted_count += 1
        
        self.evicted_count += evicted_count
        return evicted_count
    
    def get_stats(self) -> dict:
        """Get eviction policy statistics."""
        return {
            'current_strategy': type(self.current_strategy).__name__,
            'strategy_history': [type(s).__name__ for s in self.strategy_history],
            'evicted_count': self.evicted_count
        } 