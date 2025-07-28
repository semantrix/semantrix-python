#!/usr/bin/env python3
"""
Example: Distributed Cache Stores with NoOp Strategy
==================================================

This example demonstrates how to use the NoOpEvictionStrategy
with distributed cache stores that handle their own eviction.
"""

from semantrix import (
    SemantrixClient,
    BaseCacheStore,
    NoOpEvictionStrategy,
    InMemoryStore
)
from typing import Optional, Any
import time

class RedisCacheStore(BaseCacheStore):
    """Example Redis cache store implementation."""
    
    def __init__(self, redis_client=None):
        """
        Initialize Redis cache store.
        
        Args:
            redis_client: Redis client instance (for demo, we'll simulate)
        """
        self.redis_client = redis_client or self._create_mock_redis()
        self.eviction_strategy = NoOpEvictionStrategy()  # Redis handles eviction
    
    def _create_mock_redis(self):
        """Create a mock Redis client for demonstration."""
        class MockRedis:
            def __init__(self):
                self.data = {}
            
            def get(self, key):
                return self.data.get(key)
            
            def set(self, key, value, ex=None):
                self.data[key] = value
                return True
            
            def delete(self, key):
                if key in self.data:
                    del self.data[key]
                    return 1
                return 0
            
            def exists(self, key):
                return key in self.data
        
        return MockRedis()
    
    def get_exact(self, prompt: str) -> Optional[str]:
        """Get an exact match from Redis."""
        return self.redis_client.get(prompt)
    
    def add(self, prompt: str, response: str) -> None:
        """Add a prompt-response pair to Redis."""
        # Redis handles TTL and eviction automatically
        self.redis_client.set(prompt, response, ex=3600)  # 1 hour TTL
    
    def enforce_limits(self, resource_limits: Any) -> None:
        """No-op - Redis handles eviction automatically."""
        # Redis has its own eviction policies (LRU, LFU, etc.)
        # configured at the Redis server level
        pass
    
    def clear(self) -> None:
        """Clear all cached items (for demo purposes)."""
        # In real Redis, you'd use FLUSHDB or FLUSHALL
        self.redis_client.data.clear()
    
    def size(self) -> int:
        """Get the number of cached items."""
        return len(self.redis_client.data)
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            'type': 'Redis',
            'size': self.size(),
            'eviction_strategy': 'Redis-managed',
            'ttl_enabled': True
        }

class MemcachedCacheStore(BaseCacheStore):
    """Example Memcached cache store implementation."""
    
    def __init__(self, memcached_client=None):
        """
        Initialize Memcached cache store.
        
        Args:
            memcached_client: Memcached client instance (for demo, we'll simulate)
        """
        self.memcached_client = memcached_client or self._create_mock_memcached()
        self.eviction_strategy = NoOpEvictionStrategy()  # Memcached handles eviction
    
    def _create_mock_memcached(self):
        """Create a mock Memcached client for demonstration."""
        class MockMemcached:
            def __init__(self):
                self.data = {}
            
            def get(self, key):
                return self.data.get(key)
            
            def set(self, key, value, time=0):
                self.data[key] = value
                return True
            
            def delete(self, key):
                if key in self.data:
                    del self.data[key]
                    return True
                return False
        
        return MockMemcached()
    
    def get_exact(self, prompt: str) -> Optional[str]:
        """Get an exact match from Memcached."""
        return self.memcached_client.get(prompt)
    
    def add(self, prompt: str, response: str) -> None:
        """Add a prompt-response pair to Memcached."""
        # Memcached handles eviction automatically (LRU)
        self.memcached_client.set(prompt, response, time=3600)  # 1 hour TTL
    
    def enforce_limits(self, resource_limits: Any) -> None:
        """No-op - Memcached handles eviction automatically."""
        # Memcached uses LRU eviction by default
        pass
    
    def clear(self) -> None:
        """Clear all cached items (for demo purposes)."""
        self.memcached_client.data.clear()
    
    def size(self) -> int:
        """Get the number of cached items."""
        return len(self.memcached_client.data)
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            'type': 'Memcached',
            'size': self.size(),
            'eviction_strategy': 'LRU (Memcached-managed)',
            'ttl_enabled': True
        }

def demonstrate_distributed_caches():
    """Demonstrate distributed cache stores with NoOp strategy."""
    print("=== Distributed Cache Stores Example ===\n")
    
    # Example 1: Redis cache store
    print("1. Redis Cache Store (NoOp eviction strategy):")
    redis_store = RedisCacheStore()
    
    # Add some data
    redis_store.add("What is AI?", "Artificial Intelligence is a field of computer science.")
    redis_store.add("How does ML work?", "Machine learning uses algorithms to learn patterns.")
    
    print(f"   Cache size: {redis_store.size()}")
    print(f"   Stats: {redis_store.get_stats()}")
    
    # Test retrieval
    result = redis_store.get_exact("What is AI?")
    print(f"   Retrieved: {result}")
    
    # Example 2: Memcached cache store
    print("\n2. Memcached Cache Store (NoOp eviction strategy):")
    memcached_store = MemcachedCacheStore()
    
    # Add some data
    memcached_store.add("What is Python?", "Python is a programming language.")
    memcached_store.add("What is JavaScript?", "JavaScript is a programming language for web development.")
    
    print(f"   Cache size: {memcached_store.size()}")
    print(f"   Stats: {memcached_store.get_stats()}")
    
    # Test retrieval
    result = memcached_store.get_exact("What is Python?")
    print(f"   Retrieved: {result}")
    
    # Example 3: Using with SemantrixClient
    print("\n3. Using distributed cache with SemantrixClient:")
    
    # Create client with Redis cache store
    client = SemantrixClient(
        cache_store=RedisCacheStore()
    )
    
    # Add data
    client.set("What is data science?", "Data science combines statistics and programming.")
    client.set("What is deep learning?", "Deep learning uses neural networks with multiple layers.")
    
    print("   Added data to Redis cache")
    print(f"   Cache size: {client.cache.cache_store.size()}")
    
    # Test cache hit
    result = client.get("What is data science?")
    print(f"   Cache hit result: {result}")
    
    # Example 4: Compare with InMemoryStore
    print("\n4. Comparison with InMemoryStore:")
    
    # InMemoryStore with LRU strategy
    inmemory_client = SemantrixClient(
        cache_store=InMemoryStore(
            max_size=100,
            eviction_strategy=LRUEvictionStrategy()
        )
    )
    
    # Redis store with NoOp strategy
    redis_client = SemantrixClient(
        cache_store=RedisCacheStore()
    )
    
    print("   InMemoryStore (LRU eviction):")
    print(f"     Strategy: {inmemory_client.cache.cache_store.eviction_strategy}")
    print(f"     Stats: {inmemory_client.cache.cache_store.get_stats()}")
    
    print("   RedisStore (NoOp eviction):")
    print(f"     Strategy: {redis_client.cache.cache_store.eviction_strategy}")
    print(f"     Stats: {redis_client.cache.cache_store.get_stats()}")
    
    print("\n=== Distributed cache example completed! ===")


def demonstrate_noop_strategy():
    """Demonstrate the NoOp strategy specifically."""
    print("\n=== NoOp Strategy Demonstration ===\n")
    
    from semantrix import NoOpEvictionStrategy
    
    # Create NoOp strategy
    noop_strategy = NoOpEvictionStrategy()
    
    # Test behavior
    print("NoOp Strategy Behavior:")
    print(f"  Should evict (size=1000, max=100): {noop_strategy.should_evict(1000, 100)}")
    print(f"  Eviction candidates: {noop_strategy.get_eviction_candidates({'key1': 'value1'}, 1)}")
    
    # Compare with LRU strategy
    from semantrix import LRUEvictionStrategy
    lru_strategy = LRUEvictionStrategy()
    
    print("\nLRU Strategy Behavior:")
    print(f"  Should evict (size=1000, max=100): {lru_strategy.should_evict(1000, 100)}")
    print(f"  Eviction candidates: {lru_strategy.get_eviction_candidates({'key1': 'value1', 'key2': 'value2'}, 1)}")
    
    print("\n=== NoOp strategy demonstration completed! ===")


if __name__ == "__main__":
    demonstrate_distributed_caches()
    demonstrate_noop_strategy() 