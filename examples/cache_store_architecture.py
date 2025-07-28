#!/usr/bin/env python3
"""
Example: Cache Store Architecture
================================

This example demonstrates the new cache store architecture with:
- InMemoryStore as the main cache storage
- Different eviction strategies (LRU, FIFO, TTL)
- How to configure and use different strategies
"""

from semantrix import (
    SemantrixClient, 
    InMemoryStore, 
    EvictionStrategy,
    NoOpEvictionStrategy,
    LRUEvictionStrategy,
    FIFOEvictionStrategy,
    TTLStrategy
)

def demonstrate_cache_store_architecture():
    """Demonstrate the new cache store architecture."""
    print("=== Cache Store Architecture Example ===\n")
    
    # Example 1: Default InMemoryStore (uses LRU)
    print("1. Default InMemoryStore (LRU strategy):")
    default_store = InMemoryStore(max_size=5)
    default_store.add("key1", "value1")
    default_store.add("key2", "value2")
    default_store.add("key3", "value3")
    print(f"   Size: {default_store.size()}")
    print(f"   Stats: {default_store.get_stats()}")
    
    # Example 2: InMemoryStore with FIFO strategy
    print("\n2. InMemoryStore with FIFO strategy:")
    fifo_store = InMemoryStore(
        max_size=3,
        eviction_strategy=FIFOEvictionStrategy()
    )
    
    # Add items to trigger eviction
    for i in range(5):
        fifo_store.add(f"key{i}", f"value{i}")
        print(f"   Added key{i}, size: {fifo_store.size()}")
    
    print(f"   Final size: {fifo_store.size()}")
    print(f"   Stats: {fifo_store.get_stats()}")
    
    # Example 3: InMemoryStore with TTL strategy
    print("\n3. InMemoryStore with TTL strategy:")
    ttl_store = InMemoryStore(
        max_size=10,
        enable_ttl=True,
        ttl_seconds=2  # 2 seconds TTL
    )
    
    ttl_store.add("persistent_key", "persistent_value")
    ttl_store.add("expiring_key", "expiring_value")
    
    print(f"   Initial size: {ttl_store.size()}")
    print(f"   Getting persistent_key: {ttl_store.get_exact('persistent_key')}")
    print(f"   Getting expiring_key: {ttl_store.get_exact('expiring_key')}")
    
    # Wait for expiration
    import time
    print("   Waiting 3 seconds for expiration...")
    time.sleep(3)
    
    print(f"   After expiration - size: {ttl_store.size()}")
    print(f"   Getting persistent_key: {ttl_store.get_exact('persistent_key')}")
    print(f"   Getting expiring_key: {ttl_store.get_exact('expiring_key')}")
    
    # Example 4: Custom eviction strategy
    print("\n4. Custom eviction strategy:")
    class CustomEvictionStrategy(EvictionStrategy):
        """Custom strategy that evicts items with 'temp' in the key."""
        
        def should_evict(self, cache_size: int, max_size: int) -> bool:
            return cache_size > max_size
        
        def get_eviction_candidates(self, cache, max_size: int) -> list[str]:
            # First, evict temporary items
            temp_keys = [k for k in cache.keys() if 'temp' in k]
            if len(temp_keys) > 0:
                return temp_keys[:len(temp_keys)]
            
            # Then evict oldest items
            if len(cache) <= max_size:
                return []
            items_to_remove = len(cache) - max_size
            return list(cache.keys())[:items_to_remove]
    
    custom_store = InMemoryStore(
        max_size=3,
        eviction_strategy=CustomEvictionStrategy()
    )
    
    custom_store.add("persistent_key", "persistent_value")
    custom_store.add("temp_key1", "temp_value1")
    custom_store.add("temp_key2", "temp_value2")
    custom_store.add("another_key", "another_value")
    
    print(f"   After adding 4 items, size: {custom_store.size()}")
    print(f"   Stats: {custom_store.get_stats()}")
    
    # Example 5: Using with SemantrixClient
    print("\n5. Using InMemoryStore with SemantrixClient:")
    
    # Create client with TTL cache store
    client = SemantrixClient(
        cache_store=InMemoryStore(
            max_size=100,
            enable_ttl=True,
            ttl_seconds=5
        )
    )
    
    # Add some data
    client.set("What is AI?", "Artificial Intelligence is a field of computer science.")
    client.set("How does ML work?", "Machine learning uses algorithms to learn patterns.")
    
    print("   Added data to cache")
    print(f"   Cache size: {client.cache.cache_store.size()}")
    
    # Test cache hit
    result = client.get("What is AI?")
    print(f"   Cache hit result: {result}")
    
    # Wait and test expiration
    print("   Waiting 6 seconds for expiration...")
    time.sleep(6)
    
    result = client.get("What is AI?")
    print(f"   After expiration: {result}")
    print(f"   Cache size: {client.cache.cache_store.size()}")
    
    # Example 6: NoOp strategy for distributed caches
    print("\n6. NoOp strategy for distributed caches:")
    noop_store = InMemoryStore(
        max_size=1000,
        eviction_strategy=NoOpEvictionStrategy()
    )
    
    noop_store.add("distributed_key1", "distributed_value1")
    noop_store.add("distributed_key2", "distributed_value2")
    
    print(f"   NoOp store size: {noop_store.size()}")
    print(f"   NoOp strategy: {noop_store.eviction_strategy}")
    print("   Note: NoOp strategy never evicts - suitable for Redis/Memcached")
    
    print("\n=== Cache store architecture example completed! ===")


def demonstrate_eviction_strategies():
    """Demonstrate different eviction strategies."""
    print("\n=== Eviction Strategies Comparison ===\n")
    
    strategies = [
        ("NoOp", NoOpEvictionStrategy()),
        ("LRU", LRUEvictionStrategy()),
        ("FIFO", FIFOEvictionStrategy()),
        ("TTL", TTLStrategy(ttl_seconds=2))
    ]
    
    for name, strategy in strategies:
        print(f"{name} Strategy:")
        print(f"  Should evict (size=5, max=3): {strategy.should_evict(5, 3)}")
        print(f"  Should evict (size=2, max=3): {strategy.should_evict(2, 3)}")
        
        # Test eviction candidates
        test_cache = {
            'key1': 'value1',
            'key2': 'value2', 
            'key3': 'value3',
            'key4': 'value4',
            'key5': 'value5'
        }
        
        candidates = strategy.get_eviction_candidates(test_cache, 3)
        print(f"  Eviction candidates: {candidates}")
        print()


if __name__ == "__main__":
    demonstrate_cache_store_architecture()
    demonstrate_eviction_strategies() 