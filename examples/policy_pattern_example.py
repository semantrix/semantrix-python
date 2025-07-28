"""
Policy Pattern Example
=====================

Demonstrates the new EvictionPolicy pattern for cache stores.
"""

from semantrix import (
    SemantrixClient,
    LRUEvictionStrategy,
    FIFOEvictionStrategy,
    TTLStrategy,
    NoOpEvictionStrategy,
    StrategyBasedEvictionPolicy,
    NoOpEvictionPolicy,
    AdaptiveEvictionPolicy
)

def demonstrate_policy_pattern():
    """Demonstrate different eviction policies."""
    
    print("=== Eviction Policy Pattern Demo ===\n")
    
    # 1. LRU Policy (default)
    print("1. LRU Eviction Policy:")
    lru_policy = StrategyBasedEvictionPolicy(LRUEvictionStrategy())
    client_lru = SemantrixClient(
        cache_store=InMemoryStore(
            max_size=5,
            eviction_policy=lru_policy
        )
    )
    
    # Add items to trigger eviction
    for i in range(7):
        client_lru.cache.add(f"prompt_{i}", f"response_{i}")
    
    print(f"   Cache size: {client_lru.cache.cache_store.size()}")
    print(f"   Evicted count: {lru_policy.evicted_count}")
    print()
    
    # 2. FIFO Policy
    print("2. FIFO Eviction Policy:")
    fifo_policy = StrategyBasedEvictionPolicy(FIFOEvictionStrategy())
    client_fifo = SemantrixClient(
        cache_store=InMemoryStore(
            max_size=3,
            eviction_policy=fifo_policy
        )
    )
    
    for i in range(5):
        client_fifo.cache.add(f"prompt_{i}", f"response_{i}")
    
    print(f"   Cache size: {client_fifo.cache.cache_store.size()}")
    print(f"   Evicted count: {fifo_policy.evicted_count}")
    print()
    
    # 3. TTL Policy
    print("3. TTL Eviction Policy:")
    ttl_policy = StrategyBasedEvictionPolicy(TTLStrategy(ttl_seconds=1))  # 1 second TTL
    client_ttl = SemantrixClient(
        cache_store=InMemoryStore(
            max_size=10,
            eviction_policy=ttl_policy,
            enable_ttl=True
        )
    )
    
    client_ttl.cache.add("test_prompt", "test_response")
    print(f"   Cache size before TTL: {client_ttl.cache.cache_store.size()}")
    
    import time
    time.sleep(1.1)  # Wait for TTL to expire
    
    # Try to get the expired item
    result = client_ttl.cache.cache_store.get_exact("test_prompt")
    print(f"   Cache size after TTL: {client_ttl.cache.cache_store.size()}")
    print(f"   Expired item result: {result}")
    print()
    
    # 4. No-Op Policy (for distributed caches)
    print("4. No-Op Eviction Policy (for distributed caches):")
    noop_policy = NoOpEvictionPolicy()
    client_noop = SemantrixClient(
        cache_store=InMemoryStore(
            max_size=2,
            eviction_policy=noop_policy
        )
    )
    
    for i in range(5):
        client_noop.cache.add(f"prompt_{i}", f"response_{i}")
    
    print(f"   Cache size: {client_noop.cache.cache_store.size()}")
    print(f"   Evicted count: {noop_policy.evicted_count}")
    print()
    
    # 5. Adaptive Policy
    print("5. Adaptive Eviction Policy:")
    adaptive_policy = AdaptiveEvictionPolicy(LRUEvictionStrategy())
    client_adaptive = SemantrixClient(
        cache_store=InMemoryStore(
            max_size=3,
            eviction_policy=adaptive_policy
        )
    )
    
    # Start with LRU
    for i in range(4):
        client_adaptive.cache.add(f"prompt_{i}", f"response_{i}")
    
    print(f"   After LRU - Cache size: {client_adaptive.cache.cache_store.size()}")
    print(f"   After LRU - Evicted count: {adaptive_policy.evicted_count}")
    
    # Switch to FIFO
    adaptive_policy.switch_strategy(FIFOEvictionStrategy())
    for i in range(4, 7):
        client_adaptive.cache.add(f"prompt_{i}", f"response_{i}")
    
    print(f"   After FIFO switch - Cache size: {client_adaptive.cache.cache_store.size()}")
    print(f"   After FIFO switch - Evicted count: {adaptive_policy.evicted_count}")
    
    # Show stats
    stats = adaptive_policy.get_stats()
    print(f"   Strategy history: {stats['strategy_history']}")
    print()

def demonstrate_policy_comparison():
    """Compare different policies side by side."""
    
    print("=== Policy Comparison ===\n")
    
    policies = {
        "LRU": StrategyBasedEvictionPolicy(LRUEvictionStrategy()),
        "FIFO": StrategyBasedEvictionPolicy(FIFOEvictionStrategy()),
        "TTL": StrategyBasedEvictionPolicy(TTLStrategy(ttl_seconds=1)),
        "No-Op": NoOpEvictionPolicy()
    }
    
    for name, policy in policies.items():
        print(f"{name} Policy:")
        print(f"  Type: {type(policy).__name__}")
        print(f"  Strategy: {type(policy.strategy).__name__ if hasattr(policy, 'strategy') else 'N/A'}")
        print(f"  Apply method: {policy.apply.__name__}")
        print()

if __name__ == "__main__":
    demonstrate_policy_pattern()
    demonstrate_policy_comparison() 