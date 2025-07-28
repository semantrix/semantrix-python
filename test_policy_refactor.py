#!/usr/bin/env python3
"""
Simple test to verify the policy refactoring worked.
"""

def test_policy_imports():
    """Test that all policy components can be imported."""
    try:
        from semantrix.cache_store.base import EvictionPolicy, EvictionStrategy, BaseCacheStore
        print("✓ Base classes imported successfully")
        
        from semantrix.cache_store.eviction_policies import (
            StrategyBasedEvictionPolicy,
            NoOpEvictionPolicy,
            AdaptiveEvictionPolicy
        )
        print("✓ Policy classes imported successfully")
        
        from semantrix.cache_store.strategies import (
            LRUEvictionStrategy,
            FIFOEvictionStrategy,
            TTLStrategy,
            NoOpEvictionStrategy
        )
        print("✓ Strategy classes imported successfully")
        
        from semantrix.cache_store.stores import InMemoryStore
        print("✓ Store classes imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_policy_creation():
    """Test that policies can be created and used."""
    try:
        from semantrix.cache_store.eviction_policies import StrategyBasedEvictionPolicy, NoOpEvictionPolicy
        from semantrix.cache_store.strategies import LRUEvictionStrategy
        
        # Test strategy-based policy
        lru_strategy = LRUEvictionStrategy()
        lru_policy = StrategyBasedEvictionPolicy(lru_strategy)
        print("✓ StrategyBasedEvictionPolicy created successfully")
        
        # Test no-op policy
        noop_policy = NoOpEvictionPolicy()
        print("✓ NoOpEvictionPolicy created successfully")
        
        # Test policy application
        test_cache = {"key1": "value1", "key2": "value2"}
        evicted = lru_policy.apply(test_cache, max_size=1)
        print(f"✓ Policy application test passed (evicted: {evicted})")
        
        return True
    except Exception as e:
        print(f"✗ Policy creation error: {e}")
        return False

def test_store_with_policy():
    """Test that stores work with the new policy pattern."""
    try:
        from semantrix.cache_store.stores import InMemoryStore
        from semantrix.cache_store.eviction_policies import StrategyBasedEvictionPolicy
        from semantrix.cache_store.strategies import LRUEvictionStrategy
        
        # Create store with policy
        policy = StrategyBasedEvictionPolicy(LRUEvictionStrategy())
        store = InMemoryStore(max_size=3, eviction_policy=policy)
        print("✓ Store with policy created successfully")
        
        # Test store operations
        store.add("test1", "response1")
        store.add("test2", "response2")
        store.add("test3", "response3")
        store.add("test4", "response4")  # Should trigger eviction
        
        print(f"✓ Store operations test passed (size: {store.size()})")
        
        return True
    except Exception as e:
        print(f"✗ Store with policy error: {e}")
        return False

if __name__ == "__main__":
    print("Testing Policy Pattern Refactoring\n")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_policy_imports),
        ("Policy Creation Test", test_policy_creation),
        ("Store with Policy Test", test_store_with_policy)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Policy pattern refactoring successful.")
    else:
        print("❌ Some tests failed. Check the errors above.") 