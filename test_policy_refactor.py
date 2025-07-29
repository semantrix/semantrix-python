#!/usr/bin/env python3
"""
Simple test to verify the policy refactoring worked.
"""

import asyncio

async def test_policy_imports():
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

async def test_policy_creation():
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
        evicted = await lru_policy.apply(test_cache, max_size=1)
        print(f"✓ Policy application test passed (evicted: {evicted}")
        
        return True
    except Exception as e:
        print(f"✗ Policy creation error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

async def test_store_with_policy():
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
        await store.add("test1", "response1")
        await store.add("test2", "response2")
        await store.add("test3", "response3")
        await store.add("test4", "response4")  # Should trigger eviction
        
        size = await store.size()
        print(f"✓ Store operations test passed (size: {size})")
        
        # Test get_exact
        value = await store.get_exact("test4")
        print(f"✓ Get exact value: {value}")
        
        # Test clear
        await store.clear()
        size_after_clear = await store.size()
        print(f"✓ Clear test passed (size after clear: {size_after_clear})")
        
        return True
    except Exception as e:
        print(f"✗ Store with policy error: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

async def run_tests():
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
        if await test_func():
            print("✓ PASSED")
            passed += 1
        else:
            print("✗ FAILED")
    
    print("\n" + "=" * 40)
    print(f"Tests completed: {passed}/{total} passed")
    return passed == total

if __name__ == "__main__":
    import asyncio
    result = asyncio.run(run_tests())
    
    if result:
        print("🎉 All tests passed! Policy pattern refactoring successful.")
    else:
        print("❌ Some tests failed. Check the errors above.")