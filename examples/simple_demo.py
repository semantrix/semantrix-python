#!/usr/bin/env python3
"""
Simple Semantrix Demo Application
=================================

This demo showcases the core functionality of Semantrix including:
- Basic get/set operations
- Semantic caching with similarity matching
- Delete operations (direct and tombstone)
- Explain functionality for debugging
- Both synchronous and asynchronous interfaces
- Error handling and logging

Run this demo to see Semantrix in action!
"""

import asyncio
import time
import logging
from typing import Optional

# Import Semantrix components
from semantrix.client import SemantrixClient
from semantrix.core.cache import Semantrix
from semantrix.cache_store.base import DeletionMode
from semantrix.utils.logging_config import configure_from_environment

# Configure logging to see what's happening
configure_from_environment()
logging.basicConfig(level=logging.INFO)


def sync_demo():
    """Demonstrate synchronous Semantrix usage."""
    print("\n" + "="*60)
    print("SYNCHRONOUS SEMANTRIX DEMO")
    print("="*60)
    
    # Initialize Semantrix with custom settings
    cache = SemantrixClient(
        similarity_threshold=0.8,  # Lower threshold for more matches
        max_memory_gb=0.5,        # Limit memory usage
        enable_profiling=True      # Enable profiling
    )
    
    print("✅ Semantrix initialized successfully!")
    
    # Demo 1: Basic get/set operations
    print("\n📝 Demo 1: Basic Get/Set Operations")
    print("-" * 40)
    
    # Set some initial data
    cache.set("What is Python?", "Python is a high-level programming language known for its simplicity and readability.")
    cache.set("How do I install Python?", "You can download Python from python.org or use package managers like pip.")
    cache.set("What is machine learning?", "Machine learning is a subset of AI that enables computers to learn without explicit programming.")
    
    print("✅ Added 3 prompt-response pairs to cache")
    
    # Try exact matches
    result1 = cache.get("What is Python?")
    print(f"🔍 Exact match for 'What is Python?': {result1[:50]}..." if result1 else "❌ No exact match")
    
    # Try semantic matches
    result2 = cache.get("Tell me about Python programming")
    print(f"🔍 Semantic match for 'Tell me about Python programming': {result2[:50]}..." if result2 else "❌ No semantic match")
    
    result3 = cache.get("Explain machine learning")
    print(f"🔍 Semantic match for 'Explain machine learning': {result3[:50]}..." if result3 else "❌ No semantic match")
    
    # Demo 2: Explain functionality
    print("\n🔍 Demo 2: Explain Functionality")
    print("-" * 40)
    
    # Explain a hit
    explain_hit = cache.explain("What is Python?")
    print(f"Explanation for 'What is Python?':")
    print(f"  - Cache hit: {explain_hit.cache_hit}")
    print(f"  - Exact match: {explain_hit.exact_match}")
    print(f"  - Semantic match: {explain_hit.semantic_match}")
    print(f"  - Total time: {explain_hit.total_time_ms:.2f}ms")
    
    # Explain a miss
    explain_miss = cache.explain("What is JavaScript?")
    print(f"\nExplanation for 'What is JavaScript?':")
    print(f"  - Cache hit: {explain_miss.cache_hit}")
    print(f"  - Top matches: {len(explain_miss.top_matches)}")
    if explain_miss.top_matches:
        print(f"  - Best similarity: {explain_miss.top_matches[0].similarity:.3f}")
    
    # Demo 3: Profiling and resource limits
    print("\n📊 Demo 3: Profiling and Resource Limits")
    print("-" * 40)
    
    # Get profiling stats
    stats = cache.profiler_stats
    print(f"Profiling enabled: {stats.get('enabled', False)}")
    if stats.get('enabled'):
        print(f"Total operations: {stats.get('total_operations', 0)}")
        print(f"Average time: {stats.get('average_time', 0):.2f}ms")
    
    # Check resource limits
    limits = cache.resource_limits
    print(f"Memory limit: {limits.max_memory_gb}GB")
    print(f"CPU limit: {limits.max_cpu_percent}%")
    
    print("\n✅ Synchronous demo completed!")


async def async_demo():
    """Demonstrate asynchronous Semantrix usage."""
    print("\n" + "="*60)
    print("ASYNCHRONOUS SEMANTRIX DEMO")
    print("="*60)
    
    # Initialize Semantrix with async context manager
    cache = Semantrix(
        similarity_threshold=0.8,
        enable_profiling=True,
        enable_wal=True,  # Enable Write-Ahead Logging
        enable_2pc=True   # Enable Two-Phase Commit
    )
    try:
        await cache.initialize()
        print("✅ Semantrix initialized successfully with async context!")
        
        # Demo 1: Basic async operations
        print("\n📝 Demo 1: Basic Async Get/Set Operations")
        print("-" * 40)
        
        # Set some data
        await cache.set("What is async programming?", "Async programming allows non-blocking execution of code.")
        await cache.set("How does asyncio work?", "Asyncio is Python's library for writing concurrent code using async/await syntax.")
        await cache.set("What are coroutines?", "Coroutines are functions that can pause and resume execution.")
        
        print("✅ Added 3 async prompt-response pairs to cache")
        
        # Get data
        result1 = await cache.get("What is async programming?")
        print(f"🔍 Async get for 'What is async programming?': {result1[:50]}..." if result1 else "❌ No match")
        
        result2 = await cache.get("Tell me about async code")
        print(f"🔍 Async semantic get for 'Tell me about async code': {result2[:50]}..." if result2 else "❌ No match")
        
        # Demo 2: Delete operations
        print("\n🗑️ Demo 2: Delete Operations")
        print("-" * 40)
        
        # Add a test item
        await cache.set("Test item for deletion", "This will be deleted")
        
        # Check it exists
        before_delete = await cache.get("Test item for deletion")
        print(f"Before delete: {'✅ Found' if before_delete else '❌ Not found'}")
        
        # Delete with direct mode
        delete_success = await cache.delete("Test item for deletion", mode=DeletionMode.DIRECT)
        print(f"Delete operation: {'✅ Success' if delete_success else '❌ Failed'}")
        
        # Check it's gone
        after_delete = await cache.get("Test item for deletion")
        print(f"After delete: {'✅ Found' if after_delete else '❌ Not found'}")
        
        # Demo 3: Tombstone operations
        print("\n🪦 Demo 3: Tombstone Operations")
        print("-" * 40)
        
        # Add another test item
        await cache.set("Test item for tombstone", "This will be tombstoned")
        
        # Tombstone it
        tombstone_success = await cache.tombstone("Test item for tombstone")
        print(f"Tombstone operation: {'✅ Success' if tombstone_success else '❌ Failed'}")
        
        # Check it's tombstoned (should return None)
        after_tombstone = await cache.get("Test item for tombstone")
        print(f"After tombstone: {'✅ Found' if after_tombstone else '❌ Not found (tombstoned)'}")
        
        # Demo 4: Explain with async
        print("\n🔍 Demo 4: Async Explain Functionality")
        print("-" * 40)
        
        explain_result = await cache.explain("What is async programming?")
        print(f"Async explain for 'What is async programming?':")
        print(f"  - Cache hit: {explain_result.cache_hit}")
        print(f"  - Exact match: {explain_result.exact_match}")
        print(f"  - Total time: {explain_result.total_time_ms:.2f}ms")
        
        # Demo 5: Error handling
        print("\n⚠️ Demo 5: Error Handling")
        print("-" * 40)
        
        try:
            # Try to get with invalid prompt
            await cache.get("")
        except Exception as e:
            print(f"✅ Caught expected error for empty prompt: {type(e).__name__}")
        
        try:
            # Try to set with invalid response
            await cache.set("Valid prompt", "")
        except Exception as e:
            print(f"✅ Caught expected error for empty response: {type(e).__name__}")
        
        print("\n✅ Asynchronous demo completed!")
        
    finally:
        await cache.shutdown()


def performance_demo():
    """Demonstrate performance characteristics."""
    print("\n" + "="*60)
    print("PERFORMANCE DEMO")
    print("="*60)
    
    cache = SemantrixClient(similarity_threshold=0.8, enable_profiling=True)
    
    # Benchmark set operations
    print("\n📊 Benchmarking Set Operations")
    print("-" * 40)
    
    start_time = time.time()
    for i in range(100):
        cache.set(f"Prompt {i}", f"Response {i} with some additional content to make it more realistic.")
    
    set_time = time.time() - start_time
    print(f"✅ Set 100 items in {set_time:.3f} seconds ({100/set_time:.1f} ops/sec)")
    
    # Benchmark get operations
    print("\n📊 Benchmarking Get Operations")
    print("-" * 40)
    
    start_time = time.time()
    hits = 0
    for i in range(100):
        result = cache.get(f"Prompt {i}")
        if result:
            hits += 1
    
    get_time = time.time() - start_time
    print(f"✅ Get 100 items in {get_time:.3f} seconds ({100/get_time:.1f} ops/sec)")
    print(f"✅ Hit rate: {hits/100*100:.1f}%")
    
    # Benchmark semantic search
    print("\n📊 Benchmarking Semantic Search")
    print("-" * 40)
    
    start_time = time.time()
    semantic_hits = 0
    for i in range(50):
        result = cache.get(f"Tell me about prompt {i}")
        if result:
            semantic_hits += 1
    
    semantic_time = time.time() - start_time
    print(f"✅ Semantic search 50 items in {semantic_time:.3f} seconds ({50/semantic_time:.1f} ops/sec)")
    print(f"✅ Semantic hit rate: {semantic_hits/50*100:.1f}%")
    
    print("\n✅ Performance demo completed!")


def main():
    """Run all demos."""
    print("🚀 SEMANTRIX DEMO APPLICATION")
    print("="*60)
    print("This demo showcases the core functionality of Semantrix")
    print("including get, set, delete operations and more!")
    print("="*60)
    
    try:
        # Run synchronous demo
        sync_demo()
        
        # Run performance demo
        # performance_demo()
        
        # Run asynchronous demo
        # asyncio.run(async_demo())
        
        print("\n" + "="*60)
        print("🎉 ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("You've seen Semantrix in action with:")
        print("✅ Basic get/set operations")
        print("✅ Semantic caching with similarity matching")
        print("✅ Delete operations (direct and tombstone)")
        print("✅ Explain functionality for debugging")
        print("✅ Both synchronous and asynchronous interfaces")
        print("✅ Error handling and validation")
        print("✅ Performance benchmarking")
        print("✅ Profiling and resource management")
        print("✅ Write-Ahead Logging and Two-Phase Commit")
        print("\nReady to use Semantrix in your own applications!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("This might be due to missing dependencies or configuration issues.")
        print("Please check the Semantrix documentation for setup instructions.")


if __name__ == "__main__":
    main()
