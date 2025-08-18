#!/usr/bin/env python3
"""
Minimal Semantrix Demo
=====================

A very simple demo showing basic get/set operations.
This demo avoids complex async initialization issues.
"""

import asyncio
from semantrix.core.cache import Semantrix
from semantrix.cache_store.base import DeletionMode


async def minimal_demo():
    """Minimal demo showing basic operations."""
    print("🚀 Minimal Semantrix Demo")
    print("=" * 40)
    
    # Initialize Semantrix with minimal settings
    cache = Semantrix(
        similarity_threshold=0.8,
        enable_wal=False,  # Disable WAL to avoid complexity
        enable_2pc=False,  # Disable 2PC to avoid complexity
        enable_profiling=False  # Disable profiling
    )
    
    try:
        await cache.initialize()
        print("✅ Semantrix initialized!")
        
        # SET operation
        print("\n📝 Setting data...")
        await cache.set("What is Python?", "Python is a programming language.")
        await cache.set("How to install Python?", "Download from python.org")
        print("✅ Added 2 items to cache")
        
        # GET operation - exact match
        print("\n🔍 Getting data (exact match)...")
        result = await cache.get("What is Python?")
        print(f"Result: {result}")
        
        # GET operation - semantic match
        print("\n🔍 Getting data (semantic match)...")
        result = await cache.get("Tell me about Python")
        print(f"Result: {result}")
        
        # DELETE operation
        print("\n🗑️ Deleting data...")
        success = await cache.delete("What is Python?")
        print(f"Delete successful: {success}")
        
        # Verify deletion
        result = await cache.get("What is Python?")
        print(f"After delete: {result}")
        
        # TOMBSTONE operation
        print("\n🪦 Tombstoning data...")
        await cache.set("Test item", "This will be tombstoned")
        success = await cache.tombstone("Test item")
        print(f"Tombstone successful: {success}")
        
        # Verify tombstone
        result = await cache.get("Test item")
        print(f"After tombstone: {result}")
        
        print("\n✅ Minimal demo completed!")
        
    finally:
        await cache.shutdown()


def main():
    """Run the minimal demo."""
    print("🎯 MINIMAL SEMANTRIX DEMO")
    print("=" * 40)
    print("This demo shows the core operations:")
    print("- SET: Add data to cache")
    print("- GET: Retrieve data (exact and semantic)")
    print("- DELETE: Remove data")
    print("- TOMBSTONE: Mark data as deleted")
    print("=" * 40)
    
    try:
        # Run the async demo
        asyncio.run(minimal_demo())
        
        print("\n🎉 Demo completed successfully!")
        print("You now know how to use Semantrix's core operations!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Make sure you have Semantrix installed and configured properly.")


if __name__ == "__main__":
    main()
