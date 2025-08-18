#!/usr/bin/env python3
"""
Basic Semantrix Demo
===================

A simple demo showing the core get, set, and delete operations.
Perfect for getting started with Semantrix!
"""

import asyncio
from semantrix.client import SemantrixClient
from semantrix.core.cache import Semantrix
from semantrix.cache_store.base import DeletionMode


def basic_sync_demo():
    """Basic synchronous demo with get, set, delete operations."""
    print("🚀 Basic Semantrix Demo - Synchronous")
    print("=" * 50)
    
    try:
        # Initialize Semantrix with context manager
        print("Initializing Semantrix...")
        with SemantrixClient(similarity_threshold=0.8) as cache:
            print("✅ Semantrix initialized!")
            
            # SET operation
            print("\n📝 Setting data...")
            cache.set("What is Python?", "Python is a programming language.")
            cache.set("How to install Python?", "Download from python.org")
            print("✅ Added 2 items to cache")
            
            # GET operation - exact match
            print("\n🔍 Getting data (exact match)...")
            result = cache.get("What is Python?")
            print(f"Result: {result}")
            
            # GET operation - semantic match
            print("\n🔍 Getting data (semantic match)...")
            result = cache.get("Tell me about Python")
            print(f"Result: {result}")
            
            # DELETE operation
            print("\n🗑️ Deleting data...")
            success = cache.delete("What is Python?")
            print(f"Delete successful: {success}")
            
            # Verify deletion
            result = cache.get("What is Python?")
            print(f"After delete: {result}")
            
            print("\n✅ Basic sync demo completed!")
    except Exception as e:
        print(f"❌ Error in sync demo: {e}")
        import traceback
        traceback.print_exc()


async def basic_async_demo():
    """Basic asynchronous demo with get, set, delete operations."""
    print("\n🚀 Basic Semantrix Demo - Asynchronous")
    print("=" * 50)
    
    # Initialize Semantrix with async context manager
    cache = Semantrix(similarity_threshold=0.8)
    try:
        await cache.initialize()
        print("✅ Semantrix initialized!")
        
        # SET operation
        print("\n📝 Setting data...")
        await cache.set("What is async?", "Async allows non-blocking operations.")
        await cache.set("How does asyncio work?", "Asyncio provides async/await syntax.")
        print("✅ Added 2 items to cache")
        
        # GET operation
        print("\n🔍 Getting data...")
        result = await cache.get("What is async?")
        print(f"Result: {result}")
        
        # Semantic GET
        result = await cache.get("Tell me about async programming")
        print(f"Semantic result: {result}")
        
        # DELETE operation
        print("\n🗑️ Deleting data...")
        success = await cache.delete("What is async?")
        print(f"Delete successful: {success}")
        
        # Verify deletion
        result = await cache.get("What is async?")
        print(f"After delete: {result}")
        
        # TOMBSTONE operation
        print("\n🪦 Tombstoning data...")
        await cache.set("Test item", "This will be tombstoned")
        success = await cache.tombstone("Test item")
        print(f"Tombstone successful: {success}")
        
        # Verify tombstone
        result = await cache.get("Test item")
        print(f"After tombstone: {result}")
        
        print("\n✅ Basic async demo completed!")
        
    finally:
        await cache.shutdown()


def main():
    """Run the basic demo."""
    print("🎯 SEMANTRIX BASIC DEMO")
    print("=" * 50)
    print("This demo shows the core operations:")
    print("- SET: Add data to cache")
    print("- GET: Retrieve data (exact and semantic)")
    print("- DELETE: Remove data")
    print("- TOMBSTONE: Mark data as deleted")
    print("=" * 50)
    
    try:
        # Run synchronous demo
        basic_sync_demo()
        
        print("\n🎉 Demo completed successfully!")
        print("You now know how to use Semantrix's core operations!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Make sure you have Semantrix installed and configured properly.")


if __name__ == "__main__":
    main()
