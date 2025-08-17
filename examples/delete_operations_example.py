"""
Delete Operations Example
========================

This example demonstrates the various delete operations available in Semantrix:
- Direct deletion
- Semantic deletion
- Tombstoning
- Tombstone cleanup
"""

import asyncio
import logging
from semantrix import Semantrix
from semantrix.cache_store.base import DeletionMode

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Demonstrate delete operations."""
    
    # Initialize Semantrix
    async with Semantrix(
        similarity_threshold=0.85,
        enable_wal=False,  # Disable WAL for simpler testing
        enable_2pc=False   # Disable 2PC for simpler testing
    ) as cache:
        
        logger.info("=== Semantrix Delete Operations Demo ===\n")
        
        # Add some test data
        test_data = [
            ("What is the capital of France?", "The capital of France is Paris."),
            ("What is the capital of Germany?", "The capital of Germany is Berlin."),
            ("What is the capital of Italy?", "The capital of Italy is Rome."),
            ("How do I make a cake?", "To make a cake, you need flour, eggs, sugar, and butter."),
            ("How do I bake cookies?", "To bake cookies, mix flour, butter, sugar, and eggs, then bake at 350°F."),
            ("What is machine learning?", "Machine learning is a subset of artificial intelligence that enables computers to learn from data."),
            ("What is deep learning?", "Deep learning is a type of machine learning that uses neural networks with multiple layers."),
        ]
        
        logger.info("Adding test data to cache...")
        for prompt, response in test_data:
            await cache.set(prompt, response)
            logger.info(f"Added: {prompt[:50]}...")
        
        logger.info(f"Cache size: {await cache.cache_store.size()}\n")
        
        # Test 1: Direct deletion
        logger.info("=== Test 1: Direct Deletion ===")
        prompt_to_delete = "What is the capital of France?"
        
        # Verify the item exists
        result = await cache.get(prompt_to_delete)
        logger.info(f"Before deletion: {result}")
        
        # Delete the item
        success = await cache.delete(prompt_to_delete, mode=DeletionMode.DIRECT)
        logger.info(f"Direct deletion successful: {success}")
        
        # Verify it's gone
        result = await cache.get(prompt_to_delete)
        logger.info(f"After deletion: {result}\n")
        
        # Test 2: Direct deletion of multiple items
        logger.info("=== Test 2: Direct Deletion of Multiple Items ===")
        
        # Delete multiple items directly
        items_to_delete = ["What is the capital of Germany?", "What is the capital of Italy?"]
        for item in items_to_delete:
            success = await cache.delete(item, mode=DeletionMode.DIRECT)
            logger.info(f"Deleted '{item}': {success}")
        
        # Verify they're gone
        for item in items_to_delete:
            result = await cache.get(item)
            logger.info(f"After deletion - {item[:30]}...: {result}")
        
        logger.info(f"Cache size after multiple deletions: {await cache.cache_store.size()}\n")
        
        # Test 3: Tombstoning
        logger.info("=== Test 3: Tombstoning ===")
        prompt_to_tombstone = "What is the capital of Germany?"
        
        # Verify the item exists
        result = await cache.get(prompt_to_tombstone)
        logger.info(f"Before tombstoning: {result}")
        
        # Tombstone the item
        success = await cache.tombstone(prompt_to_tombstone)
        logger.info(f"Tombstoning successful: {success}")
        
        # Verify it appears deleted but still exists in storage
        result = await cache.get(prompt_to_tombstone)
        logger.info(f"After tombstoning (get): {result}")
        
        # Check if it's tombstoned
        is_tombstoned = await cache.cache_store.is_tombstoned(prompt_to_tombstone)
        logger.info(f"Is tombstoned: {is_tombstoned}")
        
        logger.info(f"Cache size after tombstoning: {await cache.cache_store.size()}\n")
        
        # Test 4: Tombstone cleanup
        logger.info("=== Test 4: Tombstone Cleanup ===")
        
        # Add another item and tombstone it
        await cache.set("Temporary item", "This will be tombstoned")
        await cache.tombstone("Temporary item")
        
        logger.info(f"Cache size before cleanup: {await cache.cache_store.size()}")
        
        # Purge tombstones
        purged_count = await cache.purge_tombstones()
        logger.info(f"Purged {purged_count} tombstoned items")
        
        logger.info(f"Cache size after cleanup: {await cache.cache_store.size()}")
        
        # Verify the tombstoned items are really gone
        result = await cache.get("Temporary item")
        logger.info(f"After cleanup - Temporary item: {result}")
        
        is_tombstoned = await cache.cache_store.is_tombstoned("Temporary item")
        logger.info(f"After cleanup - Is tombstoned: {is_tombstoned}\n")
        
        # Test 5: Explain with deletion context
        logger.info("=== Test 5: Explain with Deletion Context ===")
        
        # Add a new item
        await cache.set("What is AI?", "Artificial Intelligence is the simulation of human intelligence by machines.")
        
        # Explain a query that should match
        explain_result = await cache.explain("What is artificial intelligence?")
        logger.info(f"Explain result: {explain_result}")
        
        # Tombstone the item
        await cache.tombstone("What is AI?")
        
        # Explain the same query after tombstoning
        explain_result = await cache.explain("What is artificial intelligence?")
        logger.info(f"Explain result after tombstoning: {explain_result}")
        
        logger.info("\n=== Demo Complete ===")

if __name__ == "__main__":
    asyncio.run(main())
