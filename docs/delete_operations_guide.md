# Delete Operations Guide

Semantrix supports comprehensive delete operations with multiple modes and tombstoning support for asynchronous cleanup.

## Overview

The delete operations in Semantrix provide two main capabilities:

1. **Direct Deletion**: Immediate deletion from both cache and vector stores
2. **Tombstoning**: Mark items as deleted in both cache and vector stores, then async background process handles cleanup for both

## Deletion Modes

### Direct Deletion

Direct deletion removes items by exact key match from both the cache store and vector store.

```python
import asyncio
from semantrix import Semantrix
from semantrix.cache_store.base import DeletionMode

async def direct_deletion_example():
    async with Semantrix() as cache:
        # Add an item
        await cache.set("What is AI?", "Artificial Intelligence is...")
        
        # Delete it directly
        success = await cache.delete("What is AI?", mode=DeletionMode.DIRECT)
        print(f"Deletion successful: {success}")
        
        # Verify it's gone
        result = await cache.get("What is AI?")
        print(f"After deletion: {result}")  # None
```



### Tombstoning

Tombstoning marks items as deleted in both cache and vector stores without immediately removing them. The async background process then handles the cleanup of tombstoned items from both stores. This provides better performance for bulk operations and ensures consistent tombstoning behavior across both stores.

```python
async def tombstoning_example():
    async with Semantrix() as cache:
        # Add an item
        await cache.set("What is AI?", "Artificial Intelligence is...")
        
        # Tombstone it
        success = await cache.tombstone("What is AI?")
        print(f"Tombstoning successful: {success}")
        
        # Verify it appears deleted
        result = await cache.get("What is AI?")
        print(f"After tombstoning: {result}")  # None
        
        # But it's still marked as tombstoned
        is_tombstoned = await cache.cache_store.is_tombstoned("What is AI?")
        print(f"Is tombstoned: {is_tombstoned}")  # True
```

## Tombstone Management

### Checking Tombstone Status

You can check if an item is tombstoned:

```python
async def check_tombstone_status():
    async with Semantrix() as cache:
        await cache.set("Test item", "Test response")
        await cache.tombstone("Test item")
        
        is_tombstoned = await cache.cache_store.is_tombstoned("Test item")
        print(f"Is tombstoned: {is_tombstoned}")  # True
```

### Purging Tombstones

Tombstoned items can be permanently removed from storage:

```python
async def purge_tombstones_example():
    async with Semantrix() as cache:
        # Add and tombstone some items
        await cache.set("Item 1", "Response 1")
        await cache.set("Item 2", "Response 2")
        await cache.tombstone("Item 1")
        await cache.tombstone("Item 2")
        
        # Purge all tombstones
        purged_count = await cache.purge_tombstones()
        print(f"Purged {purged_count} tombstoned items")
        
        # Verify they're really gone
        assert await cache.get("Item 1") is None
        assert await cache.get("Item 2") is None
        assert not await cache.cache_store.is_tombstoned("Item 1")
        assert not await cache.cache_store.is_tombstoned("Item 2")
```

## Automatic Tombstone Cleanup

Semantrix includes a background task that automatically purges tombstones at regular intervals (default: 1 hour). This background process handles the cleanup of tombstoned items from both cache and vector stores, ensuring that tombstoned items don't accumulate indefinitely.

The cleanup interval can be configured during initialization:

```python
async with Semantrix(
    similarity_threshold=0.85,
    enable_wal=True,
    enable_2pc=True
) as cache:
    # The tombstone cleanup task runs automatically in the background
    # You can also manually purge tombstones when needed
    await cache.purge_tombstones()
```

## Integration with WAL and 2PC

Delete operations are fully integrated with Semantrix's Write-Ahead Logging (WAL) and Two-Phase Commit (2PC) systems:

- **WAL**: All delete operations are logged for crash recovery
- **2PC**: Delete operations are atomic across cache store and vector store
- **Idempotency**: Operations can be safely retried using operation IDs

```python
async def atomic_deletion_example():
    async with Semantrix(enable_wal=True, enable_2pc=True) as cache:
        # Add an item
        await cache.set("Test item", "Test response")
        
        # Delete with operation ID for idempotency
        operation_id = "delete-op-123"
        success = await cache.delete("Test item", operation_id=operation_id)
        print(f"Atomic deletion successful: {success}")
```

## Best Practices

### When to Use Each Deletion Mode

1. **Direct Deletion**: Use for known keys that need immediate removal from both cache and vector stores
2. **Tombstoning**: Use for bulk operations or when you want to defer cleanup for both cache and vector stores

### Performance Considerations

- **Direct deletion** is the fastest and immediately removes items from both cache and vector stores
- **Tombstoning** is fast for marking items as deleted and defers cleanup for both stores to background process

### Memory Management

- Tombstoned items still consume memory until purged
- Use `purge_tombstones()` periodically to free up memory
- The automatic cleanup task helps manage memory usage

### Error Handling

```python
async def safe_deletion_example():
    async with Semantrix() as cache:
        try:
            # Try to delete an item
            success = await cache.delete("some-key")
            if not success:
                print("Item not found or already deleted")
        except Exception as e:
            print(f"Deletion failed: {e}")
```

## API Reference

### Main Delete Methods

- `cache.delete(prompt, mode=DeletionMode.DIRECT, operation_id=None) -> bool`
- `cache.tombstone(prompt, operation_id=None) -> bool`
- `cache.purge_tombstones(operation_id=None) -> int`

### Cache Store Methods

- `cache_store.delete(key, mode=DeletionMode.DIRECT) -> bool`
- `cache_store.tombstone(key) -> bool`
- `cache_store.is_tombstoned(key) -> bool`
- `cache_store.purge_tombstones() -> int`

### DeletionMode Enum

- `DeletionMode.DIRECT`: Immediate deletion from both cache and vector stores
- `DeletionMode.TOMBSTONE`: Mark as deleted in both cache and vector stores, async background process handles cleanup for both

## Migration from Old API

The deletion API has been simplified to two modes. If you were using semantic deletion, you'll need to implement your own semantic search logic:

```python
# Old way (deprecated)
deleted_prompts = await cache.delete_semantic("query", similarity_threshold=0.8)

# New way (implement your own semantic search)
# 1. Find similar prompts using vector search
# 2. Delete them individually using DIRECT mode
similar_prompts = await cache.vector_store.search(embedding, k=10)
for prompt in similar_prompts:
    if prompt.similarity >= 0.8:
        await cache.delete(prompt.document, mode=DeletionMode.DIRECT)
```

## Examples

See the complete examples in `examples/delete_operations_example.py` for more detailed usage patterns.
