"""
Tests for delete operations in Semantrix.
"""

import pytest
import asyncio
from semantrix import Semantrix
from semantrix.cache_store.base import DeletionMode
from semantrix.exceptions import ValidationError


class TestDeleteOperations:
    """Test delete operations functionality."""
    
    @pytest.fixture
    async def cache(self):
        """Create a test cache instance."""
        cache = Semantrix(
            similarity_threshold=0.85,
            enable_wal=False,  # Disable WAL for simpler testing
            enable_2pc=False   # Disable 2PC for simpler testing
        )
        await cache.initialize()
        yield cache
        await cache.shutdown()
    
    @pytest.mark.asyncio
    async def test_direct_deletion(self, cache):
        """Test direct deletion of a prompt."""
        # Add a test item
        prompt = "What is the capital of France?"
        response = "The capital of France is Paris."
        await cache.set(prompt, response)
        
        # Verify it exists
        result = await cache.get(prompt)
        assert result == response
        
        # Delete it
        success = await cache.delete(prompt, mode=DeletionMode.DIRECT)
        assert success is True
        
        # Verify it's gone
        result = await cache.get(prompt)
        assert result is None
    
    @pytest.mark.asyncio
    async def test_tombstoning(self, cache):
        """Test tombstoning functionality."""
        # Add a test item
        prompt = "What is the capital of Germany?"
        response = "The capital of Germany is Berlin."
        await cache.set(prompt, response)
        
        # Verify it exists
        result = await cache.get(prompt)
        assert result == response
        
        # Tombstone it
        success = await cache.tombstone(prompt)
        assert success is True
        
        # Verify it appears deleted
        result = await cache.get(prompt)
        assert result is None
        
        # Verify it's marked as tombstoned
        is_tombstoned = await cache.cache_store.is_tombstoned(prompt)
        assert is_tombstoned is True
        
        # Verify cache size hasn't changed (item still exists in storage)
        original_size = await cache.cache_store.size()
        assert original_size > 0
    
    @pytest.mark.asyncio
    async def test_purge_tombstones(self, cache):
        """Test purging of tombstoned items."""
        # Add test items
        prompts = [
            ("Item 1", "Response 1"),
            ("Item 2", "Response 2"),
            ("Item 3", "Response 3"),
        ]
        
        for prompt, response in prompts:
            await cache.set(prompt, response)
        
        # Tombstone some items
        await cache.tombstone("Item 1")
        await cache.tombstone("Item 2")
        
        # Verify they appear deleted
        assert await cache.get("Item 1") is None
        assert await cache.get("Item 2") is None
        assert await cache.get("Item 3") == "Response 3"
        
        # Purge tombstones
        purged_count = await cache.purge_tombstones()
        assert purged_count == 2
        
        # Verify they're really gone
        assert await cache.get("Item 1") is None
        assert await cache.get("Item 2") is None
        assert await cache.get("Item 3") == "Response 3"
        
        # Verify they're not marked as tombstoned anymore
        assert not await cache.cache_store.is_tombstoned("Item 1")
        assert not await cache.cache_store.is_tombstoned("Item 2")
    
    @pytest.mark.asyncio
    async def test_multiple_direct_deletions(self, cache):
        """Test multiple direct deletions."""
        # Add test items
        test_data = [
            ("Item 1", "Response 1"),
            ("Item 2", "Response 2"),
            ("Item 3", "Response 3"),
        ]
        
        for prompt, response in test_data:
            await cache.set(prompt, response)
        
        # Verify all items exist
        for prompt, response in test_data:
            assert await cache.get(prompt) == response
        
        # Delete multiple items
        items_to_delete = ["Item 1", "Item 2"]
        for item in items_to_delete:
            success = await cache.delete(item, mode=DeletionMode.DIRECT)
            assert success is True
        
        # Verify deleted items are gone
        assert await cache.get("Item 1") is None
        assert await cache.get("Item 2") is None
        
        # Verify remaining item still exists
        assert await cache.get("Item 3") == "Response 3"
    
    @pytest.mark.asyncio
    async def test_deletion_modes(self, cache):
        """Test different deletion modes."""
        prompt = "Test prompt"
        response = "Test response"
        await cache.set(prompt, response)
        
        # Test direct deletion
        success = await cache.delete(prompt, mode=DeletionMode.DIRECT)
        assert success is True
        assert await cache.get(prompt) is None
        
        # Add it back
        await cache.set(prompt, response)
        
        # Test tombstone deletion
        success = await cache.delete(prompt, mode=DeletionMode.TOMBSTONE)
        assert success is True
        assert await cache.get(prompt) is None
        assert await cache.cache_store.is_tombstoned(prompt) is True
    
    @pytest.mark.asyncio
    async def test_deletion_with_nonexistent_key(self, cache):
        """Test deletion of non-existent keys."""
        # Direct deletion
        success = await cache.delete("nonexistent", mode=DeletionMode.DIRECT)
        assert success is False
        
        # Tombstone
        success = await cache.tombstone("nonexistent")
        assert success is False
        
        # Direct deletion of non-existent key should return False
        success = await cache.delete("nonexistent", mode=DeletionMode.DIRECT)
        assert success is False
    
    @pytest.mark.asyncio
    async def test_tombstone_cleanup_task(self, cache):
        """Test that tombstone cleanup task is started."""
        # The cleanup task should be started during initialization
        assert cache._tombstone_cleanup_task is not None
        assert not cache._tombstone_cleanup_task.done()
        
        # Cancel the task for cleanup
        cache._tombstone_cleanup_task.cancel()
        try:
            await cache._tombstone_cleanup_task
        except asyncio.CancelledError:
            pass
    
    @pytest.mark.asyncio
    async def test_deletion_with_validation(self, cache):
        """Test that deletion validates inputs properly."""
        # Test with empty prompt
        with pytest.raises(ValidationError):
            await cache.delete("")
        
        with pytest.raises(ValidationError):
            await cache.tombstone("")
        

        
        # Test with None prompt
        with pytest.raises(ValidationError):
            await cache.delete(None)
        
        with pytest.raises(ValidationError):
            await cache.tombstone(None)
        

    
    @pytest.mark.asyncio
    async def test_semantic_search_with_tombstones(self, cache):
        """Test that semantic search respects tombstones."""
        # Add similar items
        await cache.set("What is AI?", "Artificial Intelligence is the simulation of human intelligence.")
        await cache.set("What is ML?", "Machine Learning is a subset of AI.")
        
        # Tombstone one item
        await cache.tombstone("What is AI?")
        
        # Search for similar items
        result = await cache.get("What is artificial intelligence?")
        
        # Should not return the tombstoned item
        assert result != "Artificial Intelligence is the simulation of human intelligence."
        
        # Should potentially return the non-tombstoned item
        # (depending on similarity threshold)
        if result:
            assert result == "Machine Learning is a subset of AI."
    
    @pytest.mark.asyncio
    async def test_explain_with_tombstones(self, cache):
        """Test that explain functionality respects tombstones."""
        # Add an item
        await cache.set("What is AI?", "Artificial Intelligence is the simulation of human intelligence.")
        
        # Get explain result before tombstoning
        explain_before = await cache.explain("What is artificial intelligence?")
        
        # Tombstone the item
        await cache.tombstone("What is AI?")
        
        # Get explain result after tombstoning
        explain_after = await cache.explain("What is artificial intelligence?")
        
        # The results should be different (tombstoned item should not appear in matches)
        # This is a basic test - the exact behavior depends on the explain implementation
        assert explain_before != explain_after


if __name__ == "__main__":
    pytest.main([__file__])
