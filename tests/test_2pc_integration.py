"""
Tests for Two-Phase Commit (2PC) integration in Semantrix cache.
"""
import asyncio
import os
import shutil
import tempfile
import unittest
from typing import Any, Dict, List, Optional, Union
from unittest import IsolatedAsyncioTestCase

from semantrix.core.cache import Semantrix
from semantrix.embedding import BaseEmbedder
from semantrix.vector_store import BaseVectorStore, FAISSVectorStore
from semantrix.cache_store import BaseCacheStore, InMemoryStore

class MockFailingVectorStore(FAISSVectorStore):
    """Mock vector store that fails on specific operations for testing."""
    
    def __init__(self, dimension: int, fail_on: str = None):
        super().__init__(dimension=dimension)
        self.fail_on = fail_on
        self.calls = []
        self._collections = ["default"]
    
    async def add(
        self,
        vectors: Any,
        documents: Optional[Union[str, List[str]]] = None,
        metadatas: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        ids: Optional[Union[str, List[str]]] = None,
        **kwargs: Any
    ) -> List[str]:
        """Mock add method that can fail for testing purposes."""
        self.calls.append(('add', vectors, documents, metadatas, ids, kwargs))
        if self.fail_on == 'add':
            raise RuntimeError("Simulated failure in vector store add")
        return await super().add(vectors=vectors, documents=documents, metadatas=metadatas, ids=ids, **kwargs)
    
    async def delete(self, text_id: str):
        self.calls.append(('delete', text_id))
        if self.fail_on == 'delete':
            raise RuntimeError("Simulated failure in vector store delete")
        return await super().delete(text_id)
        
    async def create_index(self, index_type: str = "flat", **kwargs) -> bool:
        """Create or update the vector index."""
        self.calls.append(('create_index', index_type, kwargs))
        if self.fail_on == 'create_index':
            raise RuntimeError("Simulated failure in create_index")
        return True
    
    async def delete_collection(self, name: str, **kwargs) -> bool:
        """Delete a collection/namespace."""
        self.calls.append(('delete_collection', name, kwargs))
        if self.fail_on == 'delete_collection':
            raise RuntimeError("Simulated failure in delete_collection")
        if name in self._collections:
            self._collections.remove(name)
        return True
    
    async def list_collections(self) -> list[str]:
        """List all collections/namespaces in the store."""
        self.calls.append(('list_collections',))
        if self.fail_on == 'list_collections':
            raise RuntimeError("Simulated failure in list_collections")
        return self._collections.copy()

class MockFailingCacheStore(InMemoryStore):
    """Mock cache store that fails on specific operations for testing."""
    
    def __init__(self, fail_on: str = None):
        super().__init__()
        self.fail_on = fail_on
        self.calls = []
    
    async def get_exact(self, prompt: str) -> Optional[str]:
        self.calls.append(('get_exact', prompt))
        if self.fail_on == 'get_exact':
            raise RuntimeError("Simulated failure in cache store get_exact")
        return await super().get_exact(prompt)
    
    async def add(self, prompt: str, response: str) -> None:
        self.calls.append(('add', prompt))
        if self.fail_on == 'add':
            raise RuntimeError("Simulated failure in cache store add")
        return await super().add(prompt, response)
    
    async def set(self, key: str, value: str):
        # First add the call to track it
        self.calls.append(('set', key))
        
        # Check if we should fail after tracking the call
        if self.fail_on == 'set':
            # Don't call the parent's set method to simulate a failure
            raise RuntimeError("Simulated failure in cache store set")
            
        # If we get here, call the parent's set method
        return await super().set(key, value)
    
    async def delete(self, key: str):
        self.calls.append(('delete', key))
        if self.fail_on == 'delete':
            raise RuntimeError("Simulated failure in cache store delete")
        return await super().delete(key)
    
    async def enforce_limits(self, resource_limits: Any) -> None:
        self.calls.append(('enforce_limits', resource_limits))
        if self.fail_on == 'enforce_limits':
            raise RuntimeError("Simulated failure in cache store enforce_limits")
        return await super().enforce_limits(resource_limits)
    
    async def clear(self) -> None:
        self.calls.append(('clear',))
        if self.fail_on == 'clear':
            raise RuntimeError("Simulated failure in cache store clear")
        return await super().clear()

class Test2PCIntegration(IsolatedAsyncioTestCase):
    """Integration tests for Two-Phase Commit in Semantrix cache."""
    
    def setUp(self):
        # Create a temporary directory for WAL
        self.temp_dir = tempfile.mkdtemp(prefix="semantrix_test_")
        self.wal_config = {
            'log_dir': os.path.join(self.temp_dir, 'wal'),
            'max_log_size_mb': 1,
            'batch_size': 10,
            'batch_timeout_seconds': 0.1
        }
        
        # Create a simple embedder for testing
        class TestEmbedder(BaseEmbedder):
            def get_dimension(self) -> int:
                return 10
                
            async def embed(self, text: str) -> list[float]:
                # Simple deterministic embedding for testing
                return [hash(text) % 100 / 100.0 for _ in range(10)]
                
            async def encode(self, texts: list[str]) -> list[list[float]]:
                # Simple batch encoding for testing
                return [await self.embed(text) for text in texts]
        
        self.embedder = TestEmbedder()
    
    def tearDown(self):
        # Clean up temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    async def test_successful_2pc_operation(self):
        """Test a successful 2PC operation."""
        # Create Semantrix with WAL and 2PC enabled
        semantrix = Semantrix(
            embedder=self.embedder,
            enable_wal=True,
            enable_2pc=True,
            wal_config=self.wal_config
        )
        
        await semantrix.initialize()
        
        try:
            # Add a prompt-response pair
            await semantrix.set("test prompt", "test response")
            
            # Verify it was added to both stores
            self.assertEqual(await semantrix.get("test prompt"), "test response")
            
        finally:
            await semantrix.shutdown()
    
    async def test_vector_store_failure_during_2pc(self):
        """Test 2PC rollback when vector store fails during commit."""
        # Create a failing vector store
        vector_store = MockFailingVectorStore(
            dimension=self.embedder.get_dimension(),
            fail_on='add'  # Fail when adding vectors
        )
        
        # Create Semantrix with the failing vector store
        semantrix = Semantrix(
            embedder=self.embedder,
            vector_store=vector_store,
            enable_wal=True,
            enable_2pc=True,
            wal_config=self.wal_config
        )
        
        await semantrix.initialize()
        
        try:
            # This should fail during the commit phase
            with self.assertRaises(RuntimeError):
                await semantrix.set("test prompt", "test response")
            
            # Verify nothing was added to the vector store
            results = await vector_store.search(await self.embedder.embed("test prompt"))
            self.assertFalse(results)  # Should be empty list or None
            
        finally:
            await semantrix.shutdown()
    
    async def test_cache_store_failure_during_2pc(self):
        """Test 2PC rollback when cache store fails during commit."""
        # Create a failing cache store
        cache_store = MockFailingCacheStore(fail_on='add')
        
        # Create Semantrix with the failing cache store
        semantrix = Semantrix(
            embedder=self.embedder,
            cache_store=cache_store,
            enable_wal=True,
            enable_2pc=True,
            wal_config=self.wal_config
        )
        
        await semantrix.initialize()
        
        try:
            # This should fail during the commit phase
            with self.assertRaises(RuntimeError):
                await semantrix.set("test prompt", "test response")
            
            # Verify the operation was rolled back
            self.assertIsNone(await cache_store.get("test prompt"))
            
        finally:
            await semantrix.shutdown()
    
    async def test_recovery_after_failure(self):
        """Test recovery of pending operations after a failure."""
        # First, create and fail an operation
        vector_store = MockFailingVectorStore(
            dimension=self.embedder.get_dimension(),
            fail_on='add'  # Will fail on first attempt
        )
        
        semantrix = Semantrix(
            embedder=self.embedder,
            vector_store=vector_store,
            enable_wal=True,
            enable_2pc=True,
            wal_config=self.wal_config
        )
        
        await semantrix.initialize()
        
        try:
            # This will fail
            with self.assertRaises(RuntimeError):
                await semantrix.set("test prompt", "test response")
        finally:
            await semantrix.shutdown()
        
        # Now create a new Semantrix instance that will recover from WAL
        # But this time the vector store won't fail
        vector_store.fail_on = None
        
        semantrix = Semantrix(
            embedder=self.embedder,
            vector_store=vector_store,
            enable_wal=True,
            enable_2pc=True,
            wal_config=self.wal_config
        )
        
        await semantrix.initialize()
        
        try:
            # The operation should be recovered and retried
            # Since vector_store no longer fails, it should succeed
            self.assertEqual(await semantrix.get("test prompt"), "test response")
        finally:
            await semantrix.shutdown()

if __name__ == '__main__':
    unittest.main()
