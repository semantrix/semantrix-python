"""Tests for Chroma vector store implementation."""

import os
import tempfile
import uuid
from typing import List
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from unittest import IsolatedAsyncioTestCase

from semantrix.vector_store.stores.chroma import ChromaVectorStore
from semantrix.vector_store.base import DistanceMetric, VectorRecord

# Skip if chromadb is not installed
pytest.importorskip("chromadb")


class TestChromaVectorStore(IsolatedAsyncioTestCase):
    """Test cases for ChromaVectorStore."""
    
    async def test_chroma_initialization(self):
        """Test Chroma vector store initialization."""
        # Test in-memory initialization
        store = ChromaVectorStore(dimension=128)
        assert store.dimension == 128
        assert store.metric == DistanceMetric.COSINE
        assert store.in_memory is True
        assert store.persist_directory is None
        await store.close()
        
        # Test persistent initialization
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChromaVectorStore(
                dimension=256,
                metric=DistanceMetric.EUCLIDEAN,
                namespace="test",
                persist_directory=temp_dir,
                in_memory=False
            )
            assert store.dimension == 256
            assert store.metric == DistanceMetric.EUCLIDEAN
            assert store.namespace == "test"
            assert store.persist_directory == temp_dir
            assert store.in_memory is False
            await store.close()

    async def test_add_and_get_vectors(self):
        """Test adding and retrieving vectors."""
        store = ChromaVectorStore(dimension=3)
        
        # Add single vector
        vector = [0.1, 0.2, 0.3]
        vector_id = await store.add(vector, document="test")
        assert len(vector_id) == 1
        
        # Get the vector back
        results = await store.get(vector_id[0], include_vectors=True)
        assert len(results) == 1
        assert results[0].id == vector_id[0]
        assert results[0].document == "test"
        np.testing.assert_array_almost_equal(results[0].embedding, vector)
        
        # Add multiple vectors
        vectors = [[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        docs = ["doc1", "doc2"]
        metadatas = [{"source": "test1"}, {"source": "test2"}]
        ids = ["id1", "id2"]
        
        returned_ids = await store.add(vectors, documents=docs, metadatas=metadatas, ids=ids)
        assert returned_ids == ids
        
        # Get multiple vectors
        results = await store.get(ids, include_vectors=True)
        assert len(results) == 2
        assert {r.id for r in results} == set(ids)
        assert {r.document for r in results} == set(docs)
        assert {r.metadata["source"] for r in results} == {"test1", "test2"}
        
        await store.close()

    async def test_search(self):
        """Test similarity search."""
        store = ChromaVectorStore(dimension=3)
        
        # Add test vectors
        vectors = [
            [0.1, 0.1, 0.1],  # Should be most similar to query
            [0.9, 0.9, 0.9],
            [0.5, 0.5, 0.5],
        ]
        docs = [f"doc{i}" for i in range(3)]
        vector_ids = await store.add(vectors, documents=docs)
        
        # Search for vector similar to the first one
        query = [0.15, 0.15, 0.15]
        results = await store.search(query, k=2)
        
        assert len(results) == 2
        # The first vector should be most similar to the query
        assert results[0]["id"] == vector_ids[0]  # Most similar should be first
        assert results[0]["score"] >= results[1]["score"]  # Results should be sorted by score
        
        await store.close()

    async def test_update(self):
        """Test updating vectors."""
        store = ChromaVectorStore(dimension=3)
        
        # Add a vector
        vector = [0.1, 0.2, 0.3]
        vector_id = await store.add(vector, document="original", metadata={"source": "test"})
        
        # Update the vector
        new_vector = [0.4, 0.5, 0.6]
        await store.update(vector_id[0], vectors=new_vector, documents="updated", metadatas={"source": "updated"})
        
        # Verify the update
        results = await store.get(vector_id[0], include_vectors=True)
        assert len(results) == 1
        np.testing.assert_array_almost_equal(results[0].embedding, new_vector)
        assert results[0].document == "updated"
        assert results[0].metadata["source"] == "updated"
        
        await store.close()

    async def test_delete(self):
        """Test deleting vectors."""
        store = ChromaVectorStore(dimension=3)
        
        # Add vectors
        vectors = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        docs = ["doc1", "doc2"]
        ids = await store.add(vectors, documents=docs)
        
        # Verify they exist
        results = await store.get(ids, include_vectors=True)
        assert len(results) == 2
        
        # Delete one vector
        await store.delete(ids[0])
        
        # Verify it's gone
        results = await store.get(ids[0], include_vectors=True)
        assert len(results) == 0
        
        # Verify the other still exists
        results = await store.get(ids[1], include_vectors=True)
        assert len(results) == 1
        
        await store.close()

    async def test_persistence(self):
        """Test persistence functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create store with persistence
            store = ChromaVectorStore(
                dimension=3,
                persist_directory=temp_dir,
                in_memory=False
            )
            
            # Add data
            vector = [0.1, 0.2, 0.3]
            vector_id = await store.add(vector, document="test")
            
            # Close and recreate
            await store.close()
            
            store2 = ChromaVectorStore(
                dimension=3,
                persist_directory=temp_dir,
                in_memory=False
            )
            
            # Verify data persists
            results = await store2.get(vector_id[0], include_vectors=True)
            assert len(results) == 1
            assert results[0].document == "test"
            
            await store2.close()

    async def test_from_server_connection(self):
        """Test server connection functionality."""
        # Mock the ChromaDB client for server connection
        with patch('chromadb.Client') as mock_client:
            mock_collection = MagicMock()
            mock_client.return_value.get_or_create_collection.return_value = mock_collection
            
            # Test server connection
            store = ChromaVectorStore.from_server(
                dimension=128,
                host="localhost",
                port=8000
            )
            
            assert store.dimension == 128
            assert store._client is not None
            
            await store.close()


if __name__ == '__main__':
    pytest.main([__file__])
