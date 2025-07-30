"""Tests for Chroma vector store implementation."""

import os
import tempfile
import uuid
from typing import List
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from semantrix.vector_store.stores.chroma import ChromaVectorStore
from semantrix.vector_store.base import DistanceMetric, VectorRecord

# Skip if chromadb is not installed
pytest.importorskip("chromadb")


def test_chroma_initialization():
    """Test Chroma vector store initialization."""
    # Test in-memory initialization
    store = ChromaVectorStore(dimension=128)
    assert store.dimension == 128
    assert store.metric == DistanceMetric.COSINE
    assert store.in_memory is True
    assert store.persist_directory is None
    store.close()
    
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
        store.close()


def test_add_and_get_vectors():
    """Test adding and retrieving vectors."""
    store = ChromaVectorStore(dimension=3)
    
    # Add single vector
    vector = [0.1, 0.2, 0.3]
    vector_id = store.add(vector, document="test")
    assert len(vector_id) == 1
    
    # Get the vector back
    results = store.get(vector_id[0], include_vectors=True)
    assert len(results) == 1
    assert results[0].id == vector_id[0]
    assert results[0].document == "test"
    np.testing.assert_array_almost_equal(results[0].embedding, vector)
    
    # Add multiple vectors
    vectors = [[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
    docs = ["doc1", "doc2"]
    metadatas = [{"source": "test1"}, {"source": "test2"}]
    ids = ["id1", "id2"]
    
    returned_ids = store.add(vectors, documents=docs, metadatas=metadatas, ids=ids)
    assert returned_ids == ids
    
    # Get multiple vectors
    results = store.get(ids, include_vectors=True)
    assert len(results) == 2
    assert {r.id for r in results} == set(ids)
    assert {r.document for r in results} == set(docs)
    assert {r.metadata["source"] for r in results} == {"test1", "test2"}
    
    store.close()


def test_search():
    """Test similarity search."""
    store = ChromaVectorStore(dimension=3)
    
    # Add test vectors
    vectors = [
        [0.1, 0.1, 0.1],  # Should be most similar to query
        [0.9, 0.9, 0.9],
        [0.5, 0.5, 0.5],
    ]
    docs = [f"doc{i}" for i in range(3)]
    store.add(vectors, documents=docs)
    
    # Search for vector similar to the first one
    query = [0.15, 0.15, 0.15]
    results = store.search(query, k=2)
    
    assert len(results) == 2
    assert results[0]["id"] == store.add(vectors[0])[0]  # Most similar should be first
    assert results[0]["score"] >= results[1]["score"]  # Results should be sorted by score
    
    store.close()


def test_update():
    """Test updating vectors."""
    store = ChromaVectorStore(dimension=3)
    
    # Add initial vector
    vector_id = store.add([0.1, 0.2, 0.3], document="old_doc", metadatas={"key": "old_value"})[0]
    
    # Update the vector
    store.update(
        ids=vector_id,
        vectors=[0.4, 0.5, 0.6],
        documents="new_doc",
        metadatas={"key": "new_value"}
    )
    
    # Verify update
    result = store.get(vector_id)[0]
    np.testing.assert_array_almost_equal(result.embedding, [0.4, 0.5, 0.6])
    assert result.document == "new_doc"
    assert result.metadata["key"] == "new_value"
    
    store.close()


def test_delete():
    """Test deleting vectors."""
    store = ChromaVectorStore(dimension=3)
    
    # Add test vectors
    ids = ["id1", "id2", "id3"]
    vectors = [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]]
    metadatas = [{"group": "a"}, {"group": "b"}, {"group": "a"}]
    
    store.add(vectors, ids=ids, metadatas=metadatas)
    
    # Delete by ID
    store.delete(ids=["id1"])
    assert len(store.get(["id1"])) == 0
    assert len(store.get(["id2", "id3"])) == 2
    
    # Delete by filter
    store.delete(filter={"group": "a"})
    assert len(store.get(["id3"])) == 0
    assert len(store.get(["id2"])) == 1  # Only id2 should remain
    
    store.close()


def test_persistence():
    """Test persistence to disk."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create and populate store
        store1 = ChromaVectorStore(
            dimension=3,
            persist_directory=temp_dir,
            in_memory=False
        )
        vector_id = store1.add([0.1, 0.2, 0.3], document="test")[0]
        store1.close()
        
        # Reopen and check data
        store2 = ChromaVectorStore(
            dimension=3,
            persist_directory=temp_dir,
            in_memory=False
        )
        results = store2.get([vector_id])
        assert len(results) == 1
        assert results[0].document == "test"
        store2.close()


def test_from_server_connection():
    """Test creating a ChromaVectorStore with server connection."""
    # Mock the Chroma client to avoid actual network calls in tests
    with patch('chromadb.Client') as mock_client:
        # Create a mock collection
        mock_collection = MagicMock()
        mock_client.return_value.get_or_create_collection.return_value = mock_collection
        
        # Test with minimal required parameters
        store = ChromaVectorStore.from_server(
            dimension=768,
            host="test-server"
        )
        
        # Verify client was created with correct settings
        _, client_kwargs = mock_client.call_args
        settings = client_kwargs['settings']
        assert settings.chroma_api_impl == 'rest'
        assert settings.chroma_server_host == 'test-server'
        assert settings.chroma_server_http_port == 8000
        assert settings.chroma_server_ssl is False
        
        # Test with all parameters
        store = ChromaVectorStore.from_server(
            dimension=768,
            host="secure-server",
            port=8443,
            ssl=True,
            api_key="test-key",
            namespace="test-ns",
            metric=DistanceMetric.EUCLIDEAN,
            custom_setting="value"
        )
        
        # Verify all settings are passed through
        _, client_kwargs = mock_client.call_args
        settings = client_kwargs['settings']
        assert settings.chroma_server_host == 'secure-server'
        assert settings.chroma_server_http_port == 8443
        assert settings.chroma_server_ssl is True
        assert settings.chroma_server_auth_credentials == 'test-key'
        assert settings.custom_setting == 'value'
        
        # Verify collection was created with correct parameters
        mock_client.return_value.get_or_create_collection.assert_called_with(
            name="semantrix_test-ns_d768_l2",
            metadata={"hnsw:space": "l2"},
            embedding_function=None
        )
        
        # Clean up
        store.close()


if __name__ == "__main__":
    pytest.main([__file__])
