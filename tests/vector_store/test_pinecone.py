"""Tests for Pinecone vector store."""

import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, Mock

from semantrix.vector_store.stores.pinecone import PineconeVectorStore, PINE_AVAILABLE
from semantrix.vector_store.base import DistanceMetric, VectorRecord

# Skip tests if pinecone-client is not installed or PINECONE_API_KEY is not set
pytestmark = pytest.mark.skipif(
    not PINE_AVAILABLE or os.getenv("PINECONE_API_KEY") is None,
    reason="Pinecone client not available or PINECONE_API_KEY not set"
)

class TestPineconeVectorStore:
    """Test cases for PineconeVectorStore."""
    
    @pytest.fixture
    def mock_pinecone(self):
        """Mock Pinecone client and index."""
        with patch('pinecone.init') as mock_init, \
             patch('pinecone.list_indexes', return_value=[]), \
             patch('pinecone.create_index'), \
             patch('pinecone.Index') as mock_index_class:
            
            # Set up mock index
            mock_index = MagicMock()
            mock_index_class.return_value = mock_index
            
            # Mock index stats
            mock_index.describe_index_stats.return_value = {
                'dimension': 3,
                'index_fullness': 0.0,
                'namespaces': {'test-ns': {'vector_count': 0}},
                'total_vector_count': 0
            }
            
            # Mock fetch response
            mock_fetch_response = MagicMock()
            mock_fetch_response.vectors = {}
            mock_index.fetch.return_value = mock_fetch_response
            
            # Mock query response
            mock_query_response = MagicMock()
            mock_query_response.matches = []
            mock_index.query.return_value = mock_query_response
            
            yield mock_init, mock_index_class, mock_index
    
    def test_initialization(self, mock_pinecone):
        """Test PineconeVectorStore initialization."""
        mock_init, mock_index_class, mock_index = mock_pinecone
        
        # Test with minimal parameters
        store = PineconeVectorStore(
            dimension=3,
            namespace="test-ns"
        )
        
        assert store.dimension == 3
        assert store.namespace == "test-ns"
        assert store.metric == DistanceMetric.COSINE
        
        # Verify Pinecone was initialized
        mock_init.assert_called_once()
        
        # Test with custom parameters
        store = PineconeVectorStore(
            dimension=768,
            metric=DistanceMetric.EUCLIDEAN,
            environment="us-east1-gcp",
            index_name="custom-index",
            api_key="test-key"
        )
        
        assert store.dimension == 768
        assert store.metric == DistanceMetric.EUCLIDEAN
        assert store.index_name == "custom-index"
    
    def test_from_api_key(self, mock_pinecone):
        """Test from_api_key classmethod."""
        mock_init, mock_index_class, mock_index = mock_pinecone
        
        store = PineconeVectorStore.from_api_key(
            api_key="test-key",
            dimension=3,
            environment="us-west1-gcp"
        )
        
        assert store.dimension == 3
        mock_init.assert_called_once()
    
    async def test_add_and_get(self, mock_pinecone):
        """Test adding and getting vectors."""
        _, _, mock_index = mock_pinecone
        
        store = PineconeVectorStore(dimension=3, namespace="test-ns")
        
        # Mock fetch response for get()
        mock_fetch_response = MagicMock()
        mock_fetch_response.vectors = {
            "vec1": MagicMock(
                id="vec1",
                values=[0.1, 0.2, 0.3],
                metadata={"document": "test doc", "key": "value"}
            )
        }
        mock_index.fetch.return_value = mock_fetch_response
        
        # Test add
        vector = [0.1, 0.2, 0.3]
        vector_id = await store.add(
            vector,
            document="test doc",
            metadata={"key": "value"},
            id="vec1"
        )
        
        # Verify upsert was called with correct data
        mock_index.upsert.assert_called_once()
        args, kwargs = mock_index.upsert.call_args
        assert kwargs["namespace"] == "test-ns"
        assert len(args[0]) == 1
        assert args[0][0][0] == "vec1"
        
        # Test get
        results = await store.get(ids=["vec1"], include_vectors=True)
        assert len(results) == 1
        assert results[0].id == "vec1"
        assert results[0].document == "test doc"
        assert results[0].metadata == {"key": "value", "document": "test doc"}
    
    async def test_search(self, mock_pinecone):
        """Test vector search."""
        _, _, mock_index = mock_pinecone
        
        store = PineconeVectorStore(dimension=3, namespace="test-ns")
        
        # Mock query response
        mock_match = MagicMock()
        mock_match.id = "vec1"
        mock_match.score = 0.95
        mock_match.values = [0.1, 0.2, 0.3]
        mock_match.metadata = {"document": "test doc", "key": "value"}
        
        mock_query_response = MagicMock()
        mock_query_response.matches = [mock_match]
        mock_index.query.return_value = mock_query_response
        
        # Test search
        query = [0.1, 0.2, 0.3]
        results = await store.search(
            query_vector=query,
            k=1,
            include_vectors=True
        )
        
        # Verify query was called with correct parameters
        mock_index.query.assert_called_once()
        args, kwargs = mock_index.query.call_args
        assert kwargs["vector"] == query
        assert kwargs["top_k"] == 1
        assert kwargs["namespace"] == "test-ns"
        
        # Verify results
        assert len(results) == 1
        assert results[0]["id"] == "vec1"
        assert results[0]["score"] == 0.95
        assert results[0]["document"] == "test doc"
    
    async def test_update(self, mock_pinecone):
        """Test updating vectors."""
        _, _, mock_index = mock_pinecone
        
        store = PineconeVectorStore(dimension=3, namespace="test-ns")
        
        # Mock fetch response for get()
        mock_fetch_response = MagicMock()
        mock_fetch_response.vectors = {
            "vec1": MagicMock(
                id="vec1",
                values=[0.1, 0.2, 0.3],
                metadata={"document": "old doc", "key": "old"}
            )
        }
        mock_index.fetch.return_value = mock_fetch_response
        
        # Test update
        await store.update(
            ids="vec1",
            document="new doc",
            metadata={"key": "new"}
        )
        
        # Verify upsert was called with updated data
        mock_index.upsert.assert_called_once()
        args, kwargs = mock_index.upsert.call_args
        assert kwargs["namespace"] == "test-ns"
        assert len(args[0]) == 1
        assert args[0][0][0] == "vec1"
        assert args[0][0][2]["document"] == "new doc"
        assert args[0][0][2]["key"] == "new"
    
    async def test_delete(self, mock_pinecone):
        """Test deleting vectors."""
        _, _, mock_index = mock_pinecone
        
        store = PineconeVectorStore(dimension=3, namespace="test-ns")
        
        # Test delete by ID
        await store.delete(ids=["vec1"])
        mock_index.delete.assert_called_once_with(ids=["vec1"], namespace="test-ns")
        
        # Reset mock
        mock_index.delete.reset_mock()
        
        # Test delete by filter
        mock_match = MagicMock()
        mock_match.id = "vec2"
        mock_query_response = MagicMock()
        mock_query_response.matches = [mock_match]
        mock_index.query.return_value = mock_query_response
        
        await store.delete(filter={"key": "value"})
        mock_index.delete.assert_called_once_with(ids=["vec2"], namespace="test-ns")
    
    async def test_count_and_reset(self, mock_pinecone):
        """Test count and reset operations."""
        _, _, mock_index = mock_pinecone
        
        store = PineconeVectorStore(dimension=3, namespace="test-ns")
        
        # Test count
        mock_stats = MagicMock()
        mock_stats.namespaces = {"test-ns": MagicMock(vector_count=5)}
        mock_index.describe_index_stats.return_value = mock_stats
        
        count = await store.count()
        assert count == 5
        
        # Test reset
        await store.reset()
        mock_index.delete.assert_called_once_with(delete_all=True, namespace="test-ns")
