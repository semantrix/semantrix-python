"""Tests for Milvus vector store."""

import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, Mock, call

from semantrix.vector_store.stores.milvus import MilvusVectorStore, MILVUS_AVAILABLE
from semantrix.vector_store.base import DistanceMetric, VectorRecord

# Skip tests if pymilvus package is not installed or MILVUS_URI is not set
pytestmark = pytest.mark.skipif(
    not MILVUS_AVAILABLE or os.getenv("MILVUS_URI") is None,
    reason="pymilvus not available or MILVUS_URI not set"
)

class TestMilvusVectorStore:
    """Test suite for MilvusVectorStore."""
    
    @pytest.fixture
    def mock_milvus_connection(self):
        """Mock Milvus connection and collection."""
        with patch('pymilvus.connections'), \
             patch('pymilvus.utility'), \
             patch('pymilvus.Collection') as mock_collection:
            
            # Mock collection methods
            mock_col = MagicMock()
            mock_collection.return_value = mock_col
            
            # Mock query results
            mock_col.query.return_value = [
                {
                    "id": "test-id",
                    "document": "test doc",
                    "metadata": '{"source": "test"}',
                    "vector": [0.1, 0.2, 0.3]
                }
            ]
            
            # Mock search results
            class MockHit:
                def __init__(self, id_, score, entity):
                    self.id = id_
                    self.score = score
                    self.entity = entity
            
            mock_hit = MockHit(
                id_="test-id",
                score=0.1,
                entity={
                    "id": "test-id",
                    "document": "test doc",
                    "metadata": '{"source": "test"}',
                    "vector": [0.1, 0.2, 0.3]
                }
            )
            mock_col.search.return_value = [[mock_hit]]
            
            # Mock count
            mock_col.get_replica_info.return_value.row_count = 42
            
            yield mock_col
    
    @pytest.fixture
    def milvus_store(self, mock_milvus_connection):
        """Create a MilvusVectorStore instance for testing."""
        # Create a test instance with mock connection
        store = MilvusVectorStore(
            dimension=3,
            metric=DistanceMetric.COSINE,
            namespace="test-namespace"
        )
        
        # Replace the collection with our mock
        store._collection = mock_milvus_connection
        
        return store
    
    @pytest.mark.asyncio
    async def test_add_vectors(self, milvus_store, mock_milvus_connection):
        """Test adding vectors to the store."""
        # Test adding a single vector
        ids = await milvus_store.add(
            vectors=[[0.1, 0.2, 0.3]],
            documents=["test document"],
            metadatas=[{"source": "test"}]
        )
        
        assert len(ids) == 1
        assert isinstance(ids[0], str)
        
        # Verify insert was called with correct parameters
        mock_milvus_connection.insert.assert_called_once()
        insert_args = mock_milvus_connection.insert.call_args[0][0]
        assert "id" in insert_args
        assert "vector" in insert_args
        assert "document" in insert_args
        assert "metadata" in insert_args
        assert "namespace" in insert_args
        
        # Test adding multiple vectors
        mock_milvus_connection.reset_mock()
        
        vectors = [[0.2, 0.3, 0.4], [0.3, 0.4, 0.5]]
        docs = ["doc1", "doc2"]
        metas = [{"source": "test1"}, {"source": "test2"}]
        
        ids = await milvus_store.add(
            vectors=vectors,
            documents=docs,
            metadatas=metas
        )
        
        assert len(ids) == 2
        mock_milvus_connection.insert.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_vectors(self, milvus_store, mock_milvus_connection):
        """Test retrieving vectors from the store."""
        # Test get by ID
        results = await milvus_store.get(ids="test-id", include_vectors=True)
        
        assert len(results) == 1
        assert results[0].id == "test-id"
        assert results[0].document == "test doc"
        assert results[0].metadata["source"] == "test"
        assert len(results[0].embedding) == 3  # Dimension is 3 in test
        
        # Verify query was called with correct parameters
        mock_milvus_connection.query.assert_called_once()
        
        # Test get by filter
        mock_milvus_connection.reset_mock()
        
        results = await milvus_store.get(
            filter={"source": "test"},
            include_vectors=False
        )
        
        assert len(results) == 1
        assert results[0].id == "test-id"
        assert results[0].embedding is None
    
    @pytest.mark.asyncio
    async def test_search_vectors(self, milvus_store, mock_milvus_connection):
        """Test vector similarity search."""
        # Perform search
        query_vector = [0.1, 0.2, 0.3]
        results = await milvus_store.search(
            query_vector=query_vector,
            k=1,
            filter={"source": "test"}
        )
        
        assert len(results) == 1
        assert results[0]["id"] == "test-id"
        assert results[0]["document"] == "test doc"
        assert 0 <= results[0]["score"] <= 1.0
        
        # Verify search was called with correct parameters
        mock_milvus_connection.search.assert_called_once()
        args, kwargs = mock_milvus_connection.search.call_args
        assert args[0] == [query_vector]  # Query vector
        assert kwargs["limit"] == 1  # k=1
        assert "expr" in kwargs  # Filter expression should be present
    
    @pytest.mark.asyncio
    async def test_update_vectors(self, milvus_store, mock_milvus_connection):
        """Test updating vectors in the store."""
        # Mock existing document
        mock_milvus_connection.query.return_value = [
            {
                "id": "test-id",
                "document": "original doc",
                "metadata": '{"source": "original"}',
                "vector": [0.1, 0.2, 0.3]
            }
        ]
        
        # Update the vector
        await milvus_store.update(
            ids="test-id",
            documents=["updated doc"],
            metadatas=[{"source": "updated"}]
        )
        
        # Verify upsert was called with updated data
        mock_milvus_connection.upsert.assert_called_once()
        upsert_data = mock_milvus_connection.upsert.call_args[0][0]
        
        # Check that the document and metadata were updated
        assert upsert_data["document"] == ["updated doc"]
        assert '{"source": "updated"}' in upsert_data["metadata"]
    
    @pytest.mark.asyncio
    async def test_delete_vectors(self, milvus_store, mock_milvus_connection):
        """Test deleting vectors from the store."""
        # Test delete by ID
        await milvus_store.delete(ids="test-id")
        
        # Verify delete was called with correct expression
        mock_milvus_connection.delete.assert_called_once()
        expr = mock_milvus_connection.delete.call_args[0][0]
        assert "id in ['test-id']" in expr
        assert "namespace == 'test-namespace'" in expr
        
        # Test delete by filter
        mock_milvus_connection.reset_mock()
        
        await milvus_store.delete(filter={"source": "test"})
        
        # Verify delete with filter was called
        mock_milvus_connection.delete.assert_called_once()
        expr = mock_milvus_connection.delete.call_args[0][0]
        assert "json_contains_any(metadata['source']" in expr
    
    @pytest.mark.asyncio
    async def test_count(self, milvus_store, mock_milvus_connection):
        """Test counting vectors in the store."""
        # Mock the query result for count with namespace
        mock_milvus_connection.query.return_value = [{"id": "1"}, {"id": "2"}]
        
        # Test count with namespace
        count = await milvus_store.count()
        assert count == 2  # Should return the length of query result
        
        # Test count without namespace
        milvus_store.namespace = None
        count = await milvus_store.count()
        assert count == 42  # Should return row_count from get_replica_info
    
    @pytest.mark.asyncio
    async def test_reset(self, milvus_store, mock_milvus_connection):
        """Test resetting the store."""
        with patch('pymilvus.utility.drop_collection') as mock_drop_collection:
            await milvus_store.reset()
            
            # Verify drop_collection and _ensure_collection were called
            mock_drop_collection.assert_called_once_with(milvus_store.collection_name)
            mock_milvus_connection.load.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_close(self, milvus_store, mock_milvus_connection):
        """Test closing the connection."""
        await milvus_store.close()
        
        # Verify release and disconnect were called
        mock_milvus_connection.release.assert_called_once()
        
        # Verify disconnect was called
        from pymilvus import connections
        connections.disconnect.assert_called_once_with("default")

# Run tests with: pytest tests/vector_store/test_milvus.py -v
