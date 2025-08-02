"""Tests for Redis vector store."""

import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, Mock

from semantrix.vector_store.stores.redis import RedisVectorStore, REDIS_AVAILABLE
from semantrix.vector_store.base import DistanceMetric, VectorRecord

# Skip tests if redis package is not installed or REDIS_URL is not set
pytestmark = pytest.mark.skipif(
    not REDIS_AVAILABLE or os.getenv("REDIS_URL") is None,
    reason="Redis client not available or REDIS_URL not set"
)

class TestRedisVectorStore:
    """Test suite for RedisVectorStore."""
    
    @pytest.fixture
    def mock_redis_client(self):
        """Mock Redis client."""
        with patch('redis.Redis') as mock_redis:
            # Mock Redis JSON commands
            mock_redis.json = MagicMock()
            mock_redis.ft = MagicMock()
            mock_redis.pipeline = MagicMock(return_value=mock_redis)
            mock_redis.execute = MagicMock()
            
            # Mock search results
            mock_index = MagicMock()
            mock_index.info.return_value = {"num_docs": 0}
            mock_redis.ft.return_value = mock_index
            
            yield mock_redis
    
    @pytest.fixture
    def redis_store(self, mock_redis_client):
        """Create a RedisVectorStore instance for testing."""
        return RedisVectorStore(
            dimension=128,
            metric=DistanceMetric.COSINE,
            namespace="test-namespace",
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379")
        )
    
    @pytest.mark.asyncio
    async def test_add_vectors(self, redis_store, mock_redis_client):
        """Test adding vectors to the store."""
        # Mock JSON set to return success
        redis_store._client.json().set.return_value = True
        
        # Test adding a single vector
        ids = await redis_store.add(
            vectors=[[0.1] * 128],
            documents=["test document"],
            metadatas=[{"source": "test"}]
        )
        
        assert len(ids) == 1
        assert isinstance(ids[0], str)
        redis_store._client.json().set.assert_called_once()
        
        # Test adding multiple vectors
        redis_store._client.json().set.reset_mock()
        
        vectors = [[0.2] * 128, [0.3] * 128]
        docs = ["doc1", "doc2"]
        metas = [{"source": "test1"}, {"source": "test2"}]
        
        ids = await redis_store.add(
            vectors=vectors,
            documents=docs,
            metadatas=metas
        )
        
        assert len(ids) == 2
        assert redis_store._client.json().set.call_count == 2
    
    @pytest.mark.asyncio
    async def test_get_vectors(self, redis_store, mock_redis_client):
        """Test retrieving vectors from the store."""
        # Mock JSON get response
        test_id = "test-id-123"
        test_doc = {
            "document": "test doc",
            "source": "test",
            "namespace": "test-namespace",
            "vector": np.array([0.1] * 128, dtype=np.float32).tobytes(),
            "timestamp": 1234567890
        }
        redis_store._client.json().get.return_value = test_doc
        
        # Test get by ID
        results = await redis_store.get(ids=test_id, include_vectors=True)
        
        assert len(results) == 1
        assert results[0].id == test_id
        assert results[0].document == "test doc"
        assert results[0].metadata["source"] == "test"
        assert results[0].embedding is not None
        
        # Test get by filter
        mock_search_result = MagicMock()
        mock_search_result.docs = [
            MagicMock(id=f"vec:{test_id}", payload=json.dumps(test_doc))
        ]
        redis_store._client.ft().search.return_value = mock_search_result
        
        results = await redis_store.get(
            filter={"source": "test"},
            include_vectors=False
        )
        
        assert len(results) == 1
        assert results[0].id == test_id
        assert results[0].embedding is None
    
    @pytest.mark.asyncio
    async def test_search_vectors(self, redis_store, mock_redis_client):
        """Test vector similarity search."""
        # Mock search response
        test_id = "test-id-123"
        test_doc = {
            "document": "test doc",
            "source": "test",
            "namespace": "test-namespace",
            "vector": np.array([0.1] * 128, dtype=np.float32).tobytes(),
            "timestamp": 1234567890
        }
        
        mock_search_result = MagicMock()
        mock_doc = MagicMock()
        mock_doc.id = f"vec:{test_id}"
        mock_doc.payload = json.dumps(test_doc)
        mock_doc.vector = test_doc["vector"]
        mock_doc.score = 0.9
        mock_search_result.docs = [mock_doc]
        
        redis_store._client.ft().search.return_value = mock_search_result
        
        # Perform search
        query_vector = [0.1] * 128
        results = await redis_store.search(
            query_vector=query_vector,
            k=1,
            filter={"source": "test"}
        )
        
        assert len(results) == 1
        assert results[0]["id"] == test_id
        assert results[0]["document"] == "test doc"
        assert 0 <= results[0]["score"] <= 1.0
        
        # Verify search was called with correct parameters
        redis_store._client.ft().search.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_vectors(self, redis_store, mock_redis_client):
        """Test updating vectors in the store."""
        # Mock existing document
        test_id = "test-id-123"
        redis_store._client.json().get.return_value = {
            "document": "original doc",
            "source": "original",
            "namespace": "test-namespace",
            "vector": np.array([0.1] * 128, dtype=np.float32).tobytes(),
            "timestamp": 1234567890
        }
        
        # Update the vector
        await redis_store.update(
            ids=test_id,
            documents=["updated doc"],
            metadatas=[{"source": "updated"}]
        )
        
        # Verify update was called with correct data
        redis_store._client.json().set.assert_called_once()
        call_args = redis_store._client.json().set.call_args[0]
        assert call_args[0] == f"vec:{test_id}"
        assert call_args[1] == "$"
        
        # Check that the document was updated
        updated_doc = call_args[2]
        assert updated_doc["document"] == "updated doc"
        assert updated_doc["source"] == "updated"
        assert updated_doc["namespace"] == "test-namespace"
    
    @pytest.mark.asyncio
    async def test_delete_vectors(self, redis_store, mock_redis_client):
        """Test deleting vectors from the store."""
        # Test delete by ID
        test_id = "test-id-123"
        await redis_store.delete(ids=test_id)
        
        redis_store._client.delete.assert_called_once_with(f"vec:{test_id}")
        
        # Test delete by filter
        redis_store._client.delete.reset_mock()
        
        # Mock search results for filter
        mock_search_result = MagicMock()
        mock_search_result.docs = [
            MagicMock(id="vec:id1"),
            MagicMock(id="vec:id2")
        ]
        redis_store._client.ft().search.return_value = mock_search_result
        
        await redis_store.delete(filter={"source": "test"})
        
        # Should call delete with the IDs from search results
        redis_store._client.delete.assert_called_once_with("vec:id1", "vec:id2")
    
    @pytest.mark.asyncio
    async def test_count(self, redis_store, mock_redis_client):
        """Test counting vectors in the store."""
        # Mock index info
        mock_index = MagicMock()
        mock_index.info.return_value = {"num_docs": 42}
        redis_store._client.ft.return_value = mock_index
        
        count = await redis_store.count()
        assert count == 42
    
    @pytest.mark.asyncio
    async def test_reset(self, redis_store, mock_redis_client):
        """Test resetting the store."""
        mock_index = MagicMock()
        redis_store._client.ft.return_value = mock_index
        
        await redis_store.reset()
        
        # Should call dropindex and create_index
        mock_index.dropindex.assert_called_once_with(delete_documents=True)
        mock_index.create_index.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_from_url(self):
        """Test creating a store from URL."""
        with patch('redis.Redis') as mock_redis:
            store = RedisVectorStore.from_url(
                url="redis://test-redis:6379",
                dimension=128
            )
            
            assert store.dimension == 128
            mock_redis.assert_called_once()
            
            # Check that the URL was passed correctly
            call_args = mock_redis.call_args[1]
            assert call_args["host"] == "test-redis"
            assert call_args["port"] == 6379

# Run tests with: pytest tests/vector_store/test_redis.py -v
