"""Tests for Qdrant vector store."""

import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, Mock

from semantrix.vector_store.stores.qdrant import QdrantVectorStore, QDRANT_AVAILABLE
from semantrix.vector_store.base import DistanceMetric, VectorRecord

# Skip tests if qdrant-client is not installed or QDRANT_URL is not set
pytestmark = pytest.mark.skipif(
    not QDRANT_AVAILABLE or os.getenv("QDRANT_URL") is None,
    reason="Qdrant client not available or QDRANT_URL not set"
)

class TestQdrantVectorStore:
    """Test suite for QdrantVectorStore."""
    
    @pytest.fixture
    def mock_qdrant_client(self):
        """Mock Qdrant client."""
        with patch('qdrant_client.QdrantClient') as mock_client:
            # Mock collection operations
            mock_client.get_collections.return_value = Mock(
                collections=[]
            )
            
            # Mock search results
            mock_client.search.return_value = []
            
            # Mock scroll results
            mock_client.scroll.return_value = ([], None)
            
            # Mock retrieve results
            mock_client.retrieve.return_value = []
            
            # Mock collection info
            mock_client.get_collection.return_value = Mock(vectors_count=0)
            
            yield mock_client
    
    @pytest.fixture
    def qdrant_store(self, mock_qdrant_client):
        """Create a QdrantVectorStore instance for testing."""
        return QdrantVectorStore(
            dimension=128,
            metric=DistanceMetric.COSINE,
            namespace="test-namespace",
            url=os.getenv("QDRANT_URL", "http://localhost:6333")
        )
    
    @pytest.mark.asyncio
    async def test_add_vectors(self, qdrant_store, mock_qdrant_client):
        """Test adding vectors to the store."""
        mock_client = mock_qdrant_client.return_value
        
        # Test adding a single vector
        ids = await qdrant_store.add(
            vectors=[[0.1] * 128],
            documents=["test document"],
            metadatas=[{"source": "test"}]
        )
        
        assert len(ids) == 1
        assert isinstance(ids[0], str)
        mock_client.upsert.assert_called_once()
        
        # Test adding multiple vectors
        mock_client.upsert.reset_mock()
        
        vectors = [[0.2] * 128, [0.3] * 128]
        docs = ["doc1", "doc2"]
        metas = [{"source": "test1"}, {"source": "test2"}]
        
        ids = await qdrant_store.add(
            vectors=vectors,
            documents=docs,
            metadatas=metas
        )
        
        assert len(ids) == 2
        mock_client.upsert.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_vectors(self, qdrant_store, mock_qdrant_client):
        """Test retrieving vectors from the store."""
        mock_client = mock_qdrant_client.return_value
        
        # Mock retrieve response
        from qdrant_client.http.models import ScoredPoint, Record
        
        test_id = "test-id-123"
        test_vector = [0.1] * 128
        test_meta = {"source": "test", "document": "test doc"}
        
        mock_record = Record(
            id=test_id,
            payload=test_meta,
            vector=test_vector,
            version=1
        )
        mock_client.retrieve.return_value = [mock_record]
        
        # Test get by ID
        results = await qdrant_store.get(ids=test_id, include_vectors=True)
        
        assert len(results) == 1
        assert results[0].id == test_id
        assert results[0].document == "test doc"
        assert results[0].metadata["source"] == "test"
        assert results[0].embedding is not None
        
        # Test get by filter
        mock_client.scroll.return_value = ([mock_record], None)
        
        results = await qdrant_store.get(
            filter={"source": "test"},
            include_vectors=False
        )
        
        assert len(results) == 1
        assert results[0].id == test_id
        assert results[0].embedding is None
    
    @pytest.mark.asyncio
    async def test_search_vectors(self, qdrant_store, mock_qdrant_client):
        """Test vector similarity search."""
        mock_client = mock_qdrant_client.return_value
        
        # Mock search response
        from qdrant_client.http.models import ScoredPoint
        
        test_id = "test-id-123"
        test_vector = [0.1] * 128
        test_meta = {"source": "test", "document": "test doc"}
        
        mock_point = ScoredPoint(
            id=test_id,
            payload=test_meta,
            vector=test_vector,
            version=1,
            score=0.95
        )
        
        mock_client.search.return_value = [mock_point]
        
        # Perform search
        query_vector = [0.1] * 128
        results = await qdrant_store.search(
            query_vector=query_vector,
            k=1,
            filter={"source": "test"}
        )
        
        assert len(results) == 1
        assert results[0]["id"] == test_id
        assert results[0]["document"] == "test doc"
        assert results[0]["score"] == 0.95
        
        # Verify search was called with correct parameters
        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args[1]
        assert call_args["query_vector"] == query_vector
        assert call_args["limit"] == 1
    
    @pytest.mark.asyncio
    async def test_update_vectors(self, qdrant_store, mock_qdrant_client):
        """Test updating vectors in the store."""
        mock_client = mock_qdrant_client.return_value
        
        # Mock get response for existing record
        from qdrant_client.http.models import Record
        
        test_id = "test-id-123"
        test_vector = [0.1] * 128
        test_meta = {"source": "original", "document": "original doc"}
        
        mock_record = Record(
            id=test_id,
            payload=test_meta,
            vector=test_vector,
            version=1
        )
        
        # Mock get to return the existing record
        async def mock_get(*args, **kwargs):
            return [VectorRecord(
                id=test_id,
                embedding=test_vector,
                document=test_meta["document"],
                metadata={"source": test_meta["source"]},
                namespace="test-namespace"
            )]
        
        # Patch the get method
        with patch.object(qdrant_store, 'get', mock_get):
            # Update the vector
            await qdrant_store.update(
                ids=test_id,
                documents=["updated doc"],
                metadatas=[{"source": "updated"}]
            )
            
            # Verify upsert was called with updated data
            mock_client.upsert.assert_called_once()
            points = mock_client.upsert.call_args[1]["points"]
            assert len(points) == 1
            assert points[0].id == test_id
            assert points[0].payload["document"] == "updated doc"
            assert points[0].payload["source"] == "updated"
    
    @pytest.mark.asyncio
    async def test_delete_vectors(self, qdrant_store, mock_qdrant_client):
        """Test deleting vectors from the store."""
        mock_client = mock_qdrant_client.return_value
        
        # Test delete by ID
        test_id = "test-id-123"
        await qdrant_store.delete(ids=test_id)
        
        mock_client.delete.assert_called_once()
        assert mock_client.delete.call_args[1]["points_selector"] == [test_id]
        
        # Test delete by filter
        mock_client.delete.reset_mock()
        
        # Mock get to return some results
        async def mock_get(*args, **kwargs):
            return [
                VectorRecord(id="id1", embedding=None, document=None, metadata={}, namespace="test"),
                VectorRecord(id="id2", embedding=None, document=None, metadata={}, namespace="test")
            ]
        
        with patch.object(qdrant_store, 'get', mock_get):
            await qdrant_store.delete(filter={"source": "test"})
            
            # Should call delete with the IDs from get results
            mock_client.delete.assert_called_once()
            assert sorted(mock_client.delete.call_args[1]["points_selector"]) == ["id1", "id2"]
    
    @pytest.mark.asyncio
    async def test_count(self, qdrant_store, mock_qdrant_client):
        """Test counting vectors in the store."""
        mock_client = mock_qdrant_client.return_value
        
        # Mock collection info
        from qdrant_client.http.models import CollectionInfo
        mock_collection = CollectionInfo(
            status="green",
            optimizer_status=None,
            vectors_count=42,
            points_count=42,
            indexed_vectors_count=42,
            segments_count=1,
            config=None,
            payload_schema={}
        )
        mock_client.get_collection.return_value = mock_collection
        
        count = await qdrant_store.count()
        assert count == 42
    
    @pytest.mark.asyncio
    async def test_reset(self, qdrant_store, mock_qdrant_client):
        """Test resetting the store."""
        mock_client = mock_qdrant_client.return_value
        
        await qdrant_store.reset()
        
        # Should call delete_collection and create_collection
        mock_client.delete_collection.assert_called_once_with(
            collection_name=qdrant_store.collection_name
        )
        mock_client.create_collection.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_from_url(self):
        """Test creating a store from URL."""
        with patch('qdrant_client.QdrantClient') as mock_client:
            store = QdrantVectorStore.from_url(
                url="http://test-qdrant:6333",
                dimension=128,
                api_key="test-api-key"
            )
            
            assert store.dimension == 128
            mock_client.assert_called_once_with(
                url="http://test-qdrant:6333",
                api_key="test-api-key"
            )
    
    @pytest.mark.asyncio
    async def test_from_local(self):
        """Test creating a store with local storage."""
        with patch('qdrant_client.QdrantClient') as mock_client:
            store = QdrantVectorStore.from_local(
                path="/tmp/qdrant_test",
                dimension=128
            )
            
            assert store.dimension == 128
            mock_client.assert_called_once_with(
                location="/tmp/qdrant_test"
            )

# Run tests with: pytest tests/vector_store/test_qdrant.py -v
