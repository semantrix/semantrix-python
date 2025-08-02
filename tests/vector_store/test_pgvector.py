"""Tests for pgvector vector store."""

import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch, Mock

from semantrix.vector_store.stores.pgvector import PgVectorStore, PGVECTOR_AVAILABLE
from semantrix.vector_store.base import DistanceMetric, VectorRecord

# Skip tests if psycopg2-binary package is not installed or DATABASE_URL is not set
pytestmark = pytest.mark.skipif(
    not PGVECTOR_AVAILABLE or os.getenv("DATABASE_URL") is None,
    reason="psycopg2-binary not available or DATABASE_URL not set"
)

class TestPgVectorStore:
    """Test suite for PgVectorStore."""
    
    @pytest.fixture
    def mock_pg_connection(self):
        """Mock PostgreSQL connection."""
        with patch('psycopg2.connect') as mock_connect:
            # Mock connection and cursor
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_connect.return_value = mock_conn
            mock_conn.cursor.return_value = mock_cursor
            
            # Mock execute to return different results based on query
            def mock_execute(query, params=None):
                if query and "COUNT" in query:
                    mock_cursor.fetchone.return_value = (42,)  # Return count
                elif query and "SELECT" in query and "vector" in query:
                    # Return mock search results
                    mock_cursor.fetchall.return_value = [
                        ("test-id", "test doc", '{"source": "test"}', "(0.1,0.2,0.3)", 0.1)
                    ]
                elif query and "SELECT" in query:
                    # Return mock get results
                    mock_cursor.fetchall.return_value = [
                        ("test-id", "test doc", '{"source": "test"}', "(0.1,0.2,0.3)")
                    ]
                
                return mock_cursor
            
            mock_cursor.execute.side_effect = mock_execute
            
            yield mock_conn, mock_cursor
    
    @pytest.fixture
    def pg_store(self, mock_pg_connection):
        """Create a PgVectorStore instance for testing."""
        mock_conn, _ = mock_pg_connection
        
        # Create a test instance with mock connection
        store = PgVectorStore(
            dimension=3,
            metric=DistanceMetric.COSINE,
            namespace="test-namespace"
        )
        
        # Replace the connection with our mock
        store._connection = mock_conn
        
        return store
    
    @pytest.mark.asyncio
    async def test_add_vectors(self, pg_store, mock_pg_connection):
        """Test adding vectors to the store."""
        mock_conn, mock_cursor = mock_pg_connection
        
        # Test adding a single vector
        ids = await pg_store.add(
            vectors=[[0.1, 0.2, 0.3]],
            documents=["test document"],
            metadatas=[{"source": "test"}]
        )
        
        assert len(ids) == 1
        assert isinstance(ids[0], str)
        mock_cursor.execute.assert_called()
        
        # Verify the execute_values was called with correct parameters
        assert mock_cursor.execute.call_count >= 1
        
        # Test adding multiple vectors
        mock_cursor.reset_mock()
        
        vectors = [[0.2, 0.3, 0.4], [0.3, 0.4, 0.5]]
        docs = ["doc1", "doc2"]
        metas = [{"source": "test1"}, {"source": "test2"}]
        
        ids = await pg_store.add(
            vectors=vectors,
            documents=docs,
            metadatas=metas
        )
        
        assert len(ids) == 2
        assert mock_cursor.execute.call_count >= 1
    
    @pytest.mark.asyncio
    async def test_get_vectors(self, pg_store, mock_pg_connection):
        """Test retrieving vectors from the store."""
        mock_conn, mock_cursor = mock_pg_connection
        
        # Test get by ID
        results = await pg_store.get(ids="test-id", include_vectors=True)
        
        assert len(results) == 1
        assert results[0].id == "test-id"
        assert results[0].document == "test doc"
        assert results[0].metadata["source"] == "test"
        assert len(results[0].embedding) == 3  # Dimension is 3 in test
        
        # Test get by filter
        mock_cursor.reset_mock()
        
        results = await pg_store.get(
            filter={"source": "test"},
            include_vectors=False
        )
        
        assert len(results) == 1
        assert results[0].id == "test-id"
        assert results[0].embedding is None
    
    @pytest.mark.asyncio
    async def test_search_vectors(self, pg_store, mock_pg_connection):
        """Test vector similarity search."""
        mock_conn, mock_cursor = mock_pg_connection
        
        # Perform search
        query_vector = [0.1, 0.2, 0.3]
        results = await pg_store.search(
            query_vector=query_vector,
            k=1,
            filter={"source": "test"}
        )
        
        assert len(results) == 1
        assert results[0]["id"] == "test-id"
        assert results[0]["document"] == "test doc"
        assert 0 <= results[0]["score"] <= 1.0
        
        # Verify the query was built correctly
        mock_cursor.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_vectors(self, pg_store, mock_pg_connection):
        """Test updating vectors in the store."""
        mock_conn, mock_cursor = mock_pg_connection
        
        # Mock existing document
        mock_cursor.fetchall.return_value = [
            ("test-id", "original doc", '{"source": "original"}', "(0.1,0.2,0.3)")
        ]
        
        # Update the vector
        await pg_store.update(
            ids="test-id",
            documents=["updated doc"],
            metadatas=[{"source": "updated"}]
        )
        
        # Verify update was called
        assert mock_cursor.execute.call_count >= 1
        
        # Check that execute_values was called with the update query
        update_calls = [
            call for call in mock_cursor.method_calls 
            if call[0] == 'execute' and 'UPDATE' in str(call[1])
        ]
        assert len(update_calls) > 0
    
    @pytest.mark.asyncio
    async def test_delete_vectors(self, pg_store, mock_pg_connection):
        """Test deleting vectors from the store."""
        mock_conn, mock_cursor = mock_pg_connection
        
        # Test delete by ID
        await pg_store.delete(ids="test-id")
        
        # Verify delete query was executed
        delete_calls = [
            call for call in mock_cursor.method_calls 
            if call[0] == 'execute' and 'DELETE' in str(call[1])
        ]
        assert len(delete_calls) > 0
        
        # Test delete by filter
        mock_cursor.reset_mock()
        
        await pg_store.delete(filter={"source": "test"})
        
        # Verify delete with filter was executed
        delete_calls = [
            call for call in mock_cursor.method_calls 
            if call[0] == 'execute' and 'DELETE' in str(call[1])
        ]
        assert len(delete_calls) > 0
    
    @pytest.mark.asyncio
    async def test_count(self, pg_store, mock_pg_connection):
        """Test counting vectors in the store."""
        mock_conn, mock_cursor = mock_pg_connection
        
        count = await pg_store.count()
        assert count == 42
        
        # Verify the query was executed
        mock_cursor.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_reset(self, pg_store, mock_pg_connection):
        """Test resetting the store."""
        mock_conn, mock_cursor = mock_pg_connection
        
        await pg_store.reset()
        
        # Verify DROP TABLE and CREATE TABLE were called
        drop_calls = [
            call for call in mock_cursor.method_calls 
            if call[0] == 'execute' and 'DROP TABLE' in str(call[1])
        ]
        assert len(drop_calls) > 0
        
        create_calls = [
            call for call in mock_cursor.method_calls 
            if call[0] == 'execute' and 'CREATE TABLE' in str(call[1])
        ]
        assert len(create_calls) > 0
    
    @pytest.mark.asyncio
    async def test_close(self, pg_store, mock_pg_connection):
        """Test closing the connection."""
        mock_conn, mock_cursor = mock_pg_connection
        
        await pg_store.close()
        
        # Verify connection was closed
        mock_conn.close.assert_called_once()

# Run tests with: pytest tests/vector_store/test_pgvector.py -v
