import pytest
from unittest.mock import Mock, AsyncMock

from semantrix.core.cache import Semantrix
from semantrix.exceptions import (CacheOperationError, VectorOperationError, ValidationError, OperationError)
from semantrix.utils.wal import OperationType
from semantrix.utils.twophase import TwoPhaseOperation

@pytest.fixture
def mock_cache_store():
    return AsyncMock()

@pytest.fixture
def mock_vector_store():
    return AsyncMock()

@pytest.fixture
def mock_embedder():
    return AsyncMock()

@pytest.fixture
def semantrix_instance(mock_cache_store, mock_vector_store, mock_embedder):
    return Semantrix(
        cache_store=mock_cache_store,
        vector_store=mock_vector_store,
        embedder=mock_embedder,
        enable_wal=False,  # Disable WAL for simpler testing
        enable_2pc=False
    )

@pytest.mark.asyncio
async def test_set_operation_raises_operation_error_on_failure(semantrix_instance, mock_cache_store):
    """Test that a failed set operation raises OperationError."""
    mock_cache_store.add.side_effect = Exception("Cache write failed")
    semantrix_instance._enable_2pc = False # Direct operation

    with pytest.raises(OperationError, match="Failed to execute operation atomically"):
        await semantrix_instance.set("test_prompt", "test_response")

@pytest.mark.asyncio
async def test_get_operation_raises_cache_error_on_failure(semantrix_instance, mock_cache_store):
    """Test that a failed get operation raises CacheOperationError."""
    mock_cache_store.get_exact.side_effect = CacheOperationError("Cache read failed")

    with pytest.raises(CacheOperationError, match="Cache read failed"):
        await semantrix_instance.get("test_prompt")

@pytest.mark.asyncio
async def test_semantic_search_raises_vector_error_on_failure(semantrix_instance, mock_vector_store):
    """Test that a failed semantic search raises VectorOperationError."""
    mock_vector_store.search.side_effect = VectorOperationError("Vector search failed")
    semantrix_instance.embedder.embed.return_value = [0.1, 0.2, 0.3] # Mock embedding

    with pytest.raises(VectorOperationError, match="Vector search failed"):
        await semantrix_instance.semantic_search("query", top_k=3)

@pytest.mark.asyncio
async def test_set_with_invalid_data_raises_validation_error(semantrix_instance):
    """Test that set with invalid data raises ValidationError."""
    with pytest.raises(ValidationError, match="Prompt and response must be non-empty strings"):
        await semantrix_instance.set("", "response")

    with pytest.raises(ValidationError, match="Prompt and response must be non-empty strings"):
        await semantrix_instance.set("prompt", None)
