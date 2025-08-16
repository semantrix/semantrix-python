import pytest
from unittest.mock import patch, AsyncMock

from semantrix.core.cache import Semantrix
from semantrix.exceptions import (
    CacheOperationError, 
    VectorOperationError, 
    ValidationError, 
    OperationError
)
from tests.mocks.mock_stores import (
    MockCacheStore,
    MockVectorStore,
    MockEmbedder,
    MockFailingCacheStore,
    MockFailingVectorStore,
    MockFailingEmbedder,
)

# Fixtures using the new mock stores

@pytest.fixture
def mock_cache_store():
    """Provides a clean instance of MockCacheStore for each test."""
    return MockCacheStore()

@pytest.fixture
def mock_vector_store():
    """Provides a clean instance of MockVectorStore for each test."""
    return MockVectorStore(dimension=128)

@pytest.fixture
def mock_embedder():
    """Provides a clean instance of MockEmbedder for each test."""
    return MockEmbedder(dimension=128)

@pytest.fixture
def semantrix_instance(mock_cache_store, mock_vector_store, mock_embedder):
    """Provides a Semantrix instance initialized with mock stores."""
    return Semantrix(
        cache_store=mock_cache_store,
        vector_store=mock_vector_store,
        embedder=mock_embedder,
        enable_wal=False,
        enable_2pc=False  # Disable WAL and 2PC for simpler unit tests
    )

# --- Input Validation Tests ---

@pytest.mark.asyncio
async def test_set_with_invalid_data_raises_validation_error(semantrix_instance):
    """Test that set() with invalid data raises ValidationError."""
    with pytest.raises(ValidationError, match="Prompt must be a non-empty string"):
        await semantrix_instance.set("", "response")
    
    with pytest.raises(ValidationError, match="Response must be a non-empty string"):
        await semantrix_instance.set("prompt", "")
        
    with pytest.raises(ValidationError, match="Response must be a non-empty string"):
        await semantrix_instance.set("prompt", None)

@pytest.mark.asyncio
async def test_get_with_invalid_data_raises_validation_error(semantrix_instance):
    """Test that get() with invalid data raises ValidationError."""
    with pytest.raises(ValidationError, match="Prompt must be a non-empty string"):
        await semantrix_instance.get("")

@pytest.mark.asyncio
async def test_invalid_operation_id_raises_validation_error(semantrix_instance):
    """Test that an invalid operation_id raises ValidationError."""
    with pytest.raises(ValidationError, match="Operation ID must be a string if provided"):
        await semantrix_instance.set("prompt", "response", operation_id=123)
    
    with pytest.raises(ValidationError, match="Operation ID must be a string if provided"):
        await semantrix_instance.get("prompt", operation_id=123)

# --- Error Handling Tests ---

@pytest.mark.asyncio
async def test_set_operation_raises_operation_error_on_failure():
    """Test that a failed set operation raises OperationError."""
    failing_cache_store = MockFailingCacheStore(fail_on_operation="add")
    semantrix = Semantrix(
        cache_store=failing_cache_store,
        vector_store=MockVectorStore(dimension=128),
        embedder=MockEmbedder(dimension=128),
        enable_2pc=False
    )
    
    with pytest.raises(OperationError, match="Failed to set prompt"):
        await semantrix.set("test_prompt", "test_response")

@pytest.mark.asyncio
async def test_get_operation_raises_cache_error_on_cache_failure():
    """Test that a failed get operation raises CacheOperationError."""
    failing_cache_store = MockFailingCacheStore(fail_on_operation="get_exact")
    semantrix = Semantrix(
        cache_store=failing_cache_store,
        vector_store=MockVectorStore(dimension=128),
        embedder=MockEmbedder(dimension=128)
    )
    
    # Mock _get_semantic to avoid vector store interaction
    with patch.object(semantrix, '_get_semantic', return_value=None):
        with pytest.raises(CacheOperationError):
            # The retry decorator will re-raise the final error
            await semantrix.get("test_prompt")

@pytest.mark.asyncio
async def test_semantic_search_raises_vector_error_on_failure():
    """Test that a failed semantic search raises VectorOperationError."""
    failing_vector_store = MockFailingVectorStore(dimension=128, fail_on_operation="search")
    semantrix = Semantrix(
        cache_store=MockCacheStore(),
        vector_store=failing_vector_store,
        embedder=MockEmbedder(dimension=128)
    )
    
    with pytest.raises(VectorOperationError, match="Semantic search failed"):
        await semantrix.get("query")

# --- Retry Logic Tests ---

@pytest.mark.asyncio
async def test_retry_logic_succeeds_on_transient_failure(mocker):
    """Test that the @retry decorator correctly retries and succeeds."""
    mock_cache = MockCacheStore()
    semantrix = Semantrix(
        cache_store=mock_cache,
        vector_store=MockVectorStore(dimension=128),
        embedder=MockEmbedder(dimension=128)
    )

    # Mock the get_exact method to fail twice then succeed
    side_effects = [
        CacheOperationError("Transient failure 1"),
        CacheOperationError("Transient failure 2"),
        "successful response"
    ]
    mock_cache.get_exact = AsyncMock(side_effect=side_effects)
    
    # Mock _get_semantic to isolate the test to the exact match path
    mocker.patch.object(semantrix, '_get_semantic', return_value=None)

    # The call should succeed after retries
    result = await semantrix.get("test_prompt")
    
    assert result == "successful response"
    assert mock_cache.get_exact.call_count == 3

@pytest.mark.asyncio
async def test_retry_logic_fails_on_persistent_failure(mocker):
    """Test that the @retry decorator fails after max retries."""
    mock_cache = MockCacheStore()
    semantrix = Semantrix(
        cache_store=mock_cache,
        vector_store=MockVectorStore(dimension=128),
        embedder=MockEmbedder(dimension=128)
    )

    # Mock the get_exact method to always fail
    mock_cache.get_exact = AsyncMock(side_effect=CacheOperationError("Persistent failure"))
    
    # Mock _get_semantic to isolate the test
    mocker.patch.object(semantrix, '_get_semantic', return_value=None)

    # The call should raise the exception after exhausting all retries
    with pytest.raises(CacheOperationError, match="Persistent failure"):
        await semantrix.get("test_prompt")
    
    # Default is 3 retries, so 1 initial call + 3 retries = 4 calls
    assert mock_cache.get_exact.call_count == 4