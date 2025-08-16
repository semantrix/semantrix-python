import pytest
from semantrix.exceptions import (
    SemantrixError,
    ConfigurationError,
    CacheStoreError,
    CacheInitializationError,
    CacheOperationError,
    VectorStoreError,
    VectorStoreInitializationError,
    VectorOperationError,
    ValidationError,
    OperationError,
    RetryError
)

def test_semantrix_error_inheritance():
    """Test that all exceptions inherit from SemantrixError."""
    assert issubclass(ConfigurationError, SemantrixError)
    assert issubclass(CacheStoreError, SemantrixError)
    assert issubclass(VectorStoreError, SemantrixError)
    assert issubclass(ValidationError, SemantrixError)
    assert issubclass(OperationError, SemantrixError)
    assert issubclass(RetryError, SemantrixError)

def test_configuration_error_with_message():
    """Test ConfigurationError with a custom message."""
    msg = "Invalid configuration"
    error = ConfigurationError(msg)
    assert str(error) == msg

def test_configuration_error_with_original_exception():
    """Test ConfigurationError with an original exception."""
    original = ValueError("Invalid value")
    error = ConfigurationError("Configuration failed", original)
    assert "Original:" in str(error)
    assert "Invalid value" in str(error)

def test_validation_error_inheritance():
    """Test that ValidationError inherits from both SemantrixError and ValueError."""
    assert issubclass(ValidationError, ValueError)
    assert issubclass(ValidationError, SemantrixError)

def test_cache_error_hierarchy():
    """Test the cache error hierarchy."""
    assert issubclass(CacheInitializationError, CacheStoreError)
    assert issubclass(CacheOperationError, CacheStoreError)

def test_vector_error_hierarchy():
    """Test the vector store error hierarchy."""
    assert issubclass(VectorStoreInitializationError, VectorStoreError)
    assert issubclass(VectorOperationError, VectorStoreError)

def test_operation_error_usage():
    """Test OperationError usage."""
    with pytest.raises(OperationError):
        raise OperationError("Operation failed")

def test_retry_error():
    """Test RetryError usage."""
    with pytest.raises(RetryError):
        raise RetryError("Max retries exceeded")
