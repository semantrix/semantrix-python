class SemantrixError(Exception):
    """Base class for all Semantrix exceptions."""
    pass

class ConfigurationError(SemantrixError):
    """Raised when there is an error in the configuration."""
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception
        self.message = message

    def __str__(self) -> str:
        if self.original_exception:
            return f"{self.message} (Original: {str(self.original_exception)})"
        return self.message

class CacheStoreError(SemantrixError):
    """Base class for cache store related errors."""
    pass

class CacheInitializationError(CacheStoreError):
    """Raised when there is an error initializing the cache store."""
    pass

class CacheOperationError(CacheStoreError):
    """Raised when a cache operation (get/set/delete) fails."""
    pass

class VectorStoreError(SemantrixError):
    """Base class for vector store related errors."""
    pass

class VectorStoreInitializationError(VectorStoreError):
    """Raised when there is an error initializing the vector store."""
    pass

class VectorOperationError(VectorStoreError):
    """Raised when a vector operation (search/insert/delete) fails."""
    pass

class ValidationError(SemantrixError, ValueError):
    """Raised when input validation fails."""
    pass

class OperationError(SemantrixError):
    """Raised when a general operation fails."""
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception
        self.message = message

    def __str__(self) -> str:
        if self.original_exception:
            return f"{self.message} (Original: {str(self.original_exception)})"
        return self.message

class RetryError(SemantrixError):
    """Raised when the maximum number of retries is exceeded."""
    def __init__(self, message: str, original_exception: Exception = None):
        super().__init__(message)
        self.original_exception = original_exception
        self.message = message

    def __str__(self) -> str:
        if self.original_exception:
            return f"{self.message} (Original: {str(self.original_exception)})"
        return self.message
