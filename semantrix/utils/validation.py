from typing import List, Optional, Dict, Any, Union, TypeVar, Generic
import re
from semantrix.exceptions import ValidationError

T = TypeVar('T')

class ValidationError(ValueError):
    """Custom exception for validation errors."""
    pass

def validate_prompt(prompt: str):
    """Ensures prompt is a non-empty string."""
    if not isinstance(prompt, str):
        raise ValidationError("Prompt must be a string.")
    
    if not prompt.strip():
        raise ValidationError("Prompt must be a non-empty string.")
    
    # Check for reasonable length
    if len(prompt) > 10000:  # 10KB limit
        raise ValidationError("Prompt is too long. Maximum length is 10,000 characters.")

def validate_response(response: str):
    """Ensures response is a non-empty string."""
    if not isinstance(response, str):
        raise ValidationError("Response must be a string.")
    
    if not response.strip():
        raise ValidationError("Response must be a non-empty string.")
    
    # Check for reasonable length
    if len(response) > 50000:  # 50KB limit
        raise ValidationError("Response is too long. Maximum length is 50,000 characters.")

def validate_vector(vector: List[float], expected_dimension: int):
    """Checks if the vector is a list of floats and matches the expected dimension."""
    if not isinstance(vector, list):
        raise ValidationError("Vector must be a list.")
    
    if not vector:
        raise ValidationError("Vector cannot be empty.")
    
    if not all(isinstance(i, (int, float)) for i in vector):
        raise ValidationError("Vector must contain only numeric values.")
    
    if len(vector) != expected_dimension:
        raise ValidationError(f"Vector dimension mismatch. Expected {expected_dimension}, got {len(vector)}.")
    
    # Check for NaN or infinite values
    if any(not (float('-inf') < x < float('inf')) for x in vector):
        raise ValidationError("Vector contains NaN or infinite values.")

def validate_metadata(metadata: Dict[str, Any]):
    """Checks if metadata keys are strings and values are basic types."""
    if not isinstance(metadata, dict):
        raise ValidationError("Metadata must be a dictionary.")
    
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise ValidationError("Metadata keys must be strings.")
        
        if not key:
            raise ValidationError("Metadata keys cannot be empty strings.")
        
        if not isinstance(value, (str, int, float, bool, type(None))):
            raise ValidationError(f"Metadata value for key '{key}' is not a basic type (str, int, float, bool, None).")
        
        # Check string length limits
        if isinstance(value, str) and len(value) > 1000:
            raise ValidationError(f"Metadata value for key '{key}' is too long. Maximum length is 1,000 characters.")

def validate_operation_id(op_id: Optional[str]):
    """Ensures the operation ID is a valid string if provided."""
    if op_id is not None:
        if not isinstance(op_id, str):
            raise ValidationError("Operation ID must be a string.")
        
        if not op_id:
            raise ValidationError("Operation ID cannot be an empty string.")
        
        # Check for valid characters (alphanumeric, hyphens, underscores)
        if not re.match(r'^[a-zA-Z0-9_-]+$', op_id):
            raise ValidationError("Operation ID can only contain alphanumeric characters, hyphens, and underscores.")
        
        if len(op_id) > 100:
            raise ValidationError("Operation ID is too long. Maximum length is 100 characters.")

def validate_cache_key(key: str):
    """Validates cache key format and length."""
    if not isinstance(key, str):
        raise ValidationError("Cache key must be a string.")
    
    if not key:
        raise ValidationError("Cache key cannot be empty.")
    
    if len(key) > 250:
        raise ValidationError("Cache key is too long. Maximum length is 250 characters.")
    
    # Check for valid characters
    if not re.match(r'^[a-zA-Z0-9_-]+$', key):
        raise ValidationError("Cache key can only contain alphanumeric characters, hyphens, and underscores.")

def validate_ttl(ttl: Optional[int]):
    """Validates TTL (Time To Live) value."""
    if ttl is not None:
        if not isinstance(ttl, int):
            raise ValidationError("TTL must be an integer.")
        
        if ttl < 0:
            raise ValidationError("TTL cannot be negative.")
        
        if ttl > 31536000:  # 1 year in seconds
            raise ValidationError("TTL is too large. Maximum value is 31,536,000 seconds (1 year).")

def validate_batch_size(batch_size: int, max_size: int = 1000):
    """Validates batch operation size."""
    if not isinstance(batch_size, int):
        raise ValidationError("Batch size must be an integer.")
    
    if batch_size <= 0:
        raise ValidationError("Batch size must be positive.")
    
    if batch_size > max_size:
        raise ValidationError(f"Batch size is too large. Maximum value is {max_size}.")

def validate_embedding_model(model_name: str):
    """Validates embedding model name."""
    if not isinstance(model_name, str):
        raise ValidationError("Model name must be a string.")
    
    if not model_name:
        raise ValidationError("Model name cannot be empty.")
    
    if len(model_name) > 100:
        raise ValidationError("Model name is too long. Maximum length is 100 characters.")
    
    # Check for valid characters
    if not re.match(r'^[a-zA-Z0-9_\-/]+$', model_name):
        raise ValidationError("Model name can only contain alphanumeric characters, hyphens, underscores, and forward slashes.")

def validate_similarity_threshold(threshold: float):
    """Validates similarity threshold for vector search."""
    if not isinstance(threshold, (int, float)):
        raise ValidationError("Similarity threshold must be a number.")
    
    if not (0.0 <= threshold <= 1.0):
        raise ValidationError("Similarity threshold must be between 0.0 and 1.0.")

def validate_top_k(top_k: int):
    """Validates top-k parameter for vector search."""
    if not isinstance(top_k, int):
        raise ValidationError("Top-k must be an integer.")
    
    if top_k <= 0:
        raise ValidationError("Top-k must be positive.")
    
    if top_k > 1000:
        raise ValidationError("Top-k is too large. Maximum value is 1000.")

class Validator(Generic[T]):
    """Generic validator class for type-safe validation."""
    
    def __init__(self, validation_func: callable):
        self.validation_func = validation_func
    
    def __call__(self, value: T) -> T:
        """Validate the value and return it if valid."""
        self.validation_func(value)
        return value
    
    def __or__(self, other: 'Validator[T]') -> 'Validator[T]':
        """Combine validators with OR logic."""
        def combined_validator(value: T) -> T:
            try:
                return self(value)
            except ValidationError:
                return other(value)
        return Validator(combined_validator)

# Pre-defined validators
non_empty_string = Validator[str](lambda s: validate_prompt(s))
positive_integer = Validator[int](lambda i: None if isinstance(i, int) and i > 0 else _raise_validation_error("Must be a positive integer"))
valid_metadata = Validator[Dict[str, Any]](validate_metadata)
valid_vector = lambda dim: Validator[List[float]](lambda v: validate_vector(v, dim))

def _raise_validation_error(message: str):
    """Helper function to raise ValidationError."""
    raise ValidationError(message)
