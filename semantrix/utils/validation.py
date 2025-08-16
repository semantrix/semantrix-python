from typing import List, Optional, Dict, Any

class ValidationError(ValueError):
    """Custom exception for validation errors."""
    pass

def validate_prompt(prompt: str):
    """Ensures prompt is a non-empty string."""
    if not isinstance(prompt, str) or not prompt:
        raise ValidationError("Prompt must be a non-empty string.")

def validate_response(response: str):
    """Ensures response is a non-empty string."""
    if not isinstance(response, str) or not response:
        raise ValidationError("Response must be a non-empty string.")

def validate_vector(vector: List[float], expected_dimension: int):
    """Checks if the vector is a list of floats and matches the expected dimension."""
    if not isinstance(vector, list) or not all(isinstance(i, float) for i in vector):
        raise ValidationError("Vector must be a list of floats.")
    if len(vector) != expected_dimension:
        raise ValidationError(f"Vector dimension mismatch. Expected {expected_dimension}, got {len(vector)}.")

def validate_metadata(metadata: Dict[str, Any]):
    """Checks if metadata keys are strings and values are basic types."""
    if not isinstance(metadata, dict):
        raise ValidationError("Metadata must be a dictionary.")
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise ValidationError("Metadata keys must be strings.")
        if not isinstance(value, (str, int, float, bool)):
            raise ValidationError(f"Metadata value for key '{key}' is not a basic type (str, int, float, bool).")

def validate_operation_id(op_id: Optional[str]):
    """Ensures the operation ID is a valid string if provided."""
    if op_id is not None and (not isinstance(op_id, str) or not op_id):
        raise ValidationError("Operation ID must be a non-empty string if provided.")
