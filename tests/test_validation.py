import pytest
from semantrix.utils.validation import (
    ValidationError,
    validate_prompt,
    validate_response,
    validate_vector,
    validate_metadata,
    validate_operation_id
)

def test_validate_prompt():
    validate_prompt("This is a valid prompt.")
    with pytest.raises(ValidationError):
        validate_prompt("")
    with pytest.raises(ValidationError):
        validate_prompt(123)
    with pytest.raises(ValidationError):
        validate_prompt(None)

def test_validate_response():
    validate_response("This is a valid response.")
    with pytest.raises(ValidationError):
        validate_response("")
    with pytest.raises(ValidationError):
        validate_response(456)
    with pytest.raises(ValidationError):
        validate_response(None)

def test_validate_vector():
    validate_vector([1.0, 2.0, 3.0], 3)
    with pytest.raises(ValidationError):
        validate_vector([1, 2, 3], 3)  # Not floats
    with pytest.raises(ValidationError):
        validate_vector([1.0, 2.0], 3)  # Dimension mismatch
    with pytest.raises(ValidationError):
        validate_vector("not a vector", 3)
    with pytest.raises(ValidationError):
        validate_vector(None, 3)

def test_validate_metadata():
    validate_metadata({"key1": "value1", "key2": 123, "key3": 1.23, "key4": True})
    with pytest.raises(ValidationError):
        validate_metadata({"key1": [1, 2]})  # Invalid value type
    with pytest.raises(ValidationError):
        validate_metadata({123: "value"})  # Invalid key type
    with pytest.raises(ValidationError):
        validate_metadata("not a dict")
    with pytest.raises(ValidationError):
        validate_metadata(None)

def test_validate_operation_id():
    validate_operation_id("valid-op-id")
    validate_operation_id(None)
    with pytest.raises(ValidationError):
        validate_operation_id("")
    with pytest.raises(ValidationError):
        validate_operation_id(12345)
