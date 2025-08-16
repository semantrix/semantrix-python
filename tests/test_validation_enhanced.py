"""
Enhanced tests for validation functionality.
"""
import unittest
import math
from typing import List, Dict, Any

from semantrix.utils.validation import (
    validate_prompt,
    validate_response,
    validate_vector,
    validate_metadata,
    validate_operation_id,
    validate_cache_key,
    validate_ttl,
    validate_batch_size,
    validate_embedding_model,
    validate_similarity_threshold,
    validate_top_k,
    ValidationError,
    Validator,
    non_empty_string,
    positive_integer,
    valid_metadata,
    valid_vector,
)


class TestPromptValidation(unittest.TestCase):
    """Test cases for prompt validation."""

    def test_valid_prompt(self):
        """Test valid prompt passes validation."""
        valid_prompts = [
            "Hello world",
            "A" * 1000,
            "Test prompt with special chars: !@#$%^&*()",
        ]
        
        for prompt in valid_prompts:
            # Should not raise any exception
            validate_prompt(prompt)

    def test_invalid_prompt_types(self):
        """Test that non-string prompts are rejected."""
        invalid_prompts = [
            None,
            123,
            [],
            {},
            True,
        ]
        
        for prompt in invalid_prompts:
            with self.assertRaises(ValidationError) as cm:
                validate_prompt(prompt)
            self.assertIn("must be a non-empty string", str(cm.exception))

    def test_empty_prompt(self):
        """Test that empty prompts are rejected."""
        empty_prompts = ["", "   ", "\n", "\t"]
        
        for prompt in empty_prompts:
            with self.assertRaises(ValidationError) as cm:
                validate_prompt(prompt)
            self.assertIn("must be a non-empty string", str(cm.exception))

    def test_prompt_too_long(self):
        """Test that overly long prompts are rejected."""
        long_prompt = "A" * 10001
        
        with self.assertRaises(ValidationError) as cm:
            validate_prompt(long_prompt)
        self.assertIn("too long", str(cm.exception))
        self.assertIn("10,000 characters", str(cm.exception))


class TestResponseValidation(unittest.TestCase):
    """Test cases for response validation."""

    def test_valid_response(self):
        """Test valid response passes validation."""
        valid_responses = [
            "Hello world",
            "A" * 1000,
            "Test response with special chars: !@#$%^&*()",
        ]
        
        for response in valid_responses:
            # Should not raise any exception
            validate_response(response)

    def test_invalid_response_types(self):
        """Test that non-string responses are rejected."""
        invalid_responses = [
            None,
            123,
            [],
            {},
            True,
        ]
        
        for response in invalid_responses:
            with self.assertRaises(ValidationError) as cm:
                validate_response(response)
            self.assertIn("must be a non-empty string", str(cm.exception))

    def test_empty_response(self):
        """Test that empty responses are rejected."""
        empty_responses = ["", "   ", "\n", "\t"]
        
        for response in empty_responses:
            with self.assertRaises(ValidationError) as cm:
                validate_response(response)
            self.assertIn("must be a non-empty string", str(cm.exception))

    def test_response_too_long(self):
        """Test that overly long responses are rejected."""
        long_response = "A" * 50001
        
        with self.assertRaises(ValidationError) as cm:
            validate_response(long_response)
        self.assertIn("too long", str(cm.exception))
        self.assertIn("50,000 characters", str(cm.exception))


class TestVectorValidation(unittest.TestCase):
    """Test cases for vector validation."""

    def test_valid_vector(self):
        """Test valid vector passes validation."""
        valid_vectors = [
            [1.0, 2.0, 3.0],
            [0.0, -1.0, 0.5],
            [1, 2, 3],  # Integers should be accepted
        ]
        
        for vector in valid_vectors:
            # Should not raise any exception
            validate_vector(vector, 3)

    def test_invalid_vector_types(self):
        """Test that non-list vectors are rejected."""
        invalid_vectors = [
            None,
            "string",
            123,
            {},
            True,
        ]
        
        for vector in invalid_vectors:
            with self.assertRaises(ValidationError) as cm:
                validate_vector(vector, 3)
            self.assertIn("must be a list", str(cm.exception))

    def test_empty_vector(self):
        """Test that empty vectors are rejected."""
        with self.assertRaises(ValidationError) as cm:
            validate_vector([], 3)
        self.assertIn("cannot be empty", str(cm.exception))

    def test_vector_with_non_numeric_values(self):
        """Test that vectors with non-numeric values are rejected."""
        invalid_vectors = [
            ["1", 2, 3],
            [1, "2", 3],
            [1, 2, "3"],
            [1, None, 3],
            [1, [], 3],
        ]
        
        for vector in invalid_vectors:
            with self.assertRaises(ValidationError) as cm:
                validate_vector(vector, 3)
            self.assertIn("numeric values", str(cm.exception))

    def test_vector_dimension_mismatch(self):
        """Test that vectors with wrong dimensions are rejected."""
        vector = [1.0, 2.0, 3.0]
        
        with self.assertRaises(ValidationError) as cm:
            validate_vector(vector, 2)
        self.assertIn("dimension mismatch", str(cm.exception))
        self.assertIn("Expected 2, got 3", str(cm.exception))

    def test_vector_with_nan_values(self):
        """Test that vectors with NaN values are rejected."""
        vector_with_nan = [1.0, float('nan'), 3.0]
        
        with self.assertRaises(ValidationError) as cm:
            validate_vector(vector_with_nan, 3)
        self.assertIn("NaN or infinite values", str(cm.exception))

    def test_vector_with_infinite_values(self):
        """Test that vectors with infinite values are rejected."""
        vector_with_inf = [1.0, float('inf'), 3.0]
        
        with self.assertRaises(ValidationError) as cm:
            validate_vector(vector_with_inf, 3)
        self.assertIn("NaN or infinite values", str(cm.exception))


class TestMetadataValidation(unittest.TestCase):
    """Test cases for metadata validation."""

    def test_valid_metadata(self):
        """Test valid metadata passes validation."""
        valid_metadata = [
            {},
            {"key": "value"},
            {"key1": "value1", "key2": 123, "key3": 45.67, "key4": True, "key5": None},
        ]
        
        for metadata in valid_metadata:
            # Should not raise any exception
            validate_metadata(metadata)

    def test_invalid_metadata_types(self):
        """Test that non-dict metadata is rejected."""
        invalid_metadata = [
            None,
            "string",
            123,
            [],
            True,
        ]
        
        for metadata in invalid_metadata:
            with self.assertRaises(ValidationError) as cm:
                validate_metadata(metadata)
            self.assertIn("must be a dictionary", str(cm.exception))

    def test_metadata_with_invalid_keys(self):
        """Test that metadata with invalid keys is rejected."""
        invalid_metadata = [
            {123: "value"},
            {None: "value"},
            {(): "value"},  # Use tuple instead of list for unhashable type test
        ]
        
        for metadata in invalid_metadata:
            with self.assertRaises(ValidationError) as cm:
                validate_metadata(metadata)
            self.assertIn("keys must be strings", str(cm.exception))

    def test_metadata_with_empty_keys(self):
        """Test that metadata with empty keys is rejected."""
        with self.assertRaises(ValidationError) as cm:
            validate_metadata({"": "value"})
        self.assertIn("cannot be empty strings", str(cm.exception))

    def test_metadata_with_invalid_values(self):
        """Test that metadata with invalid values is rejected."""
        invalid_metadata = [
            {"key": []},
            {"key": {}},
            {"key": lambda x: x},
        ]
        
        for metadata in invalid_metadata:
            with self.assertRaises(ValidationError) as cm:
                validate_metadata(metadata)
            self.assertIn("not a basic type", str(cm.exception))

    def test_metadata_with_long_string_values(self):
        """Test that metadata with overly long string values is rejected."""
        long_string = "A" * 1001
        metadata = {"key": long_string}
        
        with self.assertRaises(ValidationError) as cm:
            validate_metadata(metadata)
        self.assertIn("too long", str(cm.exception))
        self.assertIn("1,000 characters", str(cm.exception))


class TestOperationIdValidation(unittest.TestCase):
    """Test cases for operation ID validation."""

    def test_valid_operation_id(self):
        """Test valid operation IDs pass validation."""
        valid_ids = [
            "op_123",
            "operation-456",
            "test_operation_789",
            "A" * 100,  # Maximum length
        ]
        
        for op_id in valid_ids:
            # Should not raise any exception
            validate_operation_id(op_id)

    def test_none_operation_id(self):
        """Test that None operation ID is accepted."""
        # Should not raise any exception
        validate_operation_id(None)

    def test_invalid_operation_id_types(self):
        """Test that non-string operation IDs are rejected."""
        invalid_ids = [
            123,
            [],
            {},
            True,
        ]
        
        for op_id in invalid_ids:
            with self.assertRaises(ValidationError) as cm:
                validate_operation_id(op_id)
            self.assertIn("must be a string", str(cm.exception))

    def test_empty_operation_id(self):
        """Test that empty operation IDs are rejected."""
        with self.assertRaises(ValidationError) as cm:
            validate_operation_id("")
        self.assertIn("cannot be an empty string", str(cm.exception))

    def test_operation_id_with_invalid_characters(self):
        """Test that operation IDs with invalid characters are rejected."""
        invalid_ids = [
            "op@123",
            "operation#456",
            "test operation 789",
            "op.123",
        ]
        
        for op_id in invalid_ids:
            with self.assertRaises(ValidationError) as cm:
                validate_operation_id(op_id)
            self.assertIn("alphanumeric characters, hyphens, and underscores", str(cm.exception))

    def test_operation_id_too_long(self):
        """Test that overly long operation IDs are rejected."""
        long_id = "A" * 101
        
        with self.assertRaises(ValidationError) as cm:
            validate_operation_id(long_id)
        self.assertIn("too long", str(cm.exception))
        self.assertIn("100 characters", str(cm.exception))


class TestCacheKeyValidation(unittest.TestCase):
    """Test cases for cache key validation."""

    def test_valid_cache_key(self):
        """Test valid cache keys pass validation."""
        valid_keys = [
            "key_123",
            "cache-key-456",
            "test_cache_789",
            "A" * 250,  # Maximum length
        ]
        
        for key in valid_keys:
            # Should not raise any exception
            validate_cache_key(key)

    def test_invalid_cache_key_types(self):
        """Test that non-string cache keys are rejected."""
        invalid_keys = [
            None,
            123,
            [],
            {},
            True,
        ]
        
        for key in invalid_keys:
            with self.assertRaises(ValidationError) as cm:
                validate_cache_key(key)
            self.assertIn("must be a string", str(cm.exception))

    def test_empty_cache_key(self):
        """Test that empty cache keys are rejected."""
        with self.assertRaises(ValidationError) as cm:
            validate_cache_key("")
        self.assertIn("cannot be empty", str(cm.exception))

    def test_cache_key_with_invalid_characters(self):
        """Test that cache keys with invalid characters are rejected."""
        invalid_keys = [
            "key@123",
            "cache#key",
            "test cache",
            "key.123",
        ]
        
        for key in invalid_keys:
            with self.assertRaises(ValidationError) as cm:
                validate_cache_key(key)
            self.assertIn("alphanumeric characters, hyphens, and underscores", str(cm.exception))

    def test_cache_key_too_long(self):
        """Test that overly long cache keys are rejected."""
        long_key = "A" * 251
        
        with self.assertRaises(ValidationError) as cm:
            validate_cache_key(long_key)
        self.assertIn("too long", str(cm.exception))
        self.assertIn("250 characters", str(cm.exception))


class TestTtlValidation(unittest.TestCase):
    """Test cases for TTL validation."""

    def test_valid_ttl(self):
        """Test valid TTL values pass validation."""
        valid_ttls = [
            None,
            0,
            60,
            3600,
            86400,
            31536000,  # 1 year
        ]
        
        for ttl in valid_ttls:
            # Should not raise any exception
            validate_ttl(ttl)

    def test_invalid_ttl_types(self):
        """Test that non-integer TTL values are rejected."""
        invalid_ttls = [
            "60",
            60.0,
            [],
            {},
            True,
        ]
        
        for ttl in invalid_ttls:
            with self.assertRaises(ValidationError) as cm:
                validate_ttl(ttl)
            self.assertIn("must be an integer", str(cm.exception))

    def test_negative_ttl(self):
        """Test that negative TTL values are rejected."""
        with self.assertRaises(ValidationError) as cm:
            validate_ttl(-1)
        self.assertIn("cannot be negative", str(cm.exception))

    def test_ttl_too_large(self):
        """Test that overly large TTL values are rejected."""
        with self.assertRaises(ValidationError) as cm:
            validate_ttl(31536001)
        self.assertIn("too large", str(cm.exception))
        self.assertIn("31,536,000 seconds", str(cm.exception))


class TestBatchSizeValidation(unittest.TestCase):
    """Test cases for batch size validation."""

    def test_valid_batch_size(self):
        """Test valid batch sizes pass validation."""
        valid_sizes = [
            1,
            10,
            100,
            1000,  # Default maximum
        ]
        
        for size in valid_sizes:
            # Should not raise any exception
            validate_batch_size(size)

    def test_invalid_batch_size_types(self):
        """Test that non-integer batch sizes are rejected."""
        invalid_sizes = [
            "10",
            10.0,
            [],
            {},
            True,
        ]
        
        for size in invalid_sizes:
            with self.assertRaises(ValidationError) as cm:
                validate_batch_size(size)
            self.assertIn("must be an integer", str(cm.exception))

    def test_zero_batch_size(self):
        """Test that zero batch size is rejected."""
        with self.assertRaises(ValidationError) as cm:
            validate_batch_size(0)
        self.assertIn("must be positive", str(cm.exception))

    def test_negative_batch_size(self):
        """Test that negative batch size is rejected."""
        with self.assertRaises(ValidationError) as cm:
            validate_batch_size(-1)
        self.assertIn("must be positive", str(cm.exception))

    def test_batch_size_too_large(self):
        """Test that overly large batch size is rejected."""
        with self.assertRaises(ValidationError) as cm:
            validate_batch_size(1001)
        self.assertIn("too large", str(cm.exception))
        self.assertIn("1000", str(cm.exception))

    def test_custom_max_batch_size(self):
        """Test custom maximum batch size."""
        # Should work with custom max
        validate_batch_size(500, max_size=500)
        
        # Should fail with custom max
        with self.assertRaises(ValidationError) as cm:
            validate_batch_size(501, max_size=500)
        self.assertIn("500", str(cm.exception))


class TestEmbeddingModelValidation(unittest.TestCase):
    """Test cases for embedding model validation."""

    def test_valid_model_names(self):
        """Test valid model names pass validation."""
        valid_names = [
            "text-embedding-ada-002",
            "sentence-transformers/all-MiniLM-L6-v2",
            "model_123",
            "A" * 100,  # Maximum length
        ]
        
        for name in valid_names:
            # Should not raise any exception
            validate_embedding_model(name)

    def test_invalid_model_name_types(self):
        """Test that non-string model names are rejected."""
        invalid_names = [
            None,
            123,
            [],
            {},
            True,
        ]
        
        for name in invalid_names:
            with self.assertRaises(ValidationError) as cm:
                validate_embedding_model(name)
            self.assertIn("must be a string", str(cm.exception))

    def test_empty_model_name(self):
        """Test that empty model names are rejected."""
        with self.assertRaises(ValidationError) as cm:
            validate_embedding_model("")
        self.assertIn("cannot be empty", str(cm.exception))

    def test_model_name_with_invalid_characters(self):
        """Test that model names with invalid characters are rejected."""
        invalid_names = [
            "model@123",
            "embedding#model",
            "text embedding model",
            "model.123",
        ]
        
        for name in invalid_names:
            with self.assertRaises(ValidationError) as cm:
                validate_embedding_model(name)
            self.assertIn("alphanumeric characters, hyphens, underscores, and forward slashes", str(cm.exception))

    def test_model_name_too_long(self):
        """Test that overly long model names are rejected."""
        long_name = "A" * 101
        
        with self.assertRaises(ValidationError) as cm:
            validate_embedding_model(long_name)
        self.assertIn("too long", str(cm.exception))
        self.assertIn("100 characters", str(cm.exception))


class TestSimilarityThresholdValidation(unittest.TestCase):
    """Test cases for similarity threshold validation."""

    def test_valid_thresholds(self):
        """Test valid similarity thresholds pass validation."""
        valid_thresholds = [
            0.0,
            0.5,
            1.0,
            0,
            1,
        ]
        
        for threshold in valid_thresholds:
            # Should not raise any exception
            validate_similarity_threshold(threshold)

    def test_invalid_threshold_types(self):
        """Test that non-numeric thresholds are rejected."""
        invalid_thresholds = [
            "0.5",
            [],
            {},
            True,
            None,
        ]
        
        for threshold in invalid_thresholds:
            with self.assertRaises(ValidationError) as cm:
                validate_similarity_threshold(threshold)
            self.assertIn("must be a number", str(cm.exception))

    def test_threshold_below_zero(self):
        """Test that thresholds below zero are rejected."""
        with self.assertRaises(ValidationError) as cm:
            validate_similarity_threshold(-0.1)
        self.assertIn("between 0.0 and 1.0", str(cm.exception))

    def test_threshold_above_one(self):
        """Test that thresholds above one are rejected."""
        with self.assertRaises(ValidationError) as cm:
            validate_similarity_threshold(1.1)
        self.assertIn("between 0.0 and 1.0", str(cm.exception))


class TestTopKValidation(unittest.TestCase):
    """Test cases for top-k validation."""

    def test_valid_top_k(self):
        """Test valid top-k values pass validation."""
        valid_top_ks = [
            1,
            10,
            100,
            1000,  # Maximum
        ]
        
        for top_k in valid_top_ks:
            # Should not raise any exception
            validate_top_k(top_k)

    def test_invalid_top_k_types(self):
        """Test that non-integer top-k values are rejected."""
        invalid_top_ks = [
            "10",
            10.0,
            [],
            {},
            True,
        ]
        
        for top_k in invalid_top_ks:
            with self.assertRaises(ValidationError) as cm:
                validate_top_k(top_k)
            self.assertIn("must be an integer", str(cm.exception))

    def test_zero_top_k(self):
        """Test that zero top-k is rejected."""
        with self.assertRaises(ValidationError) as cm:
            validate_top_k(0)
        self.assertIn("must be positive", str(cm.exception))

    def test_negative_top_k(self):
        """Test that negative top-k is rejected."""
        with self.assertRaises(ValidationError) as cm:
            validate_top_k(-1)
        self.assertIn("must be positive", str(cm.exception))

    def test_top_k_too_large(self):
        """Test that overly large top-k is rejected."""
        with self.assertRaises(ValidationError) as cm:
            validate_top_k(1001)
        self.assertIn("too large", str(cm.exception))
        self.assertIn("1000", str(cm.exception))


class TestValidatorClass(unittest.TestCase):
    """Test cases for Validator class."""

    def test_validator_basic_usage(self):
        """Test basic validator usage."""
        validator = Validator[int](lambda x: None if isinstance(x, int) and x > 0 else ValidationError("Must be positive integer"))
        
        # Valid value
        result = validator(5)
        self.assertEqual(result, 5)
        
        # Invalid value
        with self.assertRaises(ValidationError):
            validator(-1)

    def test_validator_or_combination(self):
        """Test validator OR combination."""
        validator1 = Validator[int](lambda x: None if isinstance(x, int) and x > 0 else ValidationError("Must be positive"))
        validator2 = Validator[int](lambda x: None if isinstance(x, int) and x < 0 else ValidationError("Must be negative"))
        
        combined = validator1 | validator2
        
        # Should accept positive numbers
        result = combined(5)
        self.assertEqual(result, 5)
        
        # Should accept negative numbers
        result = combined(-5)
        self.assertEqual(result, -5)
        
        # Should reject zero
        with self.assertRaises(ValidationError):
            combined(0)

    def test_predefined_validators(self):
        """Test predefined validators."""
        # Test non_empty_string
        result = non_empty_string("test")
        self.assertEqual(result, "test")
        
        with self.assertRaises(ValidationError):
            non_empty_string("")
        
        # Test positive_integer
        result = positive_integer(5)
        self.assertEqual(result, 5)
        
        with self.assertRaises(ValidationError):
            positive_integer(-1)
        
        # Test valid_metadata
        metadata = {"key": "value"}
        result = valid_metadata(metadata)
        self.assertEqual(result, metadata)
        
        with self.assertRaises(ValidationError):
            valid_metadata({"key": []})
        
        # Test valid_vector
        vector_validator = valid_vector(3)
        vector = [1.0, 2.0, 3.0]
        result = vector_validator(vector)
        self.assertEqual(result, vector)
        
        with self.assertRaises(ValidationError):
            vector_validator([1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
