"""
Tests for the Semantrix embedding module.
"""

import pytest
import numpy as np
import asyncio

from semantrix.embedding import BaseEmbedder, get_embedder, SentenceTransformerEmbedder, OnnxEmbedder

class TestBaseEmbedder:
    """Tests for the BaseEmbedder class."""
    
    def test_abstract_methods(self):
        """Test that BaseEmbedder cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseEmbedder()
    
    def test_async_interface(self):
        """Test the async interface of BaseEmbedder."""
        class MockEmbedder(BaseEmbedder):
            def encode(self, text: str) -> np.ndarray:
                return np.array([1.0, 2.0, 3.0])
                
            def get_dimension(self) -> int:
                return 3
        
        embedder = MockEmbedder()
        
        # Test sync methods
        result = embedder.encode("test")
        assert isinstance(result, np.ndarray)
        assert result.shape == (3,)
        
        # Test batch sync methods
        batch_result = embedder.batch_encode(["test1", "test2"])
        assert len(batch_result) == 2
        assert all(isinstance(x, np.ndarray) for x in batch_result)
        
        # Test async methods
        async def test_async():
            aresult = await embedder.aencode("test")
            assert isinstance(aresult, np.ndarray)
            
            abatch_result = await embedder.abatch_encode(["test1", "test2"])
            assert len(abatch_result) == 2
            assert all(isinstance(x, np.ndarray) for x in abatch_result)
            
            # Test context manager
            async with embedder as e:
                assert e is embedder
        
        asyncio.run(test_async())

class TestGetEmbedder:
    """Tests for the get_embedder factory function."""
    
    def test_get_sentence_transformer_embedder(self):
        """Test getting a SentenceTransformerEmbedder instance."""
        try:
            embedder = get_embedder("sentence-transformers", model_name="all-MiniLM-L6-v2")
            assert isinstance(embedder, SentenceTransformerEmbedder)
            assert embedder.get_dimension() > 0
        except ImportError:
            pytest.skip("sentence-transformers not installed")
    
    def test_get_onnx_embedder(self):
        """Test getting an OnnxEmbedder instance."""
        try:
            # This will fail if the model files don't exist, but we're just testing the factory
            with pytest.raises(ValueError):  # Should fail due to missing model files
                get_embedder("onnx", model_path="nonexistent.onnx")
        except ImportError:
            pytest.skip("onnxruntime not installed")
    
    def test_invalid_embedder(self):
        """Test that an invalid embedder name raises an error."""
        with pytest.raises(ValueError, match="Unknown embedder: invalid"):
            get_embedder("invalid")

class TestSentenceTransformerEmbedder:
    """Tests for the SentenceTransformerEmbedder class."""
    
    @pytest.fixture
    def embedder(self):
        try:
            return SentenceTransformerEmbedder(model_name="all-MiniLM-L6-v2")
        except ImportError:
            pytest.skip("sentence-transformers not installed")
    
    def test_encode(self, embedder):
        """Test synchronous encoding."""
        embedding = embedder.encode("test sentence")
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == embedder.get_dimension()
    
    def test_batch_encode(self, embedder):
        """Test synchronous batch encoding."""
        embeddings = embedder.batch_encode(["test1", "test2"])
        assert len(embeddings) == 2
        assert all(isinstance(e, np.ndarray) for e in embeddings)
    
    def test_async_encode(self, embedder):
        """Test asynchronous encoding."""
        async def test():
            embedding = await embedder.aencode("test sentence")
            assert isinstance(embedding, np.ndarray)
            assert len(embedding) == embedder.get_dimension()
        
        asyncio.run(test())
    
    def test_async_batch_encode(self, embedder):
        """Test asynchronous batch encoding."""
        async def test():
            embeddings = await embedder.abatch_encode(["test1", "test2"])
            assert len(embeddings) == 2
            assert all(isinstance(e, np.ndarray) for e in embeddings)
        
        asyncio.run(test())

class TestOnnxEmbedder:
    """Tests for the OnnxEmbedder class."""
    
    # Note: This is a basic test that doesn't require actual model files
    # Full testing would require proper ONNX model files
    
    def test_init_requires_model_path(self):
        """Test that model_path is required."""
        try:
            with pytest.raises(ValueError):
                OnnxEmbedder(model_path=None, tokenizer_name=None)
        except ImportError:
            pytest.skip("onnxruntime not installed")
    
    # Additional tests for OnnxEmbedder would require actual model files
    # and are thus omitted here

def test_imports():
    """Test that all expected classes are importable."""
    from semantrix.embedding import (
        BaseEmbedder,
        get_embedder,
        SentenceTransformerEmbedder,
        OnnxEmbedder,
    )
    
    # Just check that we can import everything without errors
    assert True