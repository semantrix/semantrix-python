"""
ONNX Runtime Embedder implementation for fast inference of transformer models.

This module provides an async-only wrapper around ONNX Runtime for text embedding.
"""

import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
from onnxruntime import InferenceSession, SessionOptions
from transformers import AutoTokenizer

from semantrix.embedding.base import BaseEmbedder


class OnnxEmbedder(BaseEmbedder):
    """
    Embedder implementation using ONNX Runtime for fast inference.
    
    This embedder provides an async interface for encoding text using ONNX Runtime.
    It's optimized for production use with ONNX Runtime and supports both CPU and GPU execution.
    """
    
    def __init__(
        self,
        model_path: str,
        tokenizer_name: Optional[str] = None,
        batch_size: int = 32,
        provider: str = "CUDAExecutionProvider",
        provider_options: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the ONNX embedder.
        
        Args:
            model_path: Path to the ONNX model file or directory containing model.onnx
            tokenizer_name: Name of the tokenizer (defaults to model_path if None)
            batch_size: Batch size for batch encoding operations
            provider: ONNX Runtime execution provider (e.g., 'CUDAExecutionProvider', 'CPUExecutionProvider')
            provider_options: Additional options for the execution provider
            
        Raises:
            ImportError: If onnxruntime or transformers is not installed
            FileNotFoundError: If the model file is not found
        """
        try:
            import onnxruntime
        except ImportError as e:
            raise ImportError(
                "onnxruntime is required for OnnxEmbedder. "
                "Install with: pip install onnxruntime"
            ) from e
            
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_path)
        
        # Set up ONNX session options
        sess_options = SessionOptions()
        
        # Initialize ONNX Runtime session
        providers = [
            (provider, provider_options or {}),
            'CPUExecutionProvider'  # Fallback to CPU if GPU is not available
        ]
        
        # Check if model_path is a directory containing model.onnx
        model_dir = Path(model_path)
        if model_dir.is_dir():
            model_path = str(model_dir / "model.onnx")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"ONNX model file not found at {model_path}")
        
        self.session = InferenceSession(model_path, sess_options, providers=providers)
        
        # Get model info
        self._dimension = self.session.get_outputs()[0].shape[-1]
        self.batch_size = batch_size
        self._model_lock = asyncio.Lock()
    
    async def encode(self, text: str) -> np.ndarray:
        """
        Encode text into a vector representation.
        
        Args:
            text: The text to encode
            
        Returns:
            Encoded vector as a numpy array with shape (dimension,)
        """
        # Use a thread pool to run the CPU-bound encoding
        loop = asyncio.get_running_loop()
        async with self._model_lock:
            return await loop.run_in_executor(None, self._encode_batch, [text])[0]
    
    async def batch_encode(self, texts: List[str]) -> List[np.ndarray]:
        """
        Encode multiple texts into vector representations.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            List of encoded vectors, one per input text
            
        Note:
            This implementation processes texts in batches to optimize performance.
            The batch size is controlled by the `batch_size` parameter passed to the constructor.
        """
        # Process in batches to avoid memory issues
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            loop = asyncio.get_running_loop()
            async with self._model_lock:
                batch_results = await loop.run_in_executor(
                    None, self._encode_batch, batch
                )
                results.extend(batch_results)
        return results
    
    def _encode_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Internal method to encode a batch of texts.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            List of encoded vectors, each with shape (dimension,)
        """
        # Tokenize the input texts
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="np",
            max_length=512
        )
        
        # Run inference
        outputs = self.session.run(None, dict(inputs))
        
        # Get the embeddings (assuming the model outputs [batch_size, seq_len, hidden_size])
        # and take the mean of the token embeddings to get sentence embeddings
        embeddings = outputs[0]
        attention_mask = inputs['attention_mask']
        
        # Create attention mask with the same shape as embeddings
        input_mask_expanded = np.broadcast_to(
            np.expand_dims(attention_mask, -1),
            embeddings.shape
        )
        
        # Apply attention mask and sum along sequence dimension
        sum_embeddings = np.sum(embeddings * input_mask_expanded, axis=1)
        
        # Normalize by the square root of the sum of the mask
        sum_mask = np.maximum(np.sum(input_mask_expanded, axis=1), 1e-9)
        sentence_embeddings = sum_embeddings / np.sqrt(sum_mask)
        
        # Normalize to unit vectors (cosine similarity)
        norms = np.linalg.norm(sentence_embeddings, axis=1, keepdims=True)
        sentence_embeddings = sentence_embeddings / norms
        
        return [emb for emb in sentence_embeddings]
    
    def get_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Dimension of the embedding vectors
        """
        return self._dimension
