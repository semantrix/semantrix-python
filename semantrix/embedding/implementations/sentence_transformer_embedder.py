"""
Sentence Transformer Embedder implementation using the sentence-transformers library.

This module provides an async-compatible wrapper around sentence-transformers for text embedding.
"""

import asyncio
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from semantrix.embedding.base import BaseEmbedder


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Embedder implementation using sentence-transformers with async support.
    
    This embedder provides an async interface for encoding text using sentence-transformers models.
    It uses a thread pool executor to run the CPU-bound encoding operations asynchronously.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32):
        """
        Initialize the embedder.
        
        Args:
            model_name: The sentence-transformers model to use
            batch_size: Batch size for batch encoding operations
            
        Raises:
            ImportError: If sentence-transformers is not installed
        """
        try:
            self.model = SentenceTransformer(model_name)
            self._dimension = self.model.get_sentence_embedding_dimension()
            self.batch_size = batch_size
            self._model_lock = asyncio.Lock()  # Thread safety for the model
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required. "
                "Install with: pip install sentence-transformers"
            ) from e
    
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
            return await loop.run_in_executor(
                None,  # Use default executor
                lambda: self.model.encode(text, convert_to_numpy=True)
            )
    
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
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            loop = asyncio.get_running_loop()
            async with self._model_lock:
                batch_results = await loop.run_in_executor(
                    None,
                    lambda b=batch: self.model.encode(b, convert_to_numpy=True)
                )
                # Convert to list of arrays if needed
                if isinstance(batch_results, np.ndarray):
                    results.extend(batch_results)
                else:
                    results.extend(list(batch_results))
        return results
    
    def get_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Dimension of the embedding vectors
        """
        return self._dimension
