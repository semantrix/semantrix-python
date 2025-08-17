"""
Sentence Transformer Embedder implementation using the sentence-transformers library.

This module provides an async-compatible wrapper around sentence-transformers for text embedding.
"""

import asyncio
import time
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from semantrix.embedding.base import BaseEmbedder
from semantrix.utils.metrics import (
    SENTENCE_TRANSFORMER_OPERATIONS_COUNTER, SENTENCE_TRANSFORMER_LATENCY_HISTOGRAM,
    SENTENCE_TRANSFORMER_MODEL_LOADING_TIME_HISTOGRAM, SENTENCE_TRANSFORMER_INFERENCE_SPEED_HISTOGRAM,
    SENTENCE_TRANSFORMER_MEMORY_EFFICIENCY_GAUGE, EMBEDDING_MODEL_LATENCY_HISTOGRAM,
    EMBEDDING_MODEL_MEMORY_USAGE_GAUGE, EMBEDDING_MODEL_THROUGHPUT_HISTOGRAM
)


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
        # Increment sentence transformer operations counter
        SENTENCE_TRANSFORMER_OPERATIONS_COUNTER.increment()
        
        # Use timer for sentence transformer latency
        with SENTENCE_TRANSFORMER_LATENCY_HISTOGRAM.time() as timer:
            # Use timer for embedding model latency
            with EMBEDDING_MODEL_LATENCY_HISTOGRAM.time() as model_timer:
                # Use a thread pool to run the CPU-bound encoding
                loop = asyncio.get_running_loop()
                async with self._model_lock:
                    start_time = time.time()
                    result = await loop.run_in_executor(
                        None,  # Use default executor
                        lambda: self.model.encode(text, convert_to_numpy=True)
                    )
                    end_time = time.time()
                    
                    # Calculate throughput (embeddings per second)
                    duration = end_time - start_time
                    if duration > 0:
                        throughput = 1.0 / duration
                        EMBEDDING_MODEL_THROUGHPUT_HISTOGRAM.observe(throughput)
                        SENTENCE_TRANSFORMER_INFERENCE_SPEED_HISTOGRAM.observe(throughput)
                    
                    # Update memory efficiency metric (approximate)
                    if hasattr(self.model, 'get_sentence_embedding_dimension'):
                        dimension = self.model.get_sentence_embedding_dimension()
                        # Rough estimate of memory usage per embedding
                        memory_per_embedding = dimension * 4  # float32
                        SENTENCE_TRANSFORMER_MEMORY_EFFICIENCY_GAUGE.set(memory_per_embedding)
                    
                    return result
    
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
