"""
Ollama Embedder implementation for text embedding using Ollama's local models.

This module provides an async interface for encoding text using models served by Ollama.
"""

import asyncio
from typing import List, Optional, Dict, Any, Union

import httpx
import numpy as np

from semantrix.embedding.base import BaseEmbedder


class OllamaEmbedder(BaseEmbedder):
    """
    Embedder implementation using Ollama's local model serving.
    
    This embedder provides an async interface for encoding text using models
    served by Ollama (https://ollama.ai/).
    """
    
    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        timeout: float = 60.0,
        batch_size: int = 10,
        **client_kwargs: Any
    ):
        """
        Initialize the Ollama embedder.
        
        Args:
            model: The name of the Ollama model to use for embeddings
            base_url: Base URL of the Ollama API server
            timeout: Request timeout in seconds
            batch_size: Number of texts to process in each batch
            **client_kwargs: Additional arguments to pass to the httpx.AsyncClient
        """
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.batch_size = batch_size
        self.client_kwargs = client_kwargs
        
        # Initialize with default dimension, will be updated on first use
        self._dimension = 768  # Common default dimension
        self._client: Optional[httpx.AsyncClient] = None
        self._model_lock = asyncio.Lock()
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create an HTTP client with connection pooling."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                **self.client_kwargs
            )
        return self._client
    
    async def _ensure_dimension(self) -> None:
        """Ensure the embedding dimension is known."""
        if not hasattr(self, '_dimension') or not self._dimension:
            # Make a test embedding to get the dimension
            test_embedding = await self._embed_single("test")
            self._dimension = len(test_embedding)
    
    async def _embed_single(self, text: str) -> List[float]:
        """Embed a single text using the Ollama API."""
        client = await self._get_client()
        
        try:
            response = await client.post(
                "/api/embeddings",
                json={
                    "model": self.model,
                    "prompt": text
                }
            )
            response.raise_for_status()
            data = response.json()
            
            if 'embedding' not in data:
                raise ValueError("Invalid response format from Ollama API")
                
            return data['embedding']
            
        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"Ollama API request failed with status {e.response.status_code}: {e.response.text}"
            ) from e
        except httpx.RequestError as e:
            raise RuntimeError(
                f"Failed to connect to Ollama API at {self.base_url}. "
                "Is the Ollama server running?"
            ) from e
    
    async def encode(self, text: str) -> np.ndarray:
        """
        Encode text into a vector representation using Ollama.
        
        Args:
            text: The text to encode
            
        Returns:
            Encoded vector as a numpy array with shape (dimension,)
            
        Raises:
            RuntimeError: If the embedding fails
        """
        async with self._model_lock:
            embedding = await self._embed_single(text)
            await self._ensure_dimension()
            return np.array(embedding, dtype=np.float32)
    
    async def batch_encode(self, texts: List[str]) -> List[np.ndarray]:
        """
        Encode multiple texts into vector representations using Ollama.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            List of encoded vectors as numpy arrays
            
        Raises:
            RuntimeError: If the embedding fails
        """
        if not texts:
            return []
            
        # Process in batches to avoid overwhelming the server
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_embeddings = await asyncio.gather(
                *(self._embed_single(text) for text in batch)
            )
            results.extend([
                np.array(embedding, dtype=np.float32)
                for embedding in batch_embeddings
            ])
            
        await self._ensure_dimension()
        return results
    
    def get_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Dimension of the embedding vectors
            
        Note:
            The dimension is determined on the first call to encode() or batch_encode().
            Before that, it returns a default value of 768.
        """
        return getattr(self, '_dimension', 768)
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
