"""
OpenAI Embedder implementation for text embedding using OpenAI's API.

This module provides an async interface for encoding text using OpenAI's embedding models.
"""

import asyncio
from typing import List, Optional, Dict, Any

import numpy as np
from openai import AsyncOpenAI, OpenAIError

from semantrix.embedding.base import BaseEmbedder


class OpenAIEmbedder(BaseEmbedder):
    """
    Embedder implementation using OpenAI's embedding API.
    
    This embedder provides an async interface for encoding text using OpenAI's API.
    It supports all models available through the OpenAI embeddings endpoint.
    """
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        **kwargs: Any
    ):
        """
        Initialize the OpenAI embedder.
        
        Args:
            model: The name of the OpenAI embedding model to use
            api_key: Your OpenAI API key. If not provided, will use OPENAI_API_KEY env var
            organization: Your OpenAI organization ID
            base_url: Custom base URL for the API (for proxies or self-hosted)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            **kwargs: Additional arguments to pass to the AsyncOpenAI client
        """
        self.model = model
        self._client = AsyncOpenAI(
            api_key=api_key,
            organization=organization,
            base_url=base_url,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs
        )
        self._dimension = self._get_model_dimension(model)
        self._model_lock = asyncio.Lock()
    
    def _get_model_dimension(self, model: str) -> int:
        """Get the expected dimension for a given OpenAI model."""
        # Common OpenAI embedding model dimensions
        model_dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
        return model_dimensions.get(model, 1536)  # Default to 1536 if unknown
    
    async def encode(self, text: str) -> np.ndarray:
        """
        Encode text into a vector representation using OpenAI's API.
        
        Args:
            text: The text to encode
            
        Returns:
            Encoded vector as a numpy array with shape (dimension,)
            
        Raises:
            RuntimeError: If the API request fails
        """
        async with self._model_lock:
            try:
                response = await self._client.embeddings.create(
                    input=text,
                    model=self.model
                )
                return np.array(response.data[0].embedding, dtype=np.float32)
            except OpenAIError as e:
                raise RuntimeError(f"OpenAI API error: {str(e)}") from e
    
    async def batch_encode(self, texts: List[str]) -> List[np.ndarray]:
        """
        Encode multiple texts into vector representations using OpenAI's API.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            List of encoded vectors as numpy arrays
            
        Raises:
            RuntimeError: If the API request fails
        """
        if not texts:
            return []
            
        async with self._model_lock:
            try:
                response = await self._client.embeddings.create(
                    input=texts,
                    model=self.model
                )
                return [
                    np.array(item.embedding, dtype=np.float32)
                    for item in response.data
                ]
            except OpenAIError as e:
                raise RuntimeError(f"OpenAI API error: {str(e)}") from e
    
    def get_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Dimension of the embedding vectors
        """
        return self._dimension
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._client.close()
