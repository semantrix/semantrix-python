"""
Mistral Embedder implementation for text embedding using Mistral's models.

This module provides an async interface for encoding text using Mistral's
embedding models, both via their API and local models.
"""

import asyncio
from typing import List, Optional, Dict, Any, Union

import numpy as np
from pydantic import BaseModel, Field

from semantrix.embedding.base import BaseEmbedder


try:
    from mistralai.client import MistralClient
    from mistralai.models.embeddings import EmbeddingResponse
    MISTRAL_AVAILABLE = True
except ImportError:
    MISTRAL_AVAILABLE = False


class MistralEmbedder(BaseEmbedder):
    """
    Embedder implementation using Mistral's embedding models.
    
    This embedder provides an async interface for encoding text using Mistral's
    embedding models, either via their API or local models.
    """
    
    class Config(BaseModel):
        """Configuration for MistralEmbedder."""
        model: str = "mistral-embed"
        api_key: Optional[str] = None
        endpoint: Optional[str] = None
        timeout: float = 30.0
        max_retries: int = 3
        batch_size: int = 32
        
    def __init__(
        self,
        model: str = "mistral-embed",
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        batch_size: int = 32,
        **client_kwargs: Any
    ):
        """
        Initialize the Mistral embedder.
        
        Args:
            model: The name of the Mistral model to use for embeddings
            api_key: Your Mistral API key. If not provided, will use MISTRAL_API_KEY env var
            endpoint: Custom API endpoint (for self-hosted or custom deployments)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            batch_size: Number of texts to process in each batch
            **client_kwargs: Additional arguments to pass to the MistralClient
        """
        if not MISTRAL_AVAILABLE:
            raise ImportError(
                "Mistral AI client is required. "
                "Install with: pip install mistralai"
            )
            
        self.config = self.Config(
            model=model,
            api_key=api_key,
            endpoint=endpoint,
            timeout=timeout,
            max_retries=max_retries,
            batch_size=batch_size
        )
        
        self._client_kwargs = client_kwargs
        self._client: Optional[MistralClient] = None
        self._model_lock = asyncio.Lock()
        
        # Initialize with default dimension, will be updated on first use
        self._dimension = 1024  # Default for mistral-embed
    
    def _get_client(self) -> MistralClient:
        """Get or create a Mistral client."""
        if self._client is None:
            self._client = MistralClient(
                api_key=self.config.api_key,
                endpoint=self.config.endpoint,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
                **self._client_kwargs
            )
        return self._client
    
    async def _async_embed_batch(
        self,
        texts: List[str],
        model: Optional[str] = None,
        **kwargs: Any
    ) -> List[List[float]]:
        """Async helper to run sync client methods in a thread pool."""
        loop = asyncio.get_running_loop()
        client = self._get_client()
        
        def _sync_embed() -> List[List[float]]:
            response = client.embeddings(
                model=model or self.config.model,
                input=texts,
                **kwargs
            )
            return [item.embedding for item in response.data]
        
        return await loop.run_in_executor(None, _sync_embed)
    
    async def encode(self, text: str) -> np.ndarray:
        """
        Encode text into a vector representation using Mistral.
        
        Args:
            text: The text to encode
            
        Returns:
            Encoded vector as a numpy array with shape (dimension,)
            
        Raises:
            RuntimeError: If the embedding fails
        """
        async with self._model_lock:
            embeddings = await self._async_embed_batch([text])
            if not embeddings or not embeddings[0]:
                raise RuntimeError("Failed to generate embeddings")
                
            # Update dimension on first use
            if not hasattr(self, '_dimension') or not self._dimension:
                self._dimension = len(embeddings[0])
                
            return np.array(embeddings[0], dtype=np.float32)
    
    async def batch_encode(self, texts: List[str]) -> List[np.ndarray]:
        """
        Encode multiple texts into vector representations using Mistral.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            List of encoded vectors as numpy arrays
            
        Raises:
            RuntimeError: If the embedding fails
        """
        if not texts:
            return []
            
        # Process in batches
        results = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i:i + self.config.batch_size]
            batch_embeddings = await self._async_embed_batch(batch)
            
            # Update dimension on first use
            if not hasattr(self, '_dimension') or not self._dimension and batch_embeddings:
                self._dimension = len(batch_embeddings[0])
                
            results.extend([
                np.array(embedding, dtype=np.float32)
                for embedding in batch_embeddings
            ])
            
        return results
    
    def get_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Dimension of the embedding vectors
            
        Note:
            The dimension is determined on the first call to encode() or batch_encode().
            Before that, it returns a default value of 1024 (for mistral-embed).
        """
        return getattr(self, '_dimension', 1024)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'MistralEmbedder':
        """
        Create a MistralEmbedder from a configuration dictionary.
        
        Args:
            config: Configuration dictionary with parameters for MistralEmbedder
            
        Returns:
            An instance of MistralEmbedder
        """
        return cls(**config)
    
    def to_config(self) -> Dict[str, Any]:
        """
        Convert the embedder's configuration to a dictionary.
        
        Returns:
            Dictionary containing the embedder's configuration
        """
        return self.config.dict()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # MistralClient doesn't require explicit cleanup, but we'll clear the reference
        self._client = None
