"""
Cohere Embedder implementation for text embedding using Cohere's API.

This module provides an async interface for encoding text using Cohere's embedding models.
"""

import asyncio
from typing import List, Optional, Dict, Any, Union, Literal

import numpy as np

from semantrix.embedding.base import BaseEmbedder


try:
    import cohere
    COHERE_AVAILABLE = True
except ImportError:
    COHERE_AVAILABLE = False


class CohereEmbedder(BaseEmbedder):
    """
    Embedder implementation using Cohere's embedding API.
    
    This embedder provides an async interface for encoding text using Cohere's
    embedding models.
    """
    
    class Config:
        """Configuration for CohereEmbedder."""
        model: str = "embed-english-v3.0"
        input_type: Literal["search_document", "search_query", "classification", "clustering"] = "search_document"
        truncate: Optional[Literal["NONE", "START", "END"]] = "END"
        
    def __init__(
        self,
        model: str = "embed-english-v3.0",
        api_key: Optional[str] = None,
        input_type: str = "search_document",
        truncate: Optional[str] = "END",
        timeout: float = 30.0,
        max_retries: int = 3,
        **client_kwargs: Any
    ):
        """
        Initialize the Cohere embedder.
        
        Args:
            model: The name of the Cohere model to use for embeddings
            api_key: Your Cohere API key. If not provided, will use COHERE_API_KEY env var
            input_type: The type of input for the embedding. One of: 'search_document', 
                      'search_query', 'classification', 'clustering'
            truncate: How to handle inputs longer than the maximum token length. 
                     One of 'NONE', 'START', 'END'
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
            **client_kwargs: Additional arguments to pass to the Cohere client
        """
        if not COHERE_AVAILABLE:
            raise ImportError(
                "Cohere client is required. "
                "Install with: pip install cohere"
            )
            
        self.config = self.Config(
            model=model,
            input_type=input_type,  # type: ignore
            truncate=truncate  # type: ignore
        )
        
        self._client_kwargs = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries,
            **client_kwargs
        }
        
        self._client: Optional[cohere.Client] = None
        self._model_lock = asyncio.Lock()
        
        # Initialize with default dimension, will be updated on first use
        self._dimension = self._get_model_dimension(model)
    
    def _get_model_dimension(self, model: str) -> int:
        """Get the expected dimension for a given Cohere model."""
        # Common Cohere embedding model dimensions
        model_dimensions = {
            "embed-english-v3.0": 1024,
            "embed-multilingual-v3.0": 1024,
            "embed-english-v2.0": 4096,
            "embed-multilingual-v2.0": 768,
            "embed-english-light-v3.0": 384,
            "embed-multilingual-light-v3.0": 384,
            "embed-english-light-v2.0": 1024,
            "embed-english-v1.0": 4096,
            "embed-multilingual-v1.0": 768,
        }
        return model_dimensions.get(model, 1024)  # Default to 1024 if unknown
    
    def _get_client(self) -> 'cohere.Client':
        """Get or create a Cohere client."""
        if self._client is None:
            self._client = cohere.Client(**self._client_kwargs)
        return self._client
    
    async def _async_embed(
        self,
        texts: List[str],
        input_type: Optional[str] = None,
        **kwargs: Any
    ) -> List[List[float]]:
        """Async helper to run sync client methods in a thread pool."""
        loop = asyncio.get_running_loop()
        client = self._get_client()
        
        def _sync_embed() -> List[List[float]]:
            response = client.embed(
                texts=texts,
                model=self.config.model,
                input_type=input_type or self.config.input_type,
                truncate=self.config.truncate,
                **kwargs
            )
            return response.embeddings
        
        return await loop.run_in_executor(None, _sync_embed)
    
    async def encode(self, text: str, **kwargs: Any) -> np.ndarray:
        """
        Encode text into a vector representation using Cohere.
        
        Args:
            text: The text to encode
            **kwargs: Additional arguments to pass to the embedding method
            
        Returns:
            Encoded vector as a numpy array with shape (dimension,)
            
        Raises:
            RuntimeError: If the embedding fails
        """
        async with self._model_lock:
            try:
                embeddings = await self._async_embed([text], **kwargs)
                if not embeddings or not embeddings[0]:
                    raise ValueError("No embeddings returned from Cohere API")
                
                # Update dimension on first use
                if not hasattr(self, '_dimension') or not self._dimension:
                    self._dimension = len(embeddings[0])
                
                return np.array(embeddings[0], dtype=np.float32)
                
            except Exception as e:
                raise RuntimeError(f"Cohere API error: {str(e)}") from e
    
    async def batch_encode(self, texts: List[str], **kwargs: Any) -> List[np.ndarray]:
        """
        Encode multiple texts into vector representations using Cohere.
        
        Args:
            texts: List of texts to encode
            **kwargs: Additional arguments to pass to the embedding method
            
        Returns:
            List of encoded vectors as numpy arrays
            
        Raises:
            RuntimeError: If the embedding fails
        """
        if not texts:
            return []
            
        async with self._model_lock:
            try:
                embeddings = await self._async_embed(texts, **kwargs)
                
                # Update dimension on first use
                if embeddings and (not hasattr(self, '_dimension') or not self._dimension):
                    self._dimension = len(embeddings[0]) if embeddings[0] else 0
                
                return [
                    np.array(embedding, dtype=np.float32)
                    for embedding in embeddings
                ]
                
            except Exception as e:
                raise RuntimeError(f"Cohere batch embedding error: {str(e)}") from e
    
    def get_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Dimension of the embedding vectors
            
        Note:
            The dimension is determined on the first call to encode() or batch_encode().
            Before that, it returns a default value based on the model.
        """
        return getattr(self, '_dimension', 1024)  # Default to 1024 if not set yet
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'CohereEmbedder':
        """
        Create a CohereEmbedder from a configuration dictionary.
        
        Args:
            config: Configuration dictionary with parameters for CohereEmbedder
            
        Returns:
            An instance of CohereEmbedder
        """
        return cls(**config)
    
    def to_config(self) -> Dict[str, Any]:
        """
        Convert the embedder's configuration to a dictionary.
        
        Returns:
            Dictionary containing the embedder's configuration
        """
        return {
            "model": self.config.model,
            "input_type": self.config.input_type,
            "truncate": self.config.truncate,
            **{k: v for k, v in self._client_kwargs.items() if k != 'api_key'}
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Cohere client doesn't require explicit cleanup, but we'll clear the reference
        self._client = None
