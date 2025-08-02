"""
LangChain Embedder implementation for text embedding using LangChain's embedding models.

This module provides an async interface for encoding text using any embedding model
supported by LangChain's integration ecosystem.
"""

import asyncio
from typing import List, Optional, Dict, Any, Type, Union

import numpy as np
from langchain_core.embeddings import Embeddings as LangChainEmbeddings
from langchain_core.runnables import RunnableConfig

from semantrix.embedding.base import BaseEmbedder


class LangChainEmbedder(BaseEmbedder):
    """
    Embedder implementation using LangChain's embedding models.
    
    This embedder provides an async interface for encoding text using any
    LangChain-compatible embedding model.
    """
    
    def __init__(
        self,
        embeddings: LangChainEmbeddings,
        batch_size: int = 32,
        **embed_kwargs: Any
    ):
        """
        Initialize the LangChain embedder.
        
        Args:
            embeddings: An instance of a LangChain Embeddings class
            batch_size: Batch size for batch encoding operations
            **embed_kwargs: Additional keyword arguments to pass to the embedder
        """
        self.embeddings = embeddings
        self.batch_size = batch_size
        self.embed_kwargs = embed_kwargs
        self._model_lock = asyncio.Lock()
        
        # Get the embedding dimension by making a test embedding
        test_embedding = self.embeddings.embed_query("test")
        self._dimension = len(test_embedding) if test_embedding else 0
    
    async def encode(self, text: str) -> np.ndarray:
        """
        Encode text into a vector representation using LangChain.
        
        Args:
            text: The text to encode
            
        Returns:
            Encoded vector as a numpy array with shape (dimension,)
            
        Raises:
            RuntimeError: If the embedding fails
        """
        try:
            # Use a thread pool to run the potentially blocking call
            loop = asyncio.get_running_loop()
            async with self._model_lock:
                embedding = await loop.run_in_executor(
                    None,
                    lambda: self.embeddings.embed_query(text)
                )
                return np.array(embedding, dtype=np.float32)
        except Exception as e:
            raise RuntimeError(f"LangChain embedding error: {str(e)}") from e
    
    async def batch_encode(self, texts: List[str]) -> List[np.ndarray]:
        """
        Encode multiple texts into vector representations using LangChain.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            List of encoded vectors as numpy arrays
            
        Raises:
            RuntimeError: If the embedding fails
        """
        if not texts:
            return []
            
        try:
            # Process in batches to avoid timeouts with large numbers of texts
            results = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i + self.batch_size]
                # Use a thread pool to run the potentially blocking call
                loop = asyncio.get_running_loop()
                async with self._model_lock:
                    batch_embeddings = await loop.run_in_executor(
                        None,
                        lambda b=batch: self.embeddings.embed_documents(b)
                    )
                    results.extend([
                        np.array(embedding, dtype=np.float32)
                        for embedding in batch_embeddings
                    ])
            return results
        except Exception as e:
            raise RuntimeError(f"LangChain batch embedding error: {str(e)}") from e
    
    def get_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Dimension of the embedding vectors
        """
        return self._dimension
    
    @classmethod
    def from_model_name(
        cls,
        model_name: str,
        batch_size: int = 32,
        **model_kwargs: Any
    ) -> 'LangChainEmbedder':
        """
        Create a LangChainEmbedder from a model name.
        
        This is a convenience method that creates the appropriate LangChain
        embeddings class based on the model name.
        
        Args:
            model_name: Name of the model to use (e.g., 'text-embedding-3-small')
            batch_size: Batch size for batch encoding operations
            **model_kwargs: Additional keyword arguments to pass to the model
            
        Returns:
            An instance of LangChainEmbedder
            
        Raises:
            ImportError: If the required LangChain integration is not installed
            ValueError: If the model name is not recognized
        """
        try:
            if 'text-embedding' in model_name or 'ada' in model_name:
                from langchain_openai import OpenAIEmbeddings
                embeddings = OpenAIEmbeddings(model=model_name, **model_kwargs)
            elif 'all-' in model_name.lower() or 'multi-qa' in model_name.lower():
                from langchain_community.embeddings import HuggingFaceEmbeddings
                embeddings = HuggingFaceEmbeddings(model_name=model_name, **model_kwargs)
            elif 'gte-' in model_name.lower() or 'bge-' in model_name.lower():
                from langchain_community.embeddings import HuggingFaceBgeEmbeddings
                embeddings = HuggingFaceBgeEmbeddings(model_name=model_name, **model_kwargs)
            else:
                # Try to use HuggingFace as a fallback
                from langchain_community.embeddings import HuggingFaceEmbeddings
                embeddings = HuggingFaceEmbeddings(model_name=model_name, **model_kwargs)
                
            return cls(embeddings=embeddings, batch_size=batch_size)
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import required LangChain module for {model_name}. "
                f"You may need to install additional packages. Original error: {str(e)}"
            )
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Some LangChain embeddings might need cleanup
        if hasattr(self.embeddings, 'client') and hasattr(self.embeddings.client, 'close'):
            await asyncio.get_event_loop().run_in_executor(
                None,
                self.embeddings.client.close
            )
