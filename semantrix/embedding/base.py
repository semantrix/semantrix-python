"""
Base embedding module for Semantrix.

This module contains the base abstract class for all embedder implementations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

class BaseEmbedder(ABC):
    """Abstract base class for text embedding models with async support.
    
    This class provides an async-first interface for text embedding models.
    All implementations should override the async methods for optimal performance.
    """
    
    @abstractmethod
    async def encode(self, text: str) -> np.ndarray:
        """
        Encode text into a vector representation.
        
        Args:
            text: The text to encode
            
        Returns:
            np.ndarray: The encoded vector with shape (dimension,)
        """
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            int: The dimension of the embedding vectors
        """
        pass
    
    async def batch_encode(self, texts: List[str]) -> List[np.ndarray]:
        """
        Encode multiple texts into vector representations.
        
        Args:
            texts: List of texts to encode
            
        Returns:
            List[np.ndarray]: List of encoded vectors, one per input text
            
        Note:
            The default implementation processes texts sequentially.
            Override this method for batch processing optimizations.
        """
        return [await self.encode(text) for text in texts]
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass