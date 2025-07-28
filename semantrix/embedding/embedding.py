from abc import ABC, abstractmethod
from typing import Any, Union
import numpy as np

class BaseEmbedder(ABC):
    """Abstract base class for text embedding models."""
    
    @abstractmethod
    def encode(self, text: str) -> Union[np.ndarray, list[float]]:
        """
        Encode text into a vector representation.
        
        Args:
            text (str): The text to encode
            
        Returns:
            Union[np.ndarray, list[float]]: The encoded vector
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

class Embedder(BaseEmbedder):
    """Default embedder implementation using sentence-transformers."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedder.
        
        Args:
            model_name (str): The sentence-transformers model to use
        """
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name)
            self._dimension = self.model.get_sentence_embedding_dimension()
        except ImportError:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
    
    def encode(self, text: str) -> np.ndarray:
        """Encode text into a vector representation."""
        return self.model.encode(text)
    
    def get_dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return self._dimension 