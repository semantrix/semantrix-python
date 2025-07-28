from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Union, Any
import numpy as np

class BaseVectorStore(ABC):
    """Abstract base class for vector storage and similarity search."""
    
    @abstractmethod
    def add(self, embedding: Union[np.ndarray, list[float]], response: str) -> None:
        """
        Add an embedding-response pair to the vector store.
        
        Args:
            embedding: The vector embedding
            response: The response text associated with the embedding
        """
        pass
    
    @abstractmethod
    def search(self, 
               embedding: Union[np.ndarray, list[float]], 
               similarity_threshold: Optional[float] = None,
               top_k: Optional[int] = None) -> Optional[Union[str, List[Tuple[str, float]]]]:
        """
        Search for similar embeddings.
        
        Args:
            embedding: The query embedding
            similarity_threshold: Minimum similarity score (0-1)
            top_k: Number of top results to return
            
        Returns:
            If top_k is None and similarity_threshold is set: returns the best match string or None
            If top_k is set: returns list of (response, similarity_score) tuples
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all stored embeddings."""
        pass
    
    @abstractmethod
    def size(self) -> int:
        """Get the number of stored embeddings."""
        pass

class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector store implementation."""
    
    def __init__(self, dimension: int = 384):
        """
        Initialize the FAISS vector store.
        
        Args:
            dimension: The dimension of the embedding vectors
        """
        try:
            import faiss
            self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            self.responses = []
            self.dimension = dimension
        except ImportError:
            raise ImportError("faiss-cpu is required. Install with: pip install faiss-cpu")
    
    def add(self, embedding: Union[np.ndarray, list[float]], response: str) -> None:
        """Add an embedding-response pair to the vector store."""
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
        elif embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)
        
        # Normalize for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        
        self.index.add(embedding.reshape(1, -1))
        self.responses.append(response)
    
    def search(self, 
               embedding: Union[np.ndarray, list[float]], 
               similarity_threshold: Optional[float] = None,
               top_k: Optional[int] = None) -> Optional[Union[str, List[Tuple[str, float]]]]:
        """Search for similar embeddings."""
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
        elif embedding.dtype != np.float32:
            embedding = embedding.astype(np.float32)
        
        # Normalize for cosine similarity
        embedding = embedding / np.linalg.norm(embedding)
        
        if top_k is None:
            # Return best match if above threshold
            if similarity_threshold is None:
                similarity_threshold = 0.8
            
            scores, indices = self.index.search(embedding.reshape(1, -1), 1)
            if scores[0][0] >= similarity_threshold:
                return self.responses[indices[0][0]]
            return None
        else:
            # Return top_k results
            scores, indices = self.index.search(embedding.reshape(1, -1), min(top_k, len(self.responses)))
            return [(self.responses[idx], float(score)) for score, idx in zip(scores[0], indices[0])]
    
    def clear(self) -> None:
        """Clear all stored embeddings."""
        self.index = type(self.index)(self.dimension)
        self.responses = []
    
    def size(self) -> int:
        """Get the number of stored embeddings."""
        return len(self.responses) 