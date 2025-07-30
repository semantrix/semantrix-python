#!/usr/bin/env python3
"""
Example: Custom implementations for Semantrix components
=====================================================

This example shows how to create custom implementations of:
- Embedders (BaseEmbedder)
- Vector Stores (BaseVectorStore) 
- Cache Stores (BaseCacheStore)

And how to use them with SemantrixClient.
"""

import numpy as np
from typing import Optional, List, Tuple, Union
from collections import OrderedDict

# Import the base classes and SemantrixClient
from semantrix.client import SemantrixClient
from semantrix.embedding import BaseEmbedder, get_embedder
from semantrix.vector_store.vector_store import BaseVectorStore
from semantrix.cache_store import BaseCacheStore


# Example 1: Custom Embedder using a simple hash-based approach
class HashEmbedder(BaseEmbedder):
    """Simple hash-based embedder for demonstration."""
    
    def __init__(self, dimension: int = 128):
        self.dimension = dimension
    
    def encode(self, text: str) -> np.ndarray:
        """Create a simple hash-based embedding."""
        import hashlib
        
        # Create a hash of the text
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to numpy array
        embedding = np.frombuffer(hash_bytes, dtype=np.float32)
        
        # Pad or truncate to desired dimension
        if len(embedding) < self.dimension:
            # Pad with zeros
            embedding = np.pad(embedding, (0, self.dimension - len(embedding)))
        else:
            # Truncate
            embedding = embedding[:self.dimension]
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding
    
    def get_dimension(self) -> int:
        return self.dimension


# Example 2: Custom Vector Store using simple cosine similarity
class SimpleVectorStore(BaseVectorStore):
    """Simple in-memory vector store using cosine similarity."""
    
    def __init__(self, dimension: int = 128):
        self.embeddings = []
        self.responses = []
        self.dimension = dimension
    
    def add(self, embedding: Union[np.ndarray, list[float]], response: str) -> None:
        """Add an embedding-response pair."""
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        self.embeddings.append(embedding)
        self.responses.append(response)
    
    def search(self, 
               embedding: Union[np.ndarray, list[float]], 
               similarity_threshold: Optional[float] = None,
               top_k: Optional[int] = None) -> Optional[Union[str, List[Tuple[str, float]]]]:
        """Search for similar embeddings using cosine similarity."""
        if isinstance(embedding, list):
            embedding = np.array(embedding, dtype=np.float32)
        
        # Normalize query embedding
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        if not self.embeddings:
            return None
        
        # Calculate similarities
        similarities = []
        for stored_embedding in self.embeddings:
            similarity = np.dot(embedding, stored_embedding)
            similarities.append(similarity)
        
        # Find best matches
        if top_k is None:
            # Return best match if above threshold
            if similarity_threshold is None:
                similarity_threshold = 0.8
            
            best_idx = np.argmax(similarities)
            if similarities[best_idx] >= similarity_threshold:
                return self.responses[best_idx]
            return None
        else:
            # Return top_k results
            indices = np.argsort(similarities)[::-1][:min(top_k, len(similarities))]
            return [(self.responses[idx], float(similarities[idx])) for idx in indices]
    
    def clear(self) -> None:
        """Clear all stored embeddings."""
        self.embeddings = []
        self.responses = []
    
    def size(self) -> int:
        """Get the number of stored embeddings."""
        return len(self.embeddings)


# Example 3: Custom Cache Store with TTL (Time To Live)
class TTLCacheStore(BaseCacheStore):
    """Cache store with Time To Live functionality."""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.cache = OrderedDict()
        self.timestamps = OrderedDict()
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
    
    def get_exact(self, prompt: str) -> Optional[str]:
        """Get an exact match, checking TTL."""
        import time
        
        if prompt in self.cache:
            # Check if expired
            if time.time() - self.timestamps[prompt] > self.ttl_seconds:
                # Remove expired item
                del self.cache[prompt]
                del self.timestamps[prompt]
                return None
            
            # Move to end (LRU)
            self.cache.move_to_end(prompt)
            self.timestamps.move_to_end(prompt)
            return self.cache[prompt]
        return None
    
    def add(self, prompt: str, response: str) -> None:
        """Add a prompt-response pair with timestamp."""
        import time
        
        self.cache[prompt] = response
        self.timestamps[prompt] = time.time()
        
        # Move to end (LRU)
        self.cache.move_to_end(prompt)
        self.timestamps.move_to_end(prompt)
    
    def enforce_limits(self, resource_limits) -> None:
        """Enforce limits by removing expired and oldest items."""
        import time
        
        current_time = time.time()
        
        # Remove expired items
        expired_keys = [
            key for key, timestamp in self.timestamps.items()
            if current_time - timestamp > self.ttl_seconds
        ]
        for key in expired_keys:
            del self.cache[key]
            del self.timestamps[key]
        
        # Remove oldest items if still over limit
        while len(self.cache) > self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            del self.timestamps[oldest_key]
    
    def clear(self) -> None:
        """Clear all cached items."""
        self.cache.clear()
        self.timestamps.clear()
    
    def size(self) -> int:
        """Get the number of cached items."""
        return len(self.cache)


def main():
    """Demonstrate custom implementations."""
    print("=== Custom Implementations Example ===\n")
    
    # Example 1: Use custom embedder
    print("1. Using custom hash-based embedder:")
    custom_embedder = HashEmbedder(dimension=64)
    client1 = SemantrixClient(
        embedder=custom_embedder,
        similarity_threshold=0.7
    )
    
    # Test the custom embedder
    client1.set("What is AI?", "Artificial Intelligence is a field of computer science.")
    result = client1.get("Explain artificial intelligence")
    print(f"   Result: {result}")
    
    # Example 2: Use custom vector store
    print("\n2. Using custom simple vector store:")
    custom_vector_store = SimpleVectorStore(dimension=64)
    client2 = SemantrixClient(
        vector_store=custom_vector_store,
        similarity_threshold=0.8
    )
    
    client2.set("How does machine learning work?", "Machine learning uses algorithms to learn patterns.")
    result = client2.get("Explain machine learning")
    print(f"   Result: {result}")
    
    # Example 3: Use custom cache store with TTL
    print("\n3. Using custom TTL cache store:")
    custom_cache_store = TTLCacheStore(max_size=100, ttl_seconds=60)  # 1 minute TTL
    client3 = SemantrixClient(
        cache_store=custom_cache_store,
        similarity_threshold=0.85
    )
    
    client3.set("What is Python?", "Python is a programming language.")
    result = client3.get("What is Python?")
    print(f"   Result: {result}")
    
    # Example 4: Mix and match components
    print("\n4. Mixing custom components:")
    client4 = SemantrixClient(
        embedder=custom_embedder,
        vector_store=custom_vector_store,
        cache_store=custom_cache_store,
        similarity_threshold=0.75
    )
    
    client4.set("What is data science?", "Data science combines statistics and programming.")
    result = client4.get("Explain data science")
    print(f"   Result: {result}")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main() 