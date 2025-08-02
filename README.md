# Semantrix · [![SSPL License](https://img.shields.io/badge/License-SSPL--1.0-ff69b4)](LICENSE)  
**Blazing-fast semantic caching for LLMs in Python, Java, and Go.**  

## Features  
- 🚀 Multi-language support (Python/Java/Go)  
- 🔍 Semantic matching (not just exact matches)  
- ⚡ Async-first API for high performance
- 🛡️ Resource management with async context managers
- 📦 Lightweight and efficient

## Quick Start  

### Installation
```bash
# Python
pip install semantrix-python

# For vector store backends, install the required dependencies:
pip install "semantrix-python[vector-stores]"

# Or install specific vector store dependencies as needed:
pip install faiss-cpu    # For FAISS (CPU version)
pip install chromadb     # For Chroma
pip install qdrant-client  # For Qdrant
pip install redis        # For Redis
pip install psycopg2-binary  # For pgvector
pip install pymilvus     # For Milvus

# Go
go get github.com/semantrix/go
```

### Basic Usage (Synchronous)
```python
from semantrix import Semantrix

# Create a cache instance
cache = Semantrix()

# Add to cache
cache.set("Explain quantum physics", "It's about...")

# Get from cache (with semantic matching)
response = cache.get("What is quantum theory?")
print(response)  # Returns cached response
```

### Async Usage (Recommended)

```python
import asyncio
from semantrix import Semantrix

async def main():
    # Create a cache instance (no cleanup needed)
    cache = Semantrix()
    
    # Add to cache
    await cache.set("Explain quantum physics", "It's about...")
    
    # Get from cache (with semantic matching)
    response = await cache.get("What is quantum theory?")
    print(response)  # Returns cached response
    
    # The cache will automatically clean up when it's garbage collected

# Run the async function
asyncio.run(main())
```

### Vector Store Usage

Semantrix supports multiple vector store backends through its caching system. Here's an example using FAISS as the vector store backend:

```python
import asyncio
import numpy as np
from semantrix import Semantrix
from semantrix.vector_store.stores import FAISSVectorStore
from semantrix.vector_store.base import DistanceMetric

async def main():
    # Initialize FAISS store
    vector_store = FAISSVectorStore(
        dimension=384,  # Dimension of your embeddings
        metric=DistanceMetric.COSINE
    )
    
    # Create a Semantrix cache with the FAISS vector store
    cache = Semantrix(
        vector_store=vector_store,
        similarity_threshold=0.8  # Adjust based on your needs
    )
    
    # Add documents to the cache (vectors are generated automatically)
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "A fast brown fox leaps over a sleeping dog",
        "The lazy dog sleeps under the tree",
        "The quick fox is very agile"
    ]
    
    for doc in documents:
        await cache.set(doc, f"Response for: {doc}")
    
    # Search for similar documents
    query = "A quick fox jumps over a dog"
    result = await cache.get(query)
    
    if result:
        print(f"Found similar document: {result}")
    else:
        print("No similar document found in cache")
    
    # The cache and vector store will be cleaned up automatically

asyncio.run(main())
```

### Available Vector Stores

Semantrix supports multiple vector store backends:

1. **FAISS** - In-memory similarity search (fast, no server required)
   ```python
   from semantrix.vector_store.stores import FAISSVectorStore
   store = FAISSVectorStore(dimension=384)
   ```

2. **Chroma** - Embedding database with persistence
   ```python
   from semantrix.vector_store.stores import ChromaVectorStore
   store = ChromaVectorStore(dimension=384, persist_directory="./chroma_db")
   ```

3. **pgvector** - PostgreSQL extension for vector search
   ```python
   from semantrix.vector_store.stores import PgVectorStore
   store = PgVectorStore(
       dimension=384,
       connection_string="postgresql://user:pass@localhost/vector_db"
   )
   ```

4. **Milvus** - High-performance vector database
   ```python
   from semantrix.vector_store.stores import MilvusVectorStore
   store = MilvusVectorStore(
       dimension=384,
       uri="http://localhost:19530",
       collection_name="documents"
   )
   ```

5. **Qdrant** - Vector search engine
   ```python
   from semantrix.vector_store.stores import QdrantVectorStore
   store = QdrantVectorStore(
       dimension=384,
       url="http://localhost:6333",
       collection_name="documents"
   )
   ```

6. **Redis** - In-memory data store with vector search
   ```python
   from semantrix.vector_store.stores import RedisVectorStore
   store = RedisVectorStore(
       dimension=384,
       redis_url="redis://localhost:6379",
       index_name="documents"
   )
   ```

See the [examples/vector_stores_example.py](examples/vector_stores_example.py) for complete usage examples.

### Advanced: Custom Cache Configuration

Semantrix provides flexible configuration options for fine-tuning cache behavior. Here's an example showing various customizations:

```python
import asyncio
import time
from semantrix import Semantrix
from semantrix.cache_store.stores import InMemoryStore, RedisCacheStore
from semantrix.cache_store.strategies import (
    LRUEvictionStrategy, 
    TTLStrategy, 
    SizeBasedEvictionStrategy
)
from semantrix.cache_store.eviction_policies import (
    StrategyBasedEvictionPolicy,
    CompositeEvictionPolicy
)
from semantrix.vector_store.stores import FAISSVectorStore
from semantrix.vector_store.base import DistanceMetric

async def main():
    # Create a composite eviction policy with multiple strategies
    ttl_strategy = TTLStrategy(ttl_seconds=3600)  # 1 hour TTL
    lru_strategy = LRUEvictionStrategy(max_size=1000)  # Max 1000 items
    size_strategy = SizeBasedEvictionStrategy(max_memory_mb=512)  # 512MB max memory
    
    # Combine strategies with different priorities
    eviction_policy = CompositeEvictionPolicy([
        (ttl_strategy, 1.0),    # Highest priority - always check TTL first
        (size_strategy, 0.7),   # Medium priority - check memory pressure
        (lru_strategy, 0.5)     # Lowest priority - fallback to LRU
    ])
    
    # Option 1: In-memory store with custom configuration
    in_memory_store = InMemoryStore(
        max_size=500,                # Max items before eviction
        eviction_policy=eviction_policy,
        enable_ttl=True,             # Enable TTL eviction
        eviction_interval=60.0,      # Run eviction every 60 seconds
        enable_memory_pressure=True, # Auto-evict under memory pressure
        memory_pressure_threshold=0.8 # 80% memory usage threshold
    )
    
    # Option 2: Redis store with custom configuration
    redis_store = RedisCacheStore(
        redis_url="redis://localhost:6379",
        namespace="semantrix_cache",
        eviction_policy=eviction_policy,
        ttl_seconds=3600,           # Default TTL for entries
        max_memory="1GB",           # Max memory to use
        max_memory_policy="allkeys-lru"  # Redis eviction policy
    )
    
    # Create a custom vector store (FAISS in this example)
    vector_store = FAISSVectorStore(
        dimension=384,              # Dimension of embeddings
        metric=DistanceMetric.COSINE,
        index_type="IVF_FLAT",      # Use IVF index for better performance
        nlist=100                   # Number of clusters for IVF
    )
    
    # Initialize Semantrix with custom configuration
    cache = Semantrix(
        cache_store=in_memory_store,  # Or redis_store
        vector_store=vector_store,
        similarity_threshold=0.85,    # Higher threshold for more precise matches
        batch_size=32,               # Batch size for vector operations
        enable_async_indexing=True,   # Process indexing in background
        log_level="INFO"             # Set logging level
    )
    
    # Example usage with the custom cache
    try:
        # Add some documents with metadata
        documents = [
            ("How to implement a neural network", "Neural networks can be implemented using...", {"topic": "ML", "difficulty": "advanced"}),
            ("Introduction to Python", "Python is a high-level programming language...", {"topic": "programming", "difficulty": "beginner"}),
            ("Advanced Python features", "Python's decorators and generators are...", {"topic": "programming", "difficulty": "intermediate"})
        ]
        
        for query, response, metadata in documents:
            await cache.set(
                query=query,
                response=response,
                metadata=metadata,
                ttl=7200  # 2 hours TTL for this specific item
            )
        
        # Perform a search with metadata filtering
        result = await cache.get(
            "What are some Python programming concepts?",
            filter_metadata={"topic": "programming"}  # Only search in programming documents
        )
        
        if result:
            print(f"Found similar content: {result}")
        
        # Get cache statistics
        stats = await cache.get_stats()
        print(f"Cache stats: {stats}")
        
    finally:
        # Cleanup (handled automatically in most cases)
        await cache.close()

# Run the example
if __name__ == "__main__":
    asyncio.run(main())
        max_size=1000,                # Max items in cache
        eviction_policy=policy,       # Our custom policy
        enable_ttl=True,              # Enable TTL eviction
        ttl_seconds=3600,             # 1 hour TTL
        eviction_interval=60.0,       # Check every minute
        enable_memory_pressure=True,  # Auto-evict under memory pressure
        memory_pressure_threshold=0.8 # 80% memory usage threshold
    )
    
    # Initialize with custom configuration
    cache = Semantrix(
        similarity_threshold=0.9,  # How similar queries should be to match
        cache_store=cache_store    # Our custom store
    )
    
    # Use the cache normally - no cleanup needed
    await cache.set("key", "value")
    value = await cache.get("key")
    print(f"Got value: {value}")

asyncio.run(main())
```
```

## License  
Semantrix uses the [SSPL-1.0 license](LICENSE). **Commercial hosting requires**:  
- Open-sourcing your service stack, OR  
- A [commercial license](mailto:licensing@semantrix.org)  

---
[📚 Documentation](--) | [💬 Discuss](--)  