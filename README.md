# Semantrix · [![SSPL License](https://img.shields.io/badge/License-SSPL--1.0-ff69b4)](LICENSE)  
**Blazing-fast semantic caching for AI applications with support for multiple backends.**  

> **Note**: This project is licensed under the [Server Side Public License (SSPL) v1.0](https://www.mongodb.com/licensing/server-side-public-license).

## Features  
- 🚀 High-performance semantic caching for AI applications
- 🔍 Multiple storage backends (In-Memory, DynamoDB, ElastiCache, Google Memorystore, and more)
- 🤖 Multiple embedding providers (OpenAI, LangChain, Ollama, Mistral, Cohere, Sentence Transformers, ONNX)
- ⚡ Async-first API for maximum performance
- 🛡️ Resource management with async context managers
- 📦 Lightweight and extensible architecture
- 🔄 Support for various eviction policies (LRU, LFU, FIFO, custom)
- 📊 Built-in monitoring and metrics

## Installation
```bash
# Core package with in-memory cache and basic embedders
pip install semantrix

# For specific backends, install optional dependencies:

# Embedding providers
pip install "semantrix[openai]"      # OpenAI embeddings
pip install "semantrix[langchain]"   # LangChain embeddings
pip install "semantrix[mistral]"     # Mistral AI embeddings
pip install "semantrix[cohere]"      # Cohere embeddings
pip install "semantrix[sentence-transformers]"  # Sentence Transformers
pip install "semantrix[onnx]"        # ONNX runtime

# Vector store backends
pip install "semantrix[faiss]"       # FAISS vector store
pip install "semantrix[chroma]"      # Chroma
pip install "semantrix[qdrant]"      # Qdrant
pip install "semantrix[pinecone]"    # Pinecone
pip install "semantrix[milvus]"      # Milvus

# Cloud backends
pip install "semantrix[dynamodb]"    # AWS DynamoDB
pip install "semantrix[elasticache]"  # AWS ElastiCache (Redis protocol)
pip install "semantrix[google-memorystore]"  # Google Cloud Memorystore

# Or install all optional dependencies
pip install "semantrix[all]"
```

## Quick Start

### Basic Usage (Synchronous)
```python
from semantrix import Semantrix

# Create a cache instance with default in-memory store
cache = Semantrix()

# Add to cache
cache.set("Explain quantum physics", "It's about...")

# Get from cache (with semantic matching)
response = cache.get("What is quantum theory?")
print(response)  # Returns semantically similar cached response
```

### Async Usage (Recommended)
```python
import asyncio
from semantrix import AsyncSemantrix

async def main():
    # Create an async cache instance
    cache = AsyncSemantrix()
    
    # Add to cache
    await cache.add("Explain quantum physics", "It's about...", ttl=3600)
    
    # Get from cache with semantic matching
    response = await cache.get("What is quantum theory?")
    print(response)  # Returns semantically similar cached response

asyncio.run(main())
```

## Embedding Providers

Semantrix supports multiple embedding providers out of the box. Here's how to use each one:

### OpenAI Embeddings
```python
from semantrix.embedding import OpenAIEmbedder
import asyncio

async def main():
    # Initialize with your API key (or set OPENAI_API_KEY environment variable)
    embedder = OpenAIEmbedder(
        model="text-embedding-3-small",  # or "text-embedding-3-large"
        api_key="your-api-key"
    )
    
    # Get embeddings
    embedding = await embedder.encode("This is a test sentence.")
    print(f"Embedding dimension: {embedder.get_dimension()}")
    print(f"Sample embedding: {embedding[:5]}...")

asyncio.run(main())
```

### LangChain Embeddings
LangChain integration allows using any LangChain-compatible embedding model:

```python
from semantrix.embedding import LangChainEmbedder
import asyncio

async def main():
    # Option 1: Auto-detect from model name
    embedder = LangChainEmbedder.from_model_name(
        model_name="all-MiniLM-L6-v2",
        batch_size=16
    )
    
    # Option 2: Use a LangChain embeddings instance directly
    # from langchain_community.embeddings import HuggingFaceEmbeddings
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # embedder = LangChainEmbedder(embeddings=embeddings)
    
    # Get embeddings
    embedding = await embedder.encode("This is a test sentence.")
    print(f"Embedding dimension: {embedder.get_dimension()}")

asyncio.run(main())
```

### Ollama Embeddings
For local models using Ollama:

```python
from semantrix.embedding import OllamaEmbedder
import asyncio

async def main():
    # Initialize with your local Ollama server
    embedder = OllamaEmbedder(
        model="nomic-embed-text",  # or any model you have pulled with Ollama
        base_url="http://localhost:11434"
    )
    
    # Get embeddings
    embedding = await embedder.encode("This is a test sentence.")
    print(f"Embedding dimension: {embedder.get_dimension()}")

asyncio.run(main())
```

### Mistral AI Embeddings
For Mistral's embedding models:

```python
from semantrix.embedding import MistralEmbedder
import asyncio

async def main():
    # Initialize with your API key (or set MISTRAL_API_KEY environment variable)
    embedder = MistralEmbedder(
        model="mistral-embed",
        api_key="your-api-key"
    )
    
    # Get embeddings
    embedding = await embedder.encode("This is a test sentence.")
    print(f"Embedding dimension: {embedder.get_dimension()}")

asyncio.run(main())
```

### Cohere Embeddings
For high-quality embeddings from Cohere's API:

```python
from semantrix.embedding import CohereEmbedder
import asyncio

async def main():
    # Initialize with your API key (or set COHERE_API_KEY environment variable)
    embedder = CohereEmbedder(
        model="embed-english-v3.0",  # or "embed-multilingual-v3.0"
        api_key="your-api-key",
        input_type="search_document",  # or "search_query" for queries
        truncate="END"  # Handle long texts: "NONE", "START", "END"
    )
    
    # Single text embedding
    embedding = await embedder.encode("This is a test sentence.")
    print(f"Embedding dimension: {embedder.get_dimension()}")
    
    # Batch processing
    texts = ["First document", "Second document"]
    embeddings = await embedder.batch_encode(texts)
    print(f"Processed {len(embeddings)} documents")

asyncio.run(main())
```

### Sentence Transformers
For local models using Sentence Transformers:

```python
from semantrix.embedding import SentenceTransformerEmbedder
import asyncio

async def main():
    # Initialize with any Sentence Transformers model
    embedder = SentenceTransformerEmbedder(
        model_name="all-MiniLM-L6-v2",
        device="cpu"  # or "cuda" for GPU
    )
    
    # Get embeddings
    embedding = await embedder.encode("This is a test sentence.")
    print(f"Embedding dimension: {embedder.get_dimension()}")

asyncio.run(main())
```

### ONNX Models
For optimized ONNX models:

```python
from semantrix.embedding import OnnxEmbedder
import asyncio

async def main():
    # Initialize with path to ONNX model
    embedder = OnnxEmbedder(
        model_path="path/to/model.onnx",
        provider="CPUExecutionProvider"  # or "CUDAExecutionProvider"
    )
    
    # Get embeddings
    embedding = await embedder.encode("This is a test sentence.")
    print(f"Embedding dimension: {embedder.get_dimension()}")

asyncio.run(main())
```
## Advanced Usage
```python
from semantrix import Semantrix

# Create a cache instance with default in-memory store
cache = Semantrix()

# Add to cache
cache.set("Explain quantum physics", "It's about...")

# Get from cache (with semantic matching)
response = cache.get("What is quantum theory?")
print(response)  # Returns semantically similar cached response
```

### Async Usage (Recommended)
```python
import asyncio
from semantrix import AsyncSemantrix

async def main():
    # Create an async cache instance
    cache = AsyncSemantrix()
    
    # Add to cache
    await cache.add("Explain quantum physics", "It's about...", ttl=3600)
    
    # Get from cache with semantic matching
    response = await cache.get("What is quantum theory?")
    print(response)  # Returns semantically similar cached response

asyncio.run(main())
```

### Cache Stores

Semantrix supports various cache store backends:

#### In-Memory (Default)
```python
from semantrix import AsyncSemantrix

# Uses in-memory store by default
cache = AsyncSemantrix()
```

#### AWS DynamoDB
```python
from semantrix.cache_store.stores import DynamoDBCacheStore

# Create a DynamoDB cache store
store = DynamoDBCacheStore(
    table_name="semantrix-cache",
    region_name="us-west-2"
)

cache = AsyncSemantrix(store=store)
```

#### AWS ElastiCache (Redis)
```python
from semantrix.cache_store.stores import ElastiCacheStore

store = ElastiCacheStore(
    endpoint="my-cache.xxxxx.ng.0001.aps1.cache.amazonaws.com:6379",
    ssl=True
)

cache = AsyncSemantrix(store=store)
```

#### Google Memorystore
```python
from semantrix.cache_store.stores import GoogleMemorystoreCacheStore

store = GoogleMemorystoreCacheStore(
    project_id="your-project-id",
    region="us-central1",
    instance_id="semantrix-cache"
)

cache = AsyncSemantrix(store=store)
```

## Advanced Features

### Custom Eviction Policies
```python
from semantrix.cache_store.policy import LRUEvictionPolicy, LFUEvictionPolicy

# Create a cache with LRU eviction
lru_cache = AsyncSemantrix(
    max_size=1000,
    eviction_policy=LRUEvictionPolicy()
)

# Or with LFU eviction
lfu_cache = AsyncSemantrix(
    max_size=1000,
    eviction_policy=LFUEvictionPolicy()
)
```

### Monitoring and Metrics
```python
# Get cache statistics
stats = await cache.get_stats()
print(f"Cache hits: {stats['hits']}")
print(f"Cache misses: {stats['misses']}")
print(f"Current size: {await cache.size()}")
```

## Documentation

For detailed documentation, see [docs/cache_store_guide.md](docs/cache_store_guide.md).

## Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the [Server Side Public License (SSPL) v1.0](https://www.mongodb.com/licensing/server-side-public-license). See the [LICENSE](LICENSE) file for details.

### About SSPL

The SSPL is a copyleft license that requires that anyone who offers the licensed software as a service must make the service's source code available under this license. For more information, please see the [SSPL FAQ](https://www.mongodb.com/licensing/server-side-public-license/faq).
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