"""
Example demonstrating different vector store implementations in Semantrix.

This example shows how to use various vector store backends with Semantrix.
Each store has different installation requirements - make sure to install the
required packages for the stores you want to use.
"""

import asyncio
import numpy as np
from semantrix.vector_store.base import DistanceMetric
from semantrix.vector_store.stores import (
    FAISSVectorStore,
    ChromaVectorStore,
    PineconeVectorStore,
    QdrantVectorStore,
    RedisVectorStore,
    PgVectorStore,
    MilvusVectorStore
)

# Sample data
documents = [
    "The quick brown fox jumps over the lazy dog",
    "A fast brown fox leaps over a sleeping dog",
    "The lazy dog sleeps under the tree",
    "The quick fox is very agile"
]

# Sample embeddings (in a real application, use a proper embedding model)
embeddings = np.random.randn(4, 384).astype(np.float32)  # 384-dim vectors
metadata = [
    {"source": "example", "id": i, "text": doc} 
    for i, doc in enumerate(documents)
]

def print_results(results, store_name):
    """Helper function to print search results."""
    print(f"\nResults from {store_name}:")
    for i, result in enumerate(results):
        print(f"{i+1}. ID: {result['id']}")
        print(f"   Score: {result['score']:.4f}")
        print(f"   Text: {result['document'][:100]}...")
        print(f"   Metadata: {result['metadata']}")

async def demo_faiss():
    """Demo using FAISS in-memory vector store."""
    print("\n=== FAISS Vector Store ===")
    
    # Initialize FAISS store (in-memory)
    store = FAISSVectorStore(
        dimension=384,
        metric=DistanceMetric.COSINE,
        persist_path="./faiss_index"  # Optional: save to disk
    )
    
    # Add documents
    await store.add(
        vectors=embeddings,
        documents=documents,
        metadatas=metadata
    )
    
    # Search
    query_embedding = np.random.randn(384).astype(np.float32)
    results = await store.search(query_embedding, k=2)
    print_results(results, "FAISS")
    
    # Cleanup (optional)
    await store.close()

async def demo_chroma():
    """Demo using Chroma vector store."""
    print("\n=== Chroma Vector Store ===")
    
    # Initialize Chroma (persists to disk by default)
    store = ChromaVectorStore(
        dimension=384,
        metric=DistanceMetric.COSINE,
        persist_directory="./chroma_db"
    )
    
    # Add documents
    await store.add(
        vectors=embeddings,
        documents=documents,
        metadatas=metadata
    )
    
    # Search with metadata filter
    query_embedding = np.random.randn(384).astype(np.float32)
    results = await store.search(
        query_embedding,
        k=2,
        filter={"source": {"$eq": "example"}}
    )
    print_results(results, "Chroma")
    
    # Cleanup
    await store.close()

async def demo_pgvector():
    """Demo using pgvector with PostgreSQL."""
    print("\n=== pgvector Vector Store ===")
    
    # Initialize pgvector store
    store = PgVectorStore(
        dimension=384,
        metric=DistanceMetric.COSINE,
        connection_string="postgresql://user:password@localhost:5432/vector_db",
        table_name="documents"
    )
    
    # Add documents
    await store.add(
        vectors=embeddings,
        documents=documents,
        metadatas=metadata
    )
    
    # Search with complex filter
    query_embedding = np.random.randn(384).astype(np.float32)
    results = await store.search(
        query_embedding,
        k=2,
        filter={
            "source": "example",
            "id": {"$gt": 0}  # Only IDs greater than 0
        }
    )
    print_results(results, "pgvector")
    
    # Cleanup
    await store.close()

async def demo_milvus():
    """Demo using Milvus vector database."""
    print("\n=== Milvus Vector Store ===")
    
    # Initialize Milvus store
    store = MilvusVectorStore(
        dimension=384,
        metric=DistanceMetric.COSINE,
        uri="http://localhost:19530",
        collection_name="documents"
    )
    
    # Add documents
    await store.add(
        vectors=embeddings,
        documents=documents,
        metadatas=metadata
    )
    
    # Search with metadata filter
    query_embedding = np.random.randn(384).astype(np.float32)
    results = await store.search(
        query_embedding,
        k=2,
        filter={"source": "example"}
    )
    print_results(results, "Milvus")
    
    # Cleanup
    await store.close()

async def main():
    """Run all vector store demos."""
    try:
        # Run FAISS demo (no external dependencies needed)
        await demo_faiss()
        
        # Uncomment the demos you want to run (make sure to install dependencies first)
        # await demo_chroma()
        # await demo_pgvector()
        # await demo_milvus()
        
    except Exception as e:
        print(f"Error: {e}")
        if "No module named" in str(e):
            print("\nNote: Some demos require additional packages. "
                  "Install them with:")
            print("pip install chromadb pymilvus psycopg2-binary")

if __name__ == "__main__":
    asyncio.run(main())
