"""
Embedding Examples for Semantrix

This file provides examples of how to use the various embedding providers
available in the Semantrix library.
"""

import asyncio
import os
from typing import List

import numpy as np

from semantrix.embedding import (
    BaseEmbedder,
    SentenceTransformerEmbedder,
    OnnxEmbedder,
    OpenAIEmbedder,
    LangChainEmbedder,
    OllamaEmbedder,
    MistralEmbedder,
    CohereEmbedder,
)


async def example_sentence_transformer():
    """Example using SentenceTransformer embedder."""
    print("\n=== SentenceTransformer Embedder ===")
    
    # Initialize the embedder with a model name
    embedder = SentenceTransformerEmbedder(
        model_name="all-MiniLM-L6-v2",  # Popular lightweight model
        device="cpu"  # Use "cuda" if you have a GPU
    )
    
    # Single text embedding
    embedding = await embedder.encode("This is a test sentence.")
    print(f"Single embedding shape: {embedding.shape}")
    
    # Batch embedding
    texts = ["First text", "Second text", "Third text"]
    embeddings = await embedder.batch_encode(texts)
    print(f"Batch embeddings length: {len(embeddings)}")
    print(f"First embedding shape: {embeddings[0].shape}")
    
    # Get the dimension
    print(f"Embedding dimension: {embedder.get_dimension()}")


async def example_onnx():
    """Example using ONNX embedder."""
    print("\n=== ONNX Embedder ===")
    
    # Initialize the embedder with a model path
    # Note: You need to have the ONNX model file available
    model_path = "path/to/your/model.onnx"
    
    if not os.path.exists(model_path):
        print(f"ONNX model not found at {model_path}. Skipping ONNX example.")
        return
        
    embedder = OnnxEmbedder(
        model_path=model_path,
        provider="CPUExecutionProvider"  # Can be "CUDAExecutionProvider" for GPU
    )
    
    # Single text embedding
    embedding = await embedder.encode("This is a test sentence.")
    print(f"Single embedding shape: {embedding.shape}")
    
    # Get the dimension
    print(f"Embedding dimension: {embedder.get_dimension()}")


async def example_openai():
    """Example using OpenAI embedder."""
    print("\n=== OpenAI Embedder ===")
    
    # Get API key from environment variable or replace with your key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY environment variable not set. Skipping OpenAI example.")
        return
    
    # Initialize the embedder
    embedder = OpenAIEmbedder(
        model="text-embedding-3-small",  # or "text-embedding-3-large"
        api_key=api_key,
        timeout=30.0  # Request timeout in seconds
    )
    
    # Single text embedding
    embedding = await embedder.encode("This is a test sentence.")
    print(f"Single embedding shape: {embedding.shape}")
    
    # Batch embedding
    texts = ["First text", "Second text", "Third text"]
    embeddings = await embedder.batch_encode(texts)
    print(f"Batch embeddings length: {len(embeddings)}")
    
    # Get the dimension
    print(f"Embedding dimension: {embedder.get_dimension()}")


async def example_langchain():
    """Example using LangChain embedder."""
    print("\n=== LangChain Embedder ===")
    
    try:
        # Option 1: Create from a model name (auto-detects the right embedder)
        embedder = LangChainEmbedder.from_model_name(
            model_name="all-MiniLM-L6-v2",  # HuggingFace model
            batch_size=16
        )
        
        # Option 2: Create with a LangChain embeddings instance directly
        # from langchain_community.embeddings import HuggingFaceEmbeddings
        # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # embedder = LangChainEmbedder(embeddings=embeddings)
        
        # Single text embedding
        embedding = await embedder.encode("This is a test sentence.")
        print(f"Single embedding shape: {embedding.shape}")
        
        # Batch embedding
        texts = ["First text", "Second text", "Third text"]
        embeddings = await embedder.batch_encode(texts)
        print(f"Batch embeddings length: {len(embeddings)}")
        
        # Get the dimension
        print(f"Embedding dimension: {embedder.get_dimension()}")
        
    except ImportError as e:
        print(f"LangChain not installed. Install with: pip install langchain-core langchain-community\nError: {e}")


async def example_ollama():
    """Example using Ollama embedder."""
    print("\n=== Ollama Embedder ===")
    
    # Initialize the embedder
    embedder = OllamaEmbedder(
        model="nomic-embed-text",  # or any other model supported by Ollama
        base_url="http://localhost:11434",  # Default Ollama URL
        timeout=60.0  # Increase timeout for local models
    )
    
    try:
        # Single text embedding
        embedding = await embedder.encode("This is a test sentence.")
        print(f"Single embedding shape: {embedding.shape}")
        
        # Batch embedding
        texts = ["First text", "Second text", "Third text"]
        embeddings = await embedder.batch_encode(texts)
        print(f"Batch embeddings length: {len(embeddings)}")
        
        # Get the dimension
        print(f"Embedding dimension: {embedder.get_dimension()}")
        
    except Exception as e:
        print(f"Failed to connect to Ollama. Make sure Ollama is running.\nError: {e}")


async def example_mistral():
    """Example using Mistral embedder."""
    print("\n=== Mistral Embedder ===")
    
    # Get API key from environment variable or replace with your key
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("MISTRAL_API_KEY environment variable not set. Skipping Mistral example.")
        return
    
    try:
        # Initialize the embedder
        embedder = MistralEmbedder(
            model="mistral-embed",  # Mistral's embedding model
            api_key=api_key,
            timeout=30.0
        )
        
        # Single text embedding
        embedding = await embedder.encode("This is a test sentence.")
        print(f"Single embedding shape: {embedding.shape}")
        
        # Batch embedding
        texts = ["First text", "Second text", "Third text"]
        embeddings = await embedder.batch_encode(texts)
        print(f"Batch embeddings length: {len(embeddings)}")
        
        # Get the dimension
        print(f"Embedding dimension: {embedder.get_dimension()}")
        
    except ImportError as e:
        print(f"Mistral AI client not installed. Install with: pip install mistralai\nError: {e}")
    except Exception as e:
        print(f"Mistral API error: {e}")


async def example_cohere():
    """Example using Cohere embedder."""
    print("\n=== Cohere Embedder Example ===")
    
    # Get API key from environment variable
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        print("COHERE_API_KEY environment variable not set. Skipping Cohere example.")
        return None
    
    try:
        # Initialize the Cohere embedder
        embedder = CohereEmbedder(
            model="embed-english-v3.0",  # or "embed-multilingual-v3.0" for multilingual
            api_key=api_key,
            input_type="search_document",  # or "search_query" for query embeddings
            truncate="END"  # How to handle long texts: "NONE", "START", "END"
        )
        
        # Single text embedding
        text = "This is a test sentence for Cohere embeddings."
        embedding = await embedder.encode(text)
        print(f"Embedding dimension: {len(embedding)}")  # Should be 1024 for v3 models
        print(f"First 10 values: {embedding[:10]}")
        
        # Batch processing
        texts = [
            "First document text",
            "Second document with more details",
            "Third document in the batch"
        ]
        
        print("\nBatch processing:")
        embeddings = await embedder.batch_encode(texts)
        for i, emb in enumerate(embeddings):
            print(f"Text {i+1} embedding dimension: {len(emb)}")
        
        # Get the model's dimension
        print(f"\nModel dimension: {embedder.get_dimension()}")
        
        return embedding
    except Exception as e:
        print(f"Error with Cohere embedder: {str(e)}")
        return None


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors or matrices."""
    return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b, axis=1))


async def compare_embeddings():
    """Compare embeddings from different providers."""
    print("\n=== Comparing Embeddings from Different Providers ===")
    
    # Initialize embedders
    embedders = [
        ("SentenceTransformer", SentenceTransformerEmbedder()),
        ("ONNX", OnnxEmbedder()),
        ("OpenAI", OpenAIEmbedder() if os.getenv("OPENAI_API_KEY") else None),
        ("LangChain", LangChainEmbedder()),
        ("Ollama", OllamaEmbedder()),
        ("Mistral", MistralEmbedder() if os.getenv("MISTRAL_API_KEY") else None),
        ("Cohere", CohereEmbedder() if os.getenv("COHERE_API_KEY") else None),
    ]
    
    # Filter out None embedders (those without required API keys)
    embedders = [(name, e) for name, e in embedders if e is not None]
    
    # Text to embed
    text = "This is a test sentence for comparing embeddings."
    
    # Get embeddings from all providers
    embeddings = {}
    for name, embedder in embedders:
        try:
            embedding = await embedder.encode(text)
            embeddings[name] = embedding
            print(f"{name}: Success (dimension: {len(embedding)})")
        except Exception as e:
            print(f"{name}: Error - {str(e)}")
    
    # Compare embeddings (simple cosine similarity)
    if len(embeddings) > 1:
        print("\nPairwise cosine similarities:")
        names = list(embeddings.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                name1, name2 = names[i], names[j]
                sim = cosine_similarity(
                    embeddings[name1].reshape(1, -1),
                    embeddings[name2].reshape(1, -1)
                )[0][0]
                print(f"{name1} <-> {name2}: {sim:.4f}")
    
    return embeddings


async def main():
    """Run all examples."""
    await example_sentence_transformer()
    await example_onnx()
    await example_openai()
    await example_langchain()
    await example_ollama()
    await example_mistral()
    await example_cohere()
    await compare_embeddings()


if __name__ == "__main__":
    asyncio.run(main())
