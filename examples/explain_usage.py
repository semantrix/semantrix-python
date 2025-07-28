#!/usr/bin/env python3
"""
Example: Using the Explain Model
===============================

This example demonstrates how to use the new ExplainResult model
to debug cache behavior and understand why queries hit or miss the cache.
"""

from semantrix import SemantrixClient, ExplainResult, CacheMatch

def demonstrate_explain():
    """Demonstrate the explain functionality."""
    print("=== Explain Model Example ===\n")
    
    # Create a client
    client = SemantrixClient(
        similarity_threshold=0.8,
        enable_profiling=True
    )
    
    # Add some test data
    print("1. Adding test data to cache:")
    test_data = [
        ("What is artificial intelligence?", "AI is a field of computer science that focuses on creating intelligent machines."),
        ("How does machine learning work?", "Machine learning uses algorithms to learn patterns from data."),
        ("Explain neural networks", "Neural networks are computing systems inspired by biological brains."),
        ("What is deep learning?", "Deep learning is a subset of machine learning using neural networks with multiple layers."),
        ("How do computers learn?", "Computers learn through algorithms that process data and identify patterns.")
    ]
    
    for prompt, response in test_data:
        client.set(prompt, response)
        print(f"   Added: '{prompt[:40]}...'")
    
    print(f"\n2. Testing cache hits:")
    
    # Test exact match
    print("\n   Testing exact match:")
    result = client.explain("What is artificial intelligence?")
    print(f"   Query: '{result.query}'")
    print(f"   Cache hit: {result.cache_hit}")
    print(f"   Exact match: {result.exact_match}")
    print(f"   Semantic match: {result.semantic_match}")
    print(f"   Best similarity: {result.best_similarity:.3f}")
    print(f"   Total time: {result.total_time_ms:.2f}ms")
    
    # Test semantic match
    print("\n   Testing semantic match:")
    result = client.explain("Explain artificial intelligence")
    print(f"   Query: '{result.query}'")
    print(f"   Cache hit: {result.cache_hit}")
    print(f"   Exact match: {result.exact_match}")
    print(f"   Semantic match: {result.semantic_match}")
    print(f"   Best similarity: {result.best_similarity:.3f}")
    print(f"   Missed by: {result.missed_by:.3f}")
    print(f"   Total time: {result.total_time_ms:.2f}ms")
    
    # Test cache miss
    print("\n   Testing cache miss:")
    result = client.explain("What is quantum computing?")
    print(f"   Query: '{result.query}'")
    print(f"   Cache hit: {result.cache_hit}")
    print(f"   Exact match: {result.exact_match}")
    print(f"   Semantic match: {result.semantic_match}")
    print(f"   Best similarity: {result.best_similarity:.3f}")
    print(f"   Missed by: {result.missed_by:.3f}")
    print(f"   Total time: {result.total_time_ms:.2f}ms")
    
    print("\n3. Detailed explain results:")
    
    # Show detailed explain for a semantic match
    result = client.explain("Tell me about AI")
    print(f"\n   Query: '{result.query}'")
    print(f"   Threshold: {result.similarity_threshold:.3f}")
    print(f"   Status: {'HIT' if result.cache_hit else 'MISS'}")
    print(f"   Match type: {'exact' if result.exact_match else 'semantic' if result.semantic_match else 'none'}")
    
    if result.top_matches:
        print("   Top matches:")
        for i, match in enumerate(result.top_matches[:3], 1):
            print(f"     {i}. Similarity: {match.similarity:.3f}")
            print(f"        Text: '{match.text[:60]}...'")
    
    if result.resource_limited:
        print(f"   Resource warnings: {', '.join(result.resource_warnings)}")
    
    print(f"   Performance:")
    print(f"     Embedding time: {result.embedding_time_ms:.2f}ms")
    print(f"     Search time: {result.search_time_ms:.2f}ms")
    print(f"     Total time: {result.total_time_ms:.2f}ms")
    
    print("\n4. String representation:")
    print(result)
    
    print("\n5. Dictionary representation:")
    result_dict = result.to_dict()
    print(f"   Keys: {list(result_dict.keys())}")
    print(f"   Cache hit: {result_dict['cache_hit']}")
    print(f"   Best similarity: {result_dict['best_similarity']:.3f}")
    
    print("\n=== Explain example completed! ===")


def demonstrate_cache_match():
    """Demonstrate CacheMatch usage."""
    print("\n=== CacheMatch Example ===\n")
    
    # Create some cache matches
    matches = [
        CacheMatch(text="AI is a field of computer science", similarity=0.95),
        CacheMatch(text="Machine learning uses algorithms", similarity=0.87),
        CacheMatch(text="Neural networks are computing systems", similarity=0.82)
    ]
    
    print("CacheMatch objects:")
    for i, match in enumerate(matches, 1):
        print(f"  {i}. {match}")
    
    # Find best match
    best_match = max(matches, key=lambda m: m.similarity)
    print(f"\nBest match: {best_match}")
    
    print("\n=== CacheMatch example completed! ===")


def demonstrate_create_explain_result():
    """Demonstrate create_explain_result function."""
    print("\n=== create_explain_result Example ===\n")
    
    from semantrix import create_explain_result
    
    # Create some cache matches
    matches = [
        CacheMatch(text="AI is a field of computer science", similarity=0.85),
        CacheMatch(text="Machine learning uses algorithms", similarity=0.78)
    ]
    
    # Create explain result
    result = create_explain_result(
        query="What is artificial intelligence?",
        similarity_threshold=0.8,
        top_matches=matches,
        cache_hit=True,
        semantic_match=True,
        embedding_time_ms=15.2,
        search_time_ms=8.7,
        total_time_ms=23.9
    )
    
    print("Created ExplainResult:")
    print(f"  Query: {result.query}")
    print(f"  Cache hit: {result.cache_hit}")
    print(f"  Best similarity: {result.best_similarity:.3f}")
    print(f"  Total time: {result.total_time_ms:.2f}ms")
    
    print("\n=== create_explain_result example completed! ===")


if __name__ == "__main__":
    demonstrate_explain()
    demonstrate_cache_match()
    demonstrate_create_explain_result() 