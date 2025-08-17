#!/usr/bin/env python3
"""
Metrics Integration Demo for Semantrix

This example demonstrates how metrics are automatically collected
and can be exported to external monitoring systems.
"""

import asyncio
import time
from semantrix import Semantrix
from semantrix.utils.metrics import (
    collect_metrics, get_metrics_registry,
    REQUEST_COUNTER, CACHE_HIT_COUNTER, CACHE_MISS_COUNTER,
    SEMANTIC_SEARCH_COUNTER, EMBEDDING_GENERATION_COUNTER,
    REDIS_OPERATIONS_COUNTER, FAISS_OPERATIONS_COUNTER,
    SENTENCE_TRANSFORMER_OPERATIONS_COUNTER,
    CACHE_HIT_RATE_GAUGE, CACHE_EFFECTIVENESS_SCORE_GAUGE,
    USER_SATISFACTION_SCORE_GAUGE, MEMORY_USAGE_GAUGE, CPU_USAGE_GAUGE
)


async def demonstrate_metrics_collection():
    """Demonstrate automatic metrics collection during cache operations."""
    print("=== Semantrix Metrics Demo ===")
    print("Initializing Semantrix cache...")
    
    # Initialize Semantrix with metrics enabled
    cache = Semantrix(
        similarity_threshold=0.8,
        enable_logging=True
    )
    
    await cache.initialize()
    
    print("\n1. Performing cache operations to generate metrics...")
    
    # Perform some cache operations
    await cache.set("What is machine learning?", "Machine learning is a subset of AI...")
    await cache.set("Explain neural networks", "Neural networks are computational models...")
    await cache.set("What is deep learning?", "Deep learning uses multiple layers...")
    
    # Try to get cached items
    result1 = await cache.get("What is machine learning?")
    result2 = await cache.get("Explain neural networks")
    
    # Try to get non-cached items (will trigger semantic search)
    result3 = await cache.get("Tell me about ML algorithms")
    result4 = await cache.get("How do neural nets work?")
    
    # Try to get completely different items
    result5 = await cache.get("What is the weather today?")
    
    print(f"Cache hits: {result1 is not None}, {result2 is not None}")
    print(f"Semantic search results: {result3 is not None}, {result4 is not None}")
    print(f"Cache miss: {result5 is None}")
    
    # Wait a moment for background metrics sync
    print("\n2. Waiting for metrics to be collected...")
    await asyncio.sleep(2)
    
    # Collect and display current metrics
    print("\n3. Current Metrics:")
    metrics = collect_metrics()
    
    print(f"Total Requests: {metrics['counters']['semantrix_requests_total']['value']}")
    print(f"Cache Hits: {metrics['counters']['semantrix_cache_hits_total']['value']}")
    print(f"Cache Misses: {metrics['counters']['semantrix_cache_misses_total']['value']}")
    print(f"Semantic Searches: {metrics['counters']['semantrix_semantic_searches_total']['value']}")
    print(f"Embeddings Generated: {metrics['counters']['semantrix_embeddings_generated_total']['value']}")
    
    # Show new cache store metrics
    if 'semantrix_redis_operations_total' in metrics['counters']:
        print(f"Redis Operations: {metrics['counters']['semantrix_redis_operations_total']['value']}")
    
    # Show new vector store metrics
    if 'semantrix_faiss_operations_total' in metrics['counters']:
        print(f"FAISS Operations: {metrics['counters']['semantrix_faiss_operations_total']['value']}")
    
    # Show new embedding metrics
    if 'semantrix_sentence_transformer_operations_total' in metrics['counters']:
        print(f"Sentence Transformer Operations: {metrics['counters']['semantrix_sentence_transformer_operations_total']['value']}")
    
    # Display histogram data
    print("\n4. Performance Metrics:")
    request_duration = metrics['histograms']['semantrix_request_duration_seconds']['summary']
    print(f"Request Duration - Count: {request_duration['count']}")
    print(f"Request Duration - Mean: {request_duration['mean']:.4f}s")
    print(f"Request Duration - Min: {request_duration['min']:.4f}s")
    print(f"Request Duration - Max: {request_duration['max']:.4f}s")
    
    if 'semantrix_semantic_search_duration_seconds' in metrics['histograms']:
        semantic_duration = metrics['histograms']['semantrix_semantic_search_duration_seconds']['summary']
        print(f"Semantic Search Duration - Mean: {semantic_duration['mean']:.4f}s")
    
    if 'semantrix_embedding_duration_seconds' in metrics['histograms']:
        embedding_duration = metrics['histograms']['semantrix_embedding_duration_seconds']['summary']
        print(f"Embedding Duration - Mean: {embedding_duration['mean']:.4f}s")
    
    # Show business intelligence metrics
    print("\n5. Business Intelligence Metrics:")
    if 'semantrix_cache_hit_rate_percentage' in metrics['gauges']:
        hit_rate = metrics['gauges']['semantrix_cache_hit_rate_percentage']['value']
        print(f"Cache Hit Rate: {hit_rate:.2f}%")
    
    if 'semantrix_cache_effectiveness_score' in metrics['gauges']:
        effectiveness = metrics['gauges']['semantrix_cache_effectiveness_score']['value']
        print(f"Cache Effectiveness Score: {effectiveness:.3f}")
    
    if 'semantrix_user_satisfaction_score' in metrics['gauges']:
        satisfaction = metrics['gauges']['semantrix_user_satisfaction_score']['value']
        print(f"User Satisfaction Score: {satisfaction:.3f}")
    
    # Show resource utilization metrics
    print("\n6. Resource Utilization Metrics:")
    if 'semantrix_memory_usage_bytes' in metrics['gauges']:
        memory_mb = metrics['gauges']['semantrix_memory_usage_bytes']['value'] / (1024 * 1024)
        print(f"Memory Usage: {memory_mb:.2f} MB")
    
    if 'semantrix_cpu_usage_percentage' in metrics['gauges']:
        cpu_usage = metrics['gauges']['semantrix_cpu_usage_percentage']['value']
        print(f"CPU Usage: {cpu_usage:.2f}%")
    
    # Demonstrate error tracking
    print("\n7. Testing error tracking...")
    try:
        await cache.get("")  # This should trigger a validation error
    except Exception as e:
        print(f"Expected error caught: {type(e).__name__}")
    
    # Wait for error metrics to be recorded
    await asyncio.sleep(1)
    
    # Check error metrics
    updated_metrics = collect_metrics()
    print(f"Total Errors: {updated_metrics['counters']['semantrix_errors_total']['value']}")
    
    print("\n6. Metrics Export Ready for External Systems:")
    print("These metrics can be exported to:")
    print("- Prometheus (via OpenTelemetry)")
    print("- Datadog (via OpenTelemetry)")
    print("- Jaeger (via OpenTelemetry)")
    print("- Any OpenTelemetry-compatible system")
    
    # Demonstrate manual metrics sync (if OpenTelemetry is available)
    try:
        from semantrix.integrations.opentelemetry import sync_metrics_to_opentelemetry
        print("\n7. Syncing metrics to OpenTelemetry...")
        sync_metrics_to_opentelemetry()
        print("Metrics synced successfully!")
    except ImportError:
        print("\n7. OpenTelemetry not available - metrics ready for manual export")
    
    await cache.shutdown()
    print("\n=== Demo Complete ===")


def demonstrate_manual_metrics():
    """Demonstrate manual metrics usage."""
    print("\n=== Manual Metrics Usage ===")
    
    # Get the metrics registry
    registry = get_metrics_registry()
    
    # Create custom metrics
    custom_counter = registry.counter("demo.custom_operations", "Custom operations counter")
    custom_gauge = registry.gauge("demo.active_sessions", "Number of active sessions")
    custom_histogram = registry.histogram("demo.processing_time", "Processing time distribution")
    
    # Use the metrics
    custom_counter.increment()
    custom_gauge.set(5)
    
    with custom_histogram.time() as timer:
        time.sleep(0.1)  # Simulate some processing
    
    # Collect and display
    metrics = collect_metrics()
    print(f"Custom Operations: {metrics['counters']['demo.custom_operations']['value']}")
    print(f"Active Sessions: {metrics['gauges']['demo.active_sessions']['value']}")
    print(f"Processing Time - Count: {metrics['histograms']['demo.processing_time']['summary']['count']}")


async def main():
    """Main demonstration function."""
    print("Semantrix Metrics Integration Demo")
    print("=" * 50)
    
    # Demonstrate automatic metrics collection
    await demonstrate_metrics_collection()
    
    # Demonstrate manual metrics usage
    demonstrate_manual_metrics()
    
    print("\nKey Benefits:")
    print("✅ Automatic metrics collection for all cache operations")
    print("✅ Performance tracking with histograms")
    print("✅ Error rate monitoring")
    print("✅ Cache hit/miss ratio tracking")
    print("✅ Semantic search performance monitoring")
    print("✅ Embedding generation tracking")
    print("✅ Ready for external monitoring systems")


if __name__ == "__main__":
    asyncio.run(main())
