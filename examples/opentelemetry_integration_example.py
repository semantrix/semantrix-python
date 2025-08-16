"""
OpenTelemetry Integration Example for Semantrix.

This example demonstrates how to integrate OpenTelemetry with Semantrix
for comprehensive observability including metrics, tracing, and logging.
"""
import asyncio
import time
from typing import List, Dict, Any

# Semantrix imports
from semantrix.integrations.opentelemetry import (
    initialize_opentelemetry,
    trace_operation,
    trace_span,
    trace_span_async,
    add_span_event,
    set_span_attribute,
    sync_metrics_to_opentelemetry,
    trace_cache_operation,
    trace_vector_store_operation,
    trace_embedding_operation,
    trace_with_reliability_features,
    shutdown_opentelemetry,
)
from semantrix.utils.metrics import (
    MetricsRegistry,
    counter,
    gauge,
    histogram,
    timer,
    collect_metrics,
)
from semantrix.utils.logging import (
    initialize_logging,
    get_logger,
    with_correlation_id,
)
from semantrix.utils.retry import retry
from semantrix.utils.circuit_breaker import circuit_breaker
from semantrix.utils.timeout import timeout


def setup_opentelemetry():
    """Initialize OpenTelemetry with various exporters."""
    print("🔧 Initializing OpenTelemetry...")
    
    # Initialize OpenTelemetry with multiple exporters
    initialize_opentelemetry(
        service_name="semantrix-example",
        service_version="1.0.0",
        environment="development",
        
        # OTLP endpoints (for Jaeger, Zipkin, etc.)
        traces_endpoint="http://localhost:4317",  # OTLP traces
        metrics_endpoint="http://localhost:4318",  # OTLP metrics
        
        # Alternative exporters
        jaeger_endpoint="http://localhost:14268/api/traces",
        zipkin_endpoint="http://localhost:9411/api/v2/spans",
        
        # Configuration
        sampling_rate=1.0,  # Sample all traces
        enable_console_export=True,  # Also export to console for debugging
        enable_auto_instrumentation=True,  # Auto-instrument common libraries
        resource_attributes={
            "service.instance.id": "example-instance-1",
            "deployment.region": "us-west-2",
        }
    )
    print("✅ OpenTelemetry initialized successfully!")


def setup_logging():
    """Initialize structured logging with correlation IDs."""
    print("📝 Setting up structured logging...")
    
    initialize_logging(
        log_level="INFO",
        log_format="json",
        log_file="semantrix_example.log",
        include_correlation_id=True
    )
    print("✅ Logging initialized successfully!")


# Example cache operations with tracing
@trace_cache_operation("get")
@retry(ValueError, max_retries=3)
async def get_from_cache(key: str) -> str:
    """Simulate getting data from cache with tracing and retry."""
    logger = get_logger("semantrix.cache")
    
    # Simulate cache miss occasionally
    if key == "missing_key":
        raise ValueError("Cache miss")
    
    logger.info(f"Cache hit for key: {key}")
    return f"cached_value_for_{key}"


@trace_cache_operation("set")
async def set_in_cache(key: str, value: str, ttl: int = 3600) -> bool:
    """Simulate setting data in cache with tracing."""
    logger = get_logger("semantrix.cache")
    
    # Simulate cache operation
    await asyncio.sleep(0.01)
    
    logger.info(f"Cache set for key: {key} with TTL: {ttl}")
    return True


# Example vector store operations with tracing
@trace_vector_store_operation("search")
@timeout(5.0, "Vector search timed out")
async def search_vectors(query: str, top_k: int = 10) -> List[Dict[str, Any]]:
    """Simulate vector search with tracing and timeout."""
    logger = get_logger("semantrix.vector_store")
    
    # Simulate vector search
    await asyncio.sleep(0.1)
    
    results = [
        {"id": f"doc_{i}", "score": 0.9 - (i * 0.1), "content": f"Result {i}"}
        for i in range(min(top_k, 5))
    ]
    
    logger.info(f"Vector search completed for query: {query}, found {len(results)} results")
    return results


@trace_vector_store_operation("add")
async def add_vectors(vectors: List[Dict[str, Any]]) -> bool:
    """Simulate adding vectors to store with tracing."""
    logger = get_logger("semantrix.vector_store")
    
    # Simulate vector addition
    await asyncio.sleep(0.05)
    
    logger.info(f"Added {len(vectors)} vectors to store")
    return True


# Example embedding operations with tracing
@trace_embedding_operation("encode")
@circuit_breaker(failure_threshold=3, recovery_timeout=30.0)
async def encode_text(text: str, model: str = "text-embedding-ada-002") -> List[float]:
    """Simulate text encoding with tracing and circuit breaker."""
    logger = get_logger("semantrix.embedding")
    
    # Simulate encoding
    await asyncio.sleep(0.02)
    
    # Simulate occasional failures
    if "error" in text.lower():
        raise ConnectionError("Embedding service unavailable")
    
    # Return mock embedding
    embedding = [0.1 + i * 0.01 for i in range(1536)]  # OpenAI ada-002 dimension
    
    logger.info(f"Encoded text with model: {model}, embedding dimension: {len(embedding)}")
    return embedding


# Example business logic with comprehensive tracing
@trace_with_reliability_features("process_document")
async def process_document(document_id: str, content: str) -> Dict[str, Any]:
    """
    Process a document with comprehensive tracing and reliability features.
    
    This function demonstrates how to use OpenTelemetry tracing with
    Semantrix reliability features for end-to-end observability.
    """
    logger = get_logger("semantrix.document_processor")
    
    # Create correlation ID for this request
    with with_correlation_id(f"doc_{document_id}") as correlation_id:
        logger.info(f"Starting document processing", extra={
            "document_id": document_id,
            "content_length": len(content)
        })
        
        # Add custom span attributes
        set_span_attribute("document.id", document_id)
        set_span_attribute("document.content_length", len(content))
        
        try:
            # Step 1: Check cache
            with trace_span("check_cache") as span:
                span.set_attribute("cache.key", f"doc_{document_id}")
                
                cached_result = await get_from_cache(f"doc_{document_id}")
                if cached_result:
                    add_span_event("cache.hit", {"cache.key": f"doc_{document_id}"})
                    return {"status": "cached", "result": cached_result}
                
                add_span_event("cache.miss", {"cache.key": f"doc_{document_id}"})
            
            # Step 2: Encode text
            with trace_span("encode_text") as span:
                span.set_attribute("embedding.model", "text-embedding-ada-002")
                
                embedding = await encode_text(content)
                span.set_attribute("embedding.dimension", len(embedding))
            
            # Step 3: Search similar documents
            with trace_span("search_similar") as span:
                span.set_attribute("search.query", content[:100])  # First 100 chars
                
                similar_docs = await search_vectors(content, top_k=5)
                span.set_attribute("search.results_count", len(similar_docs))
            
            # Step 4: Store results
            with trace_span("store_results") as span:
                result = {
                    "document_id": document_id,
                    "embedding": embedding,
                    "similar_documents": similar_docs,
                    "processed_at": time.time()
                }
                
                # Cache the result
                await set_in_cache(f"doc_{document_id}", str(result))
                
                span.set_attribute("result.size", len(str(result)))
            
            logger.info("Document processing completed successfully", extra={
                "document_id": document_id,
                "similar_docs_found": len(similar_docs)
            })
            
            return result
            
        except Exception as e:
            logger.error("Document processing failed", extra={
                "document_id": document_id,
                "error": str(e)
            }, exc_info=True)
            raise


async def demonstrate_metrics_integration():
    """Demonstrate metrics integration with OpenTelemetry."""
    print("📊 Demonstrating metrics integration...")
    
    # Create some metrics
    request_counter = counter("requests_total", "Total requests")
    error_counter = counter("errors_total", "Total errors")
    response_time = histogram("response_time_seconds", "Response time distribution")
    active_connections = gauge("active_connections", "Active connections")
    
    # Simulate some operations
    for i in range(10):
        request_counter.increment()
        
        with timer("operation_duration") as t:
            # Simulate work
            await asyncio.sleep(0.01)
            
            # Simulate occasional errors
            if i % 3 == 0:
                error_counter.increment()
        
        response_time.observe(t.stop())
        active_connections.set(i + 1)
    
    # Sync metrics to OpenTelemetry
    sync_metrics_to_opentelemetry()
    print("✅ Metrics synchronized to OpenTelemetry!")


async def demonstrate_tracing_scenarios():
    """Demonstrate various tracing scenarios."""
    print("🔍 Demonstrating tracing scenarios...")
    
    # Scenario 1: Normal processing
    print("\n📄 Scenario 1: Normal document processing")
    try:
        result = await process_document("doc_001", "This is a sample document about machine learning.")
        print(f"✅ Processing completed: {result['status'] if 'status' in result else 'processed'}")
    except Exception as e:
        print(f"❌ Processing failed: {e}")
    
    # Scenario 2: Processing with error (to test circuit breaker)
    print("\n📄 Scenario 2: Processing with error (testing circuit breaker)")
    for i in range(5):
        try:
            result = await process_document("doc_002", "This document contains error in the text.")
            print(f"✅ Processing completed: {result['status'] if 'status' in result else 'processed'}")
        except Exception as e:
            print(f"❌ Processing failed (attempt {i+1}): {e}")
    
    # Scenario 3: Cache hit scenario
    print("\n📄 Scenario 3: Cache hit scenario")
    try:
        result = await process_document("doc_001", "This is a sample document about machine learning.")
        print(f"✅ Processing completed: {result['status'] if 'status' in result else 'processed'}")
    except Exception as e:
        print(f"❌ Processing failed: {e}")


async def demonstrate_manual_tracing():
    """Demonstrate manual tracing with context managers."""
    print("🔧 Demonstrating manual tracing...")
    
    # Manual span creation
    with trace_span("manual_operation", attributes={"operation.type": "manual"}) as span:
        span.set_attribute("custom.attribute", "manual_value")
        
        # Add events
        add_span_event("operation.started", {"timestamp": time.time()})
        
        # Simulate work
        await asyncio.sleep(0.02)
        
        # Add more events
        add_span_event("operation.completed", {"duration": 0.02})
    
    # Async span creation
    async with trace_span_async("async_manual_operation", attributes={"operation.type": "async_manual"}) as span:
        span.set_attribute("async.attribute", "async_value")
        
        # Simulate async work
        await asyncio.sleep(0.01)
        
        # Add events
        add_span_event("async.operation.step", {"step": "completed"})
    
    print("✅ Manual tracing completed!")


async def demonstrate_correlation_ids():
    """Demonstrate correlation ID usage for request tracing."""
    print("🆔 Demonstrating correlation IDs...")
    
    # Process multiple documents with correlation IDs
    documents = [
        ("doc_001", "First document content"),
        ("doc_002", "Second document content"),
        ("doc_003", "Third document content"),
    ]
    
    for doc_id, content in documents:
        with with_correlation_id(f"batch_{doc_id}"):
            logger.info(f"Processing document in batch", extra={
                "document_id": doc_id,
                "batch_size": len(documents)
            })
            
            try:
                result = await process_document(doc_id, content)
                logger.info("Document processed successfully", extra={
                    "document_id": doc_id,
                    "result_size": len(str(result))
                })
            except Exception as e:
                logger.error("Document processing failed", extra={
                    "document_id": doc_id,
                    "error": str(e)
                })
    
    print("✅ Correlation ID demonstration completed!")


async def main():
    """Main function demonstrating OpenTelemetry integration."""
    print("🚀 Semantrix OpenTelemetry Integration Example")
    print("=" * 50)
    
    try:
        # Setup
        setup_logging()
        setup_opentelemetry()
        
        # Demonstrate features
        await demonstrate_metrics_integration()
        await demonstrate_tracing_scenarios()
        await demonstrate_manual_tracing()
        await demonstrate_correlation_ids()
        
        print("\n" + "=" * 50)
        print("✅ All demonstrations completed successfully!")
        print("\n📊 Check your OpenTelemetry backend (Jaeger, Zipkin, etc.) to see:")
        print("   - Distributed traces with spans and events")
        print("   - Metrics with labels and values")
        print("   - Correlation IDs linking related operations")
        print("   - Circuit breaker and retry behavior in traces")
        print("   - Performance metrics and error rates")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        raise
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        shutdown_opentelemetry()
        print("✅ Cleanup completed!")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
