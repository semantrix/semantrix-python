# Semantrix Reliability Guide

This document provides comprehensive guidance on the reliability features implemented in Semantrix, including error handling, retry mechanisms, circuit breakers, timeouts, validation, logging, and monitoring.

## Table of Contents

1. [Overview](#overview)
2. [Error Handling & Recovery](#error-handling--recovery)
3. [Testing & Validation](#testing--validation)
4. [Monitoring & Observability](#monitoring--observability)
5. [OpenTelemetry Integration](#opentelemetry-integration)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

## Overview

Semantrix has been enhanced with comprehensive reliability features to ensure robust operation in production environments. These improvements include:

- **Error Handling & Recovery**: Custom exception hierarchy, retry mechanisms, circuit breakers, and timeouts
- **Testing & Validation**: Comprehensive input validation, unit tests, and integration tests
- **Monitoring & Observability**: Structured logging, metrics collection, and correlation IDs
- **OpenTelemetry Integration**: Distributed tracing, metrics bridging, and external observability
- **Documentation**: Complete API documentation and operational runbooks

## Error Handling & Recovery

### Exception Hierarchy

Semantrix uses a structured exception hierarchy for better error handling:

```python
from semantrix.exceptions import (
    SemantrixError,           # Base exception
    ConfigurationError,       # Configuration issues
    CacheStoreError,          # Cache store problems
    VectorStoreError,         # Vector store issues
    ValidationError,          # Input validation failures
    OperationError,           # General operation failures
    RetryError,              # Retry mechanism failures
    TimeoutError,            # Operation timeouts
)
```

### Retry Mechanisms

The retry system provides exponential backoff with jitter to handle transient failures:

```python
from semantrix.utils.retry import retry

@retry(
    exceptions=(ValueError, ConnectionError),
    max_retries=3,
    initial_delay=1.0,
    max_delay=30.0,
    backoff_factor=2.0,
    jitter=0.1
)
async def unreliable_operation():
    # Your operation here
    pass
```

**Parameters:**
- `exceptions`: Exception types to retry on
- `max_retries`: Maximum number of retry attempts
- `initial_delay`: Initial delay between retries (seconds)
- `max_delay`: Maximum delay between retries (seconds)
- `backoff_factor`: Multiplier for delay between retries
- `jitter`: Random jitter factor to avoid thundering herd

### Circuit Breakers

Circuit breakers prevent cascading failures by temporarily stopping operations that are likely to fail:

```python
from semantrix.utils.circuit_breaker import circuit_breaker

@circuit_breaker(
    failure_threshold=5,
    recovery_timeout=60.0,
    expected_exception=ConnectionError
)
async def external_service_call():
    # Your external service call here
    pass
```

**Circuit States:**
- **CLOSED**: Normal operation, calls pass through
- **OPEN**: Circuit is open, calls fail fast
- **HALF_OPEN**: Testing if service is back to normal

### Timeouts

Timeout mechanisms prevent operations from hanging indefinitely:

```python
from semantrix.utils.timeout import timeout, with_timeout

# Using decorator
@timeout(30.0, "Operation timed out")
async def long_running_operation():
    # Your operation here
    pass

# Using context manager
async with with_timeout(30.0) as ctx:
    result = await ctx.run_async(some_async_operation())
```

## Testing & Validation

### Input Validation

Comprehensive validation functions ensure data integrity:

```python
from semantrix.utils.validation import (
    validate_prompt,
    validate_response,
    validate_vector,
    validate_metadata,
    validate_cache_key,
    validate_ttl,
    validate_batch_size,
    validate_embedding_model,
    validate_similarity_threshold,
    validate_top_k
)

# Validate inputs
validate_prompt("Your prompt here")
validate_vector([1.0, 2.0, 3.0], expected_dimension=3)
validate_metadata({"key": "value"})
```

### Validator Classes

Type-safe validation with composable validators:

```python
from semantrix.utils.validation import Validator, non_empty_string, positive_integer

# Custom validator
custom_validator = Validator[str](lambda s: None if len(s) > 5 else ValidationError("Too short"))

# Combined validators
combined = non_empty_string | custom_validator
result = combined("test string")
```

### Testing Best Practices

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions
3. **Property-Based Testing**: Test edge cases and invariants
4. **Performance Tests**: Ensure acceptable performance under load

```python
import pytest
from semantrix.utils.retry import retry

def test_retry_mechanism():
    call_count = 0
    
    @retry(ValueError, max_retries=2)
    def failing_function():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Temporary failure")
        return "success"
    
    result = failing_function()
    assert result == "success"
    assert call_count == 3
```

## Monitoring & Observability

### Structured Logging

Semantrix uses structured JSON logging with correlation IDs:

```python
from semantrix.utils.logging import (
    initialize_logging,
    get_logger,
    set_correlation_id,
    with_correlation_id
)

# Initialize logging
initialize_logging(
    log_level="INFO",
    log_format="json",
    log_file="semantrix.log",
    include_correlation_id=True
)

# Get logger
logger = get_logger("semantrix.cache")

# Use correlation IDs for tracing
with with_correlation_id("request-123") as correlation_id:
    logger.info("Processing request", extra={"user_id": "user-456"})
```

### Metrics Collection

Comprehensive metrics for monitoring system health:

```python
from semantrix.utils.metrics import (
    counter,
    gauge,
    histogram,
    timer,
    collect_metrics
)

# Define metrics
request_counter = counter("requests_total", "Total requests")
error_counter = counter("errors_total", "Total errors")
response_time = histogram("response_time_seconds", "Response time distribution")
active_connections = gauge("active_connections", "Active connections")

# Use metrics
request_counter.increment()
error_counter.increment()

with timer("operation_duration") as t:
    # Your operation here
    pass

# Collect all metrics
metrics = collect_metrics()
```

### Metrics Types

1. **Counters**: Only increase (requests, errors, cache hits)
2. **Gauges**: Can go up and down (active connections, memory usage)
3. **Histograms**: Track value distributions (response times, request sizes)
4. **Timers**: Measure operation durations

## OpenTelemetry Integration

Semantrix provides comprehensive OpenTelemetry integration for distributed tracing, metrics export, and external observability systems.

### Overview

The OpenTelemetry integration enables:
- **Distributed Tracing**: Track requests across service boundaries
- **Metrics Bridging**: Export Semantrix metrics to OpenTelemetry format
- **External Observability**: Integration with Jaeger, Zipkin, Prometheus, etc.
- **Automatic Instrumentation**: Built-in tracing for reliability features

### Initialization

```python
from semantrix.integrations.opentelemetry import initialize_opentelemetry

# Basic initialization
initialize_opentelemetry(
    service_name="semantrix-service",
    service_version="1.0.0",
    environment="production",
    enable_console_export=True
)

# Advanced initialization with external systems
initialize_opentelemetry(
    service_name="semantrix-service",
    service_version="1.0.0",
    environment="production",
    traces_endpoint="http://localhost:4317",  # OTLP traces
    metrics_endpoint="http://localhost:4318",  # OTLP metrics
    jaeger_endpoint="http://localhost:14268/api/traces",
    zipkin_endpoint="http://localhost:9411/api/v2/spans",
    sampling_rate=0.1,  # Sample 10% of traces
    enable_auto_instrumentation=True,
    resource_attributes={
        "service.instance.id": "instance-1",
        "deployment.region": "us-west-2",
    }
)
```

### Tracing Operations

#### Automatic Tracing Decorators

```python
from semantrix.integrations.opentelemetry import (
    trace_operation,
    trace_cache_operation,
    trace_vector_store_operation,
    trace_embedding_operation,
)

# General operation tracing
@trace_operation("process_document", attributes={"operation.type": "document_processing"})
async def process_document(content: str):
    # Your operation here
    pass

# Cache operation tracing
@trace_cache_operation("get")
async def get_from_cache(key: str):
    # Cache operation here
    pass

# Vector store operation tracing
@trace_vector_store_operation("search")
async def search_vectors(query: str):
    # Vector search here
    pass

# Embedding operation tracing
@trace_embedding_operation("encode")
async def encode_text(text: str):
    # Text encoding here
    pass
```

#### Manual Tracing

```python
from semantrix.integrations.opentelemetry import (
    trace_span,
    trace_span_async,
    add_span_event,
    set_span_attribute,
)

# Synchronous span
with trace_span("manual_operation", attributes={"operation.type": "manual"}) as span:
    span.set_attribute("custom.attribute", "value")
    add_span_event("operation.started", {"timestamp": time.time()})
    # Your operation here
    add_span_event("operation.completed", {"duration": 0.1})

# Asynchronous span
async with trace_span_async("async_operation") as span:
    span.set_attribute("async.attribute", "value")
    # Your async operation here
    await some_async_operation()
```

### Metrics Bridging

Automatically bridge Semantrix metrics to OpenTelemetry format:

```python
from semantrix.integrations.opentelemetry import sync_metrics_to_opentelemetry
from semantrix.utils.metrics import get_metrics_registry

# Create and use metrics
registry = get_metrics_registry()
counter = registry.counter("requests_total", "Total requests")
counter.increment()

# Sync to OpenTelemetry
sync_metrics_to_opentelemetry(registry)
```

### Integration with Reliability Features

OpenTelemetry integrates seamlessly with Semantrix reliability features:

```python
from semantrix.integrations.opentelemetry import trace_with_reliability_features
from semantrix.utils.retry import retry
from semantrix.utils.circuit_breaker import circuit_breaker
from semantrix.utils.timeout import timeout

@trace_with_reliability_features("external_api_call")
@retry(ConnectionError, max_retries=3)
@circuit_breaker(failure_threshold=5)
@timeout(30.0)
async def call_external_api():
    # This function will have comprehensive tracing including:
    # - Retry attempts
    # - Circuit breaker state changes
    # - Timeout events
    # - Performance metrics
    pass
```

### Correlation IDs

Link traces with structured logging:

```python
from semantrix.utils.logging import with_correlation_id
from semantrix.integrations.opentelemetry import trace_operation

@trace_operation("process_request")
async def process_request(request_id: str):
    with with_correlation_id(request_id):
        logger.info("Processing request", extra={"request_id": request_id})
        # All logs and traces will be linked by correlation ID
        result = await some_operation()
        logger.info("Request completed", extra={"request_id": request_id})
        return result
```

### Configuration Options

#### Sampling Configuration

```python
# Sample all traces (development)
initialize_opentelemetry(sampling_rate=1.0)

# Sample 10% of traces (production)
initialize_opentelemetry(sampling_rate=0.1)

# Sample based on parent trace
initialize_opentelemetry(sampling_rate=0.05)  # 5% of traces
```

#### Export Configuration

```python
# Console export for debugging
initialize_opentelemetry(enable_console_export=True)

# OTLP export for external systems
initialize_opentelemetry(
    traces_endpoint="http://jaeger:4317",
    metrics_endpoint="http://prometheus:4318"
)

# Multiple exporters
initialize_opentelemetry(
    traces_endpoint="http://jaeger:4317",
    jaeger_endpoint="http://jaeger:14268/api/traces",
    zipkin_endpoint="http://zipkin:9411/api/v2/spans"
)
```

#### Resource Attributes

```python
initialize_opentelemetry(
    service_name="semantrix-api",
    service_version="1.2.3",
    environment="production",
    resource_attributes={
        "service.instance.id": "api-1",
        "deployment.region": "us-west-2",
        "deployment.environment": "prod",
        "team": "ml-platform",
        "component": "semantrix",
    }
)
```

### External System Integration

#### Jaeger Integration

```python
# Initialize with Jaeger
initialize_opentelemetry(
    jaeger_endpoint="http://jaeger:14268/api/traces",
    service_name="semantrix-service"
)
```

#### Zipkin Integration

```python
# Initialize with Zipkin
initialize_opentelemetry(
    zipkin_endpoint="http://zipkin:9411/api/v2/spans",
    service_name="semantrix-service"
)
```

#### Prometheus Integration

```python
# Metrics will be available in Prometheus format
# via the OTLP metrics endpoint
initialize_opentelemetry(
    metrics_endpoint="http://prometheus:4318",
    service_name="semantrix-service"
)
```

### Best Practices

1. **Service Naming**: Use consistent service names across environments
2. **Sampling**: Use appropriate sampling rates for production vs development
3. **Resource Attributes**: Include relevant metadata for filtering and grouping
4. **Correlation IDs**: Use correlation IDs to link traces with logs
5. **Error Handling**: Always record exceptions in spans
6. **Performance**: Use sampling to control trace volume in high-traffic systems

### Troubleshooting

#### Common Issues

1. **No Traces Appearing**: Check sampling rate and exporter configuration
2. **High Memory Usage**: Reduce sampling rate or use head-based sampling
3. **Export Failures**: Verify endpoint URLs and network connectivity
4. **Missing Metrics**: Ensure metrics synchronization is called regularly

#### Debug Configuration

```python
# Enable console export for debugging
initialize_opentelemetry(
    enable_console_export=True,
    sampling_rate=1.0,  # Sample all traces
    service_name="semantrix-debug"
)
```

## Best Practices

### Error Handling

1. **Use Specific Exceptions**: Catch specific exception types rather than generic ones
2. **Provide Context**: Include relevant context in error messages
3. **Log Errors**: Always log errors with appropriate detail
4. **Graceful Degradation**: Implement fallback mechanisms

```python
try:
    result = await external_service_call()
except ConnectionError as e:
    logger.error("External service unavailable", exc_info=True)
    result = await fallback_service_call()
except TimeoutError as e:
    logger.error("External service timeout", exc_info=True)
    result = await cached_response()
```

### Retry Strategies

1. **Exponential Backoff**: Increase delay between retries
2. **Jitter**: Add randomness to prevent thundering herd
3. **Circuit Breakers**: Stop retrying when service is down
4. **Timeout**: Set reasonable timeouts for all operations

### Validation

1. **Validate Early**: Validate inputs as soon as possible
2. **Comprehensive Validation**: Check types, ranges, and formats
3. **Clear Error Messages**: Provide helpful validation error messages
4. **Performance**: Keep validation overhead minimal

### Monitoring

1. **Key Metrics**: Monitor request rate, error rate, response time
2. **Alerts**: Set up alerts for critical thresholds
3. **Logging**: Use structured logging with correlation IDs
4. **Dashboards**: Create dashboards for system health

## Troubleshooting

### Common Issues

#### High Error Rates

1. Check circuit breaker status
2. Review retry configuration
3. Monitor external service health
4. Check resource limits

#### Performance Issues

1. Monitor response time histograms
2. Check cache hit rates
3. Review timeout settings
4. Analyze resource usage

#### Memory Issues

1. Monitor memory gauges
2. Check for memory leaks
3. Review cache eviction policies
4. Analyze object lifecycle

### Debugging Tools

#### Log Analysis

```bash
# Search for errors
grep "ERROR" semantrix.log

# Find requests by correlation ID
grep "correlation_id.*request-123" semantrix.log

# Analyze response times
grep "duration_seconds" semantrix.log | jq '.duration_seconds'
```

#### Metrics Analysis

```python
# Get current metrics
from semantrix.utils.metrics import collect_metrics
metrics = collect_metrics()

# Analyze error rates
error_rate = metrics['counters']['errors_total']['value'] / metrics['counters']['requests_total']['value']

# Check response time percentiles
response_time_summary = metrics['histograms']['response_time_seconds']['summary']
```

### Recovery Procedures

#### Circuit Breaker Recovery

1. Check circuit breaker state
2. Verify external service health
3. Reset circuit breaker if needed
4. Monitor recovery attempts

#### Cache Recovery

1. Check cache store health
2. Verify cache configuration
3. Clear corrupted cache entries
4. Restart cache service if necessary

#### Database Recovery

1. Check database connectivity
2. Verify connection pool health
3. Review transaction logs
4. Restore from backup if needed

## Configuration

### Logging Configuration

```python
# Basic configuration
initialize_logging(
    log_level="INFO",
    log_format="json",
    log_file="logs/semantrix.log",
    max_bytes=10*1024*1024,  # 10MB
    backup_count=5
)

# Advanced configuration
initialize_logging(
    log_level="DEBUG",
    log_format="text",
    include_correlation_id=False
)
```

### Retry Configuration

```python
# Conservative retry settings
@retry(
    exceptions=(ConnectionError, TimeoutError),
    max_retries=3,
    initial_delay=1.0,
    max_delay=10.0,
    backoff_factor=2.0,
    jitter=0.1
)

# Aggressive retry settings
@retry(
    exceptions=(ValueError,),
    max_retries=5,
    initial_delay=0.1,
    max_delay=60.0,
    backoff_factor=1.5,
    jitter=0.2
)
```

### Circuit Breaker Configuration

```python
# Conservative circuit breaker
@circuit_breaker(
    failure_threshold=3,
    recovery_timeout=30.0,
    expected_exception=ConnectionError
)

# Aggressive circuit breaker
@circuit_breaker(
    failure_threshold=10,
    recovery_timeout=120.0,
    expected_exception=(ConnectionError, TimeoutError)
)
```

## Conclusion

The reliability improvements in Semantrix provide a robust foundation for production deployments. By following the best practices outlined in this guide, you can ensure your Semantrix-based applications are resilient, observable, and maintainable.

For additional support, refer to the API documentation and test suite for specific implementation details.
