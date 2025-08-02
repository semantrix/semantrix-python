# Cache Store Guide

This guide provides detailed information about the cache store implementations available in Semantrix, with a focus on the newly added persistent and distributed cache stores.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Cache Store Implementations](#cache-store-implementations)
   - [In-Memory Store](#in-memory-store)
   - [DynamoDB Cache Store](#dynamodb-cache-store)
   - [ElastiCache Store](#elasticache-store)
   - [Google Memorystore Cache Store](#google-memorystore-cache-store)
5. [Advanced Configuration](#advanced-configuration)
6. [Performance Tuning](#performance-tuning)
7. [Monitoring and Metrics](#monitoring-and-metrics)
8. [Troubleshooting](#troubleshooting)
9. [Migration Guide](#migration-guide)

## Overview

Semantrix provides multiple cache store implementations to suit different deployment scenarios, from simple in-memory caches to distributed, persistent caches. All cache stores implement a common interface for consistency.

## Installation

### Core Installation

```bash
pip install semantrix
```

### Optional Dependencies

For specific cache stores, install the required dependencies:

```bash
# For DynamoDB
pip install boto3

# For ElastiCache (Redis protocol)
pip install redis

# For Google Memorystore
pip install google-cloud-redis

# For all optional dependencies
pip install semantrix[all]
```

## Quick Start

```python
from semantrix.cache_store.stores import InMemoryStore, DynamoDBCacheStore, ElastiCacheStore, GoogleMemorystoreCacheStore

# In-memory cache (default)
cache = InMemoryStore()

# DynamoDB cache
dynamo_cache = DynamoDBCacheStore(
    table_name="semantrix-cache",
    region_name="us-west-2"
)

# ElastiCache (Redis protocol)
elasti_cache = ElastiCacheStore(
    endpoint="my-cache.xxxxx.ng.0001.aps1.cache.amazonaws.com:6379",
    ssl=True
)

# Google Memorystore
google_cache = GoogleMemorystoreCacheStore(
    project_id="your-project-id",
    region="us-central1",
    instance_id="semantrix-cache"
)

# Using the cache
async def example():
    # Add to cache
    await cache.add("key1", "value1", ttl=3600)  # Cache for 1 hour
    
    # Get from cache
    value = await cache.get_exact("key1")
    print(f"Cached value: {value}")
    
    # Get cache stats
    stats = await cache.get_stats()
    print(f"Cache stats: {stats}")
```

## Cache Store Implementations

### In-Memory Store

**Use Case**: Development, testing, or single-process applications.

```python
from semantrix.cache_store.stores import InMemoryStore

cache = InMemoryStore(
    max_size=1000,  # Maximum number of items
    eviction_policy="lru"  # LRU, LFU, or FIFO
)
```

### DynamoDB Cache Store

**Use Case**: Serverless applications, AWS environments, when you need a managed NoSQL solution.

```python
from semantrix.cache_store.stores import DynamoDBCacheStore

cache = DynamoDBCacheStore(
    table_name="semantrix-cache",
    region_name="us-west-2",
    read_capacity_units=5,
    write_capacity_units=5,
    ttl_attribute="expires_at"
)
```

**Prerequisites**:
- AWS credentials configured (via environment variables, IAM roles, or AWS config)
- DynamoDB table with appropriate permissions

### ElastiCache Store

**Use Case**: High-performance, low-latency caching in AWS environments.

```python
from semantrix.cache_store.stores import ElastiCacheStore

# Using Redis protocol (recommended)
cache = ElastiCacheStore(
    endpoint="my-cache.xxxxx.ng.0001.aps1.cache.amazonaws.com:6379",
    ssl=True,
    ssl_ca_certs="/path/to/redis-ca.pem"
)

# Or using Memcached protocol
cache = ElastiCacheStore(
    endpoint="my-memcached.xxxxx.ng.0001.aps1.cache.amazonaws.com:11211",
    use_redis=False
)
```

**Prerequisites**:
- ElastiCache cluster with appropriate security groups
- Network access from your application

### Google Memorystore Cache Store

**Use Case**: Managed Redis on Google Cloud Platform.

```python
from semantrix.cache_store.stores import GoogleMemorystoreCacheStore

cache = GoogleMemorystoreCacheStore(
    project_id="your-project-id",
    region="us-central1",
    instance_id="semantrix-cache",
    memory_size_gb=1
)

# List instances
instances = await GoogleMemorystoreCacheStore.list_instances(
    project_id="your-project-id",
    region="us-central1"
)
```

**Prerequisites**:
- Google Cloud SDK installed and configured
- Appropriate IAM permissions
- Memorystore instance created

## Advanced Configuration

### Connection Pooling

All cache stores that make network connections use connection pooling. You can configure the pool size and timeouts:

```python
# Example with DynamoDB
cache = DynamoDBCacheStore(
    table_name="semantrix-cache",
    region_name="us-west-2",
    max_pool_connections=100,
    connect_timeout=5.0,
    read_timeout=5.0
)
```

### Retry Logic

Network-based cache stores implement automatic retries for transient failures:

```python
# Example with Google Memorystore
cache = GoogleMemorystoreCacheStore(
    project_id="your-project-id",
    region="us-central1",
    instance_id="semantrix-cache",
    retry_attempts=3,
    retry_delay=0.5
)
```

## Performance Tuning

### Batch Operations

For bulk operations, use batch methods when available:

```python
# Bulk add
items = {"key1": "value1", "key2": "value2", "key3": "value3"}
await cache.batch_add(items, ttl=3600)

# Bulk get
keys = ["key1", "key2", "key3"]
results = await cache.batch_get(keys)
```

### Compression

For large values, enable compression:

```python
cache = InMemoryStore(compress_threshold=1024)  # Compress values > 1KB
```

## Monitoring and Metrics

All cache stores provide a `get_stats()` method for monitoring:

```python
# Get cache statistics
stats = await cache.get_stats()
print(f"Cache hits: {stats.get('hits', 0)}")
print(f"Cache misses: {stats.get('misses', 0)}")
print(f"Size: {await cache.size()}")
```

## Troubleshooting

### Common Issues

1. **Connection Timeouts**:
   - Check network connectivity and security groups
   - Verify the cache instance is running
   - Check DNS resolution

2. **Authentication Errors**:
   - Verify credentials and IAM permissions
   - Check service account configuration

3. **Performance Issues**:
   - Monitor cache hit/miss ratios
   - Check for hot keys
   - Consider scaling up the cache instance

## Migration Guide

### Upgrading from Previous Versions

If you're upgrading from a previous version of Semantrix, check the [changelog](CHANGELOG.md) for any breaking changes to the cache store APIs.

### Migrating Between Cache Stores

To migrate data between different cache stores:

```python
async def migrate_cache(source, destination):
    # Get all keys (implementation depends on source store)
    keys = await source.keys()
    
    # Copy each item
    for key in keys:
        value = await source.get_exact(key)
        if value is not None:
            await destination.add(key, value)
    
    print(f"Migrated {len(keys)} items")
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
