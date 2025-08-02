# Cache Store Implementations

Semantrix provides multiple cache store implementations that can be used for different deployment scenarios. This document covers the available cache stores and how to use them.

## Table of Contents

1. [Available Cache Stores](#available-cache-stores)
2. [Common Features](#common-features)
3. [Installation](#installation)
4. [Usage Examples](#usage-examples)
   - [DynamoDB Cache Store](#dynamodb-cache-store)
   - [ElastiCache Store](#elasticache-store)
   - [Google Memorystore Cache Store](#google-memorystore-cache-store)
5. [Performance Considerations](#performance-considerations)
6. [Troubleshooting](#troubleshooting)

## Available Cache Stores

| Store | Description | Best For |
|-------|-------------|----------|
| `InMemoryStore` | In-memory cache (default) | Development, testing, single-process applications |
| `RedisCacheStore` | Redis-backed cache | Distributed caching, high performance |
| `MemcachedCacheStore` | Memcached-backed cache | Simple distributed caching |
| `SQLiteCacheStore` | SQLite-backed persistent cache | Local development, simple persistence |
| `MongoDBCacheStore` | MongoDB-backed cache | Document storage, flexible schema |
| `PostgreSQLCacheStore` | PostgreSQL-backed cache | Relational data, complex queries |
| `DocumentDBCacheStore` | AWS DocumentDB-backed cache | MongoDB-compatible on AWS |
| `DynamoDBCacheStore` | AWS DynamoDB-backed cache | Serverless, scalable NoSQL on AWS |
| `ElastiCacheStore` | AWS ElastiCache (Redis/Memcached) | Managed Redis/Memcached on AWS |
| `GoogleMemorystoreCacheStore` | Google Cloud Memorystore for Redis | Managed Redis on Google Cloud |

## Common Features

All cache store implementations support:

- Asynchronous operations (async/await)
- Time-to-live (TTL) for cached items
- Eviction policies (LRU, LFU, FIFO, etc.)
- Connection pooling
- Resource cleanup
- Monitoring and statistics

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

## Usage Examples

### DynamoDB Cache Store

```python
from semantrix.cache_store.stores import DynamoDBCacheStore

# Initialize with default settings
cache = DynamoDBCacheStore(
    table_name="semantrix-cache",
    region_name="ap-south-1"  # Mumbai region
)

# Or with custom configuration
cache = DynamoDBCacheStore(
    table_name="semantrix-cache",
    region_name="ap-south-1",
    read_capacity_units=10,
    write_capacity_units=10,
    ttl_attribute="expires_at"
)

# Using with async/await
async def example():
    await cache.add("key1", "value1", ttl=3600)  # Cache for 1 hour
    value = await cache.get_exact("key1")
    print(f"Cached value: {value}")
```

### ElastiCache Store

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

# Using with async/await
async def example():
    await cache.add("key1", "value1", ttl=3600)
    value = await cache.get_exact("key1")
    print(f"Cached value: {value}")
```

### Google Memorystore Cache Store

```python
from semantrix.cache_store.stores import GoogleMemorystoreCacheStore

# Initialize with project and instance details
cache = GoogleMemorystoreCacheStore(
    project_id="your-project-id",
    region="us-central1",
    instance_id="semantrix-cache",
    memory_size_gb=1
)

# Using with async/await
async def example():
    await cache.add("key1", "value1", ttl=3600)
    value = await cache.get_exact("key1")
    print(f"Cached value: {value}")
    
    # List all instances
    instances = await GoogleMemorystoreCacheStore.list_instances(
        project_id="your-project-id",
        region="us-central1"
    )
    print(f"Available instances: {instances}")
```

## Performance Considerations

1. **DynamoDB**:
   - Enable auto-scaling for production workloads
   - Use provisioned capacity for predictable workloads
   - Consider using DAX for read-heavy workloads

2. **ElastiCache (Redis)**:
   - Use Redis cluster mode for large datasets
   - Enable in-transit encryption for security
   - Configure appropriate node types based on workload

3. **Google Memorystore**:
   - Choose appropriate machine types
   - Enable read replicas for high availability
   - Configure memory appropriately for your workload

## Troubleshooting

### Common Issues

1. **Connection Timeouts**:
   - Check firewall rules and security groups
   - Verify network connectivity
   - Check if the instance is running

2. **Authentication Errors**:
   - Verify IAM permissions
   - Check credentials and service account permissions
   - Ensure proper authentication method is used

3. **Performance Issues**:
   - Check instance metrics (CPU, memory, network)
   - Review query patterns and indexes
   - Consider scaling up or enabling read replicas

### Getting Help

For additional help, please open an issue on the [GitHub repository](https://github.com/semantrix/semantrix-python).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
