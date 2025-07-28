# Redis Integration with Semantrix

> **Best Practice:**
> 
> Always accept a `redis.Redis` (or compatible) client instance directly in your custom cache store constructor. This gives users full control over connection settings, works with all Redis deployments (self-hosted, cloud, cluster, etc.), and makes your code more testable and flexible. Do **not** accept host/port directly or create the client inside your class—let the user manage the client!

## Memcached Support

Semantrix also supports Memcached out of the box:

```python
from semantrix import MemcachedCacheStore
from pymemcache.client.base import Client

memcached_client = Client(('localhost', 11211))
memcached_store = MemcachedCacheStore(memcached_client)

from semantrix import SemantrixClient
client = SemantrixClient(cache_store=memcached_store)
```

Memcached is a great choice for simple, fast, ephemeral caching in distributed environments.

---

## Quick Start with Helper

If you just want to get started quickly, use the provided helper:

```python
from semantrix import create_redis_cache_store  # Top-level import!
import redis

redis_client = redis.Redis(host='localhost', port=6379)
redis_store = create_redis_cache_store(redis_client)

from semantrix import SemantrixClient
client = SemantrixClient(cache_store=redis_store)
```

This will set up a Redis-backed cache store with sensible defaults.

---

## Direct Usage

You can also use the official Redis cache store directly:

```python
from semantrix import RedisCacheStore
import redis

redis_client = redis.Redis(host='localhost', port=6379)
redis_store = RedisCacheStore(redis_client)

from semantrix import SemantrixClient
client = SemantrixClient(cache_store=redis_store)
```

---

This guide shows how to use Redis as a cache store with Semantrix.

## Installation

```bash
pip install redis semantrix
```

## Basic Usage

### 1. Create Redis Cache Store

```python
import redis
from semantrix import BaseCacheStore, NoOpEvictionPolicy

class RedisCacheStore(BaseCacheStore):
    """Redis-based cache store implementation."""
    
    def __init__(self, redis_client, key_prefix="semantrix:"):
        self.redis = redis_client
        self.key_prefix = key_prefix
        self.eviction_policy = NoOpEvictionPolicy()  # Redis handles eviction
    
    def get_exact(self, prompt: str):
        """Get exact match from Redis."""
        key = f"{self.key_prefix}{hash(prompt)}"
        value = self.redis.get(key)
        return value.decode('utf-8') if value else None
    
    def add(self, prompt: str, response: str):
        """Add to Redis cache."""
        key = f"{self.key_prefix}{hash(prompt)}"
        self.redis.set(key, response)
    
    def enforce_limits(self, resource_limits):
        """Redis handles its own eviction."""
        pass
    
    def clear(self):
        """Clear all cached items."""
        pattern = f"{self.key_prefix}*"
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)
    
    def size(self):
        """Get cache size."""
        pattern = f"{self.key_prefix}*"
        keys = self.redis.keys(pattern)
        return len(keys)
    
    def get_eviction_policy(self):
        """Get eviction policy."""
        return self.eviction_policy
```

### 2. Use with SemantrixClient

```python
from semantrix import SemantrixClient

# Create Redis client
redis_client = redis.Redis(
    host='localhost',
    port=6379,
    db=0
)

# Create Redis cache store
redis_store = RedisCacheStore(
    redis_client=redis_client,
    key_prefix="myapp:semantrix:"
)

# Use with SemantrixClient
client = SemantrixClient(
    cache_store=redis_store,
    similarity_threshold=0.85
)

# Test caching
prompt = "What is the capital of France?"
response = "The capital of France is Paris."

# Add to cache
client.cache.add(prompt, response)

# Get from cache
cached = client.cache.cache_store.get_exact(prompt)
print(f"Cached response: {cached}")
```

## Production Setup

### Redis Configuration

```python
# Production Redis configuration
redis_config = {
    'host': 'redis.example.com',
    'port': 6379,
    'password': 'your_password',
    'ssl': True,
    'decode_responses': False,
    'socket_timeout': 10,
    'retry_on_timeout': True
}

redis_client = redis.Redis(**redis_config)
```