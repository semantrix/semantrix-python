"""
Simple Redis Usage Example
=========================

Basic example of how to use Redis as a cache store with Semantrix.
"""

# Step 1: Install Redis
# pip install redis

# Step 2: Create Redis Cache Store
from semantrix import BaseCacheStore, NoOpEvictionPolicy
from typing import Any

class RedisCacheStore(BaseCacheStore):
    """
    Simple Redis cache store implementation.

    Args:
        redis_client: An instance of redis.Redis, redis.cluster.RedisCluster, or any compatible client.
        key_prefix: Prefix for cache keys (default: 'semantrix:')
    """
    
    def __init__(self, redis_client: Any, key_prefix="semantrix:"):
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

# Step 3: Usage Example
def use_redis_with_semantrix():
    """Example of using Redis with Semantrix."""
    
    # Import required libraries
    try:
        import redis
        from semantrix import SemantrixClient
    except ImportError:
        print("Install required packages: pip install redis semantrix")
        return
    
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
    print(f"✓ Added to Redis: {prompt}")
    
    # Get from cache
    cached = client.cache.cache_store.get_exact(prompt)
    print(f"✓ Retrieved from Redis: {cached}")
    
    # Get cache size
    size = redis_store.size()
    print(f"✓ Cache size: {size} items")

# Step 4: Production Example
def production_redis_setup():
    """Production-ready Redis setup."""
    
    import redis
    from semantrix import SemantrixClient, NoOpEvictionPolicy
    
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
    
    # Create cache store with NoOp policy (Redis handles eviction)
    redis_store = RedisCacheStore(
        redis_client=redis_client,
        key_prefix="prod:semantrix:"
    )
    
    # Use with SemantrixClient
    client = SemantrixClient(
        cache_store=redis_store,
        similarity_threshold=0.9,  # Higher threshold for production
        max_memory_gb=2.0  # Memory limits
    )
    
    return client

if __name__ == "__main__":
    print("Redis Integration Example")
    print("=" * 30)
    
    # Try the simple example
    try:
        use_redis_with_semantrix()
        print("\n🎉 Redis integration works!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Make sure Redis server is running: redis-server") 