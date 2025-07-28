"""
Redis Cache Store Example
========================

Demonstrates how to implement Redis as a cache store using the BaseCacheStore interface.
"""

import json
import time
from typing import Optional, Any, Dict
from semantrix import BaseCacheStore, EvictionPolicy, NoOpEvictionPolicy

class RedisCacheStore(BaseCacheStore):
    """
    Redis-based cache store implementation.

    Args:
        redis_client: An instance of redis.Redis, redis.cluster.RedisCluster, or any compatible client.
        key_prefix: Prefix for cache keys (default: 'semantrix:')
        eviction_policy: Eviction policy (defaults to NoOpEvictionPolicy since Redis handles eviction)
    """
    def __init__(self, 
                 redis_client: Any,  # Accepts redis.Redis or compatible
                 key_prefix: str = "semantrix:",
                 eviction_policy: Optional[EvictionPolicy] = None):
        """
        Initialize Redis cache store.
        
        Args:
            redis_client: Redis client instance (redis.Redis)
            key_prefix: Prefix for cache keys
            eviction_policy: Eviction policy (defaults to NoOp since Redis handles eviction)
        """
        self.redis = redis_client
        self.key_prefix = key_prefix
        self.eviction_policy = eviction_policy or NoOpEvictionPolicy()
        
    def _get_key(self, prompt: str) -> str:
        """Get Redis key for a prompt."""
        return f"{self.key_prefix}{hash(prompt)}"
    
    def get_exact(self, prompt: str) -> Optional[str]:
        """Get an exact match for a prompt from Redis."""
        key = self._get_key(prompt)
        value = self.redis.get(key)
        
        if value is not None:
            # Deserialize the response
            try:
                data = json.loads(value.decode('utf-8'))
                return data.get('response')
            except (json.JSONDecodeError, KeyError):
                # Fallback for old format
                return value.decode('utf-8')
        
        return None
    
    def add(self, prompt: str, response: str) -> None:
        """Add a prompt-response pair to Redis."""
        key = self._get_key(prompt)
        
        # Serialize the data
        data = {
            'response': response,
            'timestamp': time.time(),
            'prompt_hash': hash(prompt)
        }
        
        # Store in Redis
        self.redis.set(key, json.dumps(data))
    
    def enforce_limits(self, resource_limits: Any) -> None:
        """Enforce limits - Redis handles its own eviction."""
        # Redis handles eviction automatically, so we use NoOp policy
        pass
    
    def clear(self) -> None:
        """Clear all cached items from Redis."""
        # Delete all keys with our prefix
        pattern = f"{self.key_prefix}*"
        keys = self.redis.keys(pattern)
        if keys:
            self.redis.delete(*keys)
    
    def size(self) -> int:
        """Get the number of cached items in Redis."""
        pattern = f"{self.key_prefix}*"
        keys = self.redis.keys(pattern)
        return len(keys)
    
    def get_eviction_policy(self) -> EvictionPolicy:
        """Get the eviction policy for this cache store."""
        return self.eviction_policy
    
    def get_stats(self) -> dict:
        """Get Redis cache statistics."""
        pattern = f"{self.key_prefix}*"
        keys = self.redis.keys(pattern)
        
        return {
            'size': len(keys),
            'redis_info': self.redis.info(),
            'eviction_policy': type(self.eviction_policy).__name__,
            'key_prefix': self.key_prefix
        }

# Example usage with SemantrixClient
def demonstrate_redis_integration():
    """Demonstrate Redis integration with SemantrixClient."""
    
    try:
        import redis
        
        # Create Redis client
        redis_client = redis.Redis(
            host='localhost',
            port=6379,
            db=0,
            decode_responses=False  # We handle encoding ourselves
        )
        
        # Test Redis connection
        redis_client.ping()
        print("✓ Redis connection successful")
        
        # Create Redis cache store
        redis_store = RedisCacheStore(
            redis_client=redis_client,
            key_prefix="semantrix:cache:"
        )
        
        # Use with SemantrixClient
        from semantrix import SemantrixClient
        
        client = SemantrixClient(
            cache_store=redis_store,
            similarity_threshold=0.85
        )
        
        # Test caching
        prompt = "What is the capital of France?"
        response = "The capital of France is Paris."
        
        # Add to cache
        client.cache.add(prompt, response)
        print(f"✓ Added to Redis cache: {prompt}")
        
        # Retrieve from cache
        cached_response = client.cache.cache_store.get_exact(prompt)
        print(f"✓ Retrieved from Redis: {cached_response}")
        
        # Get stats
        stats = redis_store.get_stats()
        print(f"✓ Redis cache stats: {stats['size']} items")
        
        return True
        
    except ImportError:
        print("✗ Redis library not installed. Install with: pip install redis")
        return False
    except Exception as e:
        print(f"✗ Redis error: {e}")
        return False

# Example with Redis configuration
def create_redis_store_with_config():
    """Create Redis store with custom configuration."""
    
    try:
        import redis
        
        # Redis configuration
        redis_config = {
            'host': 'localhost',
            'port': 6379,
            'db': 0,
            'password': None,  # Set if Redis requires authentication
            'decode_responses': False,
            'socket_timeout': 5,
            'socket_connect_timeout': 5,
            'retry_on_timeout': True
        }
        
        redis_client = redis.Redis(**redis_config)
        
        # Create cache store with custom prefix
        redis_store = RedisCacheStore(
            redis_client=redis_client,
            key_prefix="myapp:semantrix:",
            eviction_policy=NoOpEvictionPolicy()  # Redis handles eviction
        )
        
        return redis_store
        
    except ImportError:
        print("Redis library not available")
        return None

# Example with Redis cluster
def create_redis_cluster_store():
    """Create Redis store with Redis cluster."""
    
    try:
        from redis.cluster import RedisCluster
        
        # Redis cluster configuration
        cluster_config = {
            'startup_nodes': [
                {'host': 'localhost', 'port': 7000},
                {'host': 'localhost', 'port': 7001},
                {'host': 'localhost', 'port': 7002}
            ],
            'decode_responses': False
        }
        
        redis_cluster = RedisCluster(**cluster_config)
        
        # Create cache store
        redis_store = RedisCacheStore(
            redis_client=redis_cluster,
            key_prefix="cluster:semantrix:"
        )
        
        return redis_store
        
    except ImportError:
        print("Redis cluster not available")
        return None

if __name__ == "__main__":
    print("Redis Cache Store Example")
    print("=" * 40)
    
    # Test Redis integration
    success = demonstrate_redis_integration()
    
    if success:
        print("\n🎉 Redis integration successful!")
        print("\nTo use Redis with Semantrix:")
        print("1. Install Redis: pip install redis")
        print("2. Start Redis server")
        print("3. Use RedisCacheStore with SemantrixClient")
    else:
        print("\n❌ Redis integration failed. Check Redis server and dependencies.") 