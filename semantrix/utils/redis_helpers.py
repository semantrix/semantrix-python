from typing import Any, Optional
from semantrix import BaseCacheStore, NoOpEvictionPolicy
from semantrix.cache_store.stores import RedisCacheStore

def create_redis_cache_store(redis_client: Any, key_prefix: str = "semantrix:", eviction_policy: Optional[Any] = None) -> BaseCacheStore:
    """
    Helper to create a RedisCacheStore from a redis client.
    Args:
        redis_client: An instance of redis.Redis, redis.cluster.RedisCluster, or compatible.
        key_prefix: Prefix for cache keys.
        eviction_policy: Optionally override the default NoOpEvictionPolicy.
    Returns:
        An instance of RedisCacheStore implementing BaseCacheStore.
    """
    return RedisCacheStore(
        redis_client=redis_client,
        key_prefix=key_prefix,
        eviction_policy=eviction_policy or NoOpEvictionPolicy()
    ) 