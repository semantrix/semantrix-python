"""
Amazon ElastiCache store implementation for Semantrix.

This module provides cache store implementations for Amazon ElastiCache,
supporting both Redis and Memcached protocols.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union, cast

from redis.asyncio import Redis
from redis.exceptions import RedisError

from semantrix.cache_store.base import BaseCacheStore, EvictionPolicy, NoOpEvictionPolicy

# Configure logging
logger = logging.getLogger(__name__)

class ElastiCacheStore(BaseCacheStore):
    """
    Amazon ElastiCache store implementation.
    
    This implementation uses the Redis protocol by default, which is recommended
    for most use cases due to its rich feature set and better performance.
    
    Args:
        endpoint: The ElastiCache endpoint (e.g., 'my-cluster.xxxxx.ng.0001.aps1.cache.amazonaws.com:6379')
        port: The port number (default: 6379 for Redis, 11211 for Memcached)
        use_redis: Whether to use Redis protocol (True) or Memcached protocol (False)
        ssl: Whether to use SSL/TLS (default: True for production, False for local testing)
        ssl_ca_certs: Path to CA certificate file (required if using SSL)
        username: Username for authentication (if using Redis 6+ ACLs)
        password: Password for authentication
        **kwargs: Additional arguments for BaseCacheStore
    """
    
    def __init__(
        self,
        endpoint: str,
        port: Optional[int] = None,
        use_redis: bool = True,
        ssl: bool = True,
        ssl_ca_certs: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        
        # Parse endpoint and port
        if ":" in endpoint:
            endpoint, port_str = endpoint.split(":", 1)
            port = int(port_str)
        
        self.endpoint = endpoint
        self.port = port or (6379 if use_redis else 11211)
        self.use_redis = use_redis
        self.ssl = ssl
        self.ssl_ca_certs = ssl_ca_certs
        self.username = username
        self.password = password
        
        # Initialize Redis client (we'll use Redis client for both protocols for simplicity)
        self._client: Optional[Redis] = None
        self._connected = False
        self._lock = asyncio.Lock()
    
    async def _ensure_connected(self) -> None:
        """Ensure we have an active connection to ElastiCache."""
        if self._connected and self._client is not None:
            try:
                # Simple ping to check connection
                await self._client.ping()
                return
            except Exception:
                self._connected = False
                if self._client:
                    await self._client.close()
        
        async with self._lock:
            if not self._connected:
                try:
                    # Configure SSL if needed
                    ssl_cert_reqs = None
                    if self.ssl:
                        ssl_cert_reqs = 'required' if self.ssl_ca_certs else None
                    
                    # Create Redis client
                    self._client = Redis(
                        host=self.endpoint,
                        port=self.port,
                        username=self.username,
                        password=self.password,
                        ssl=self.ssl,
                        ssl_ca_certs=self.ssl_ca_certs,
                        ssl_cert_reqs=ssl_cert_reqs,
                        decode_responses=True,
                        socket_timeout=5.0,
                        socket_connect_timeout=5.0,
                        retry_on_timeout=True,
                        max_connections=100,
                        health_check_interval=30,
                    )
                    
                    # Test the connection
                    await self._client.ping()
                    self._connected = True
                    logger.info(f"Connected to ElastiCache at {self.endpoint}:{self.port} ({'Redis' if self.use_redis else 'Memcached'})")
                    
                except Exception as e:
                    self._connected = False
                    logger.error(f"Failed to connect to ElastiCache: {e}")
                    raise
    
    async def get_exact(self, prompt: str) -> Optional[str]:
        """Get a cached response if it exists and is not expired."""
        try:
            await self._ensure_connected()
            if not self._client:
                return None
                
            value = await self._client.get(prompt)
            if value and self._client:
                # Update last accessed time (for LRU eviction)
                await self._client.hset(f"{prompt}:meta", "last_accessed", str(time.time()))
                
            return value
            
        except RedisError as e:
            logger.error(f"Error getting item from ElastiCache: {e}")
            return None
    
    async def add(self, prompt: str, response: str, ttl: Optional[float] = None) -> None:
        """Add a response to the cache."""
        try:
            await self._ensure_connected()
            if not self._client:
                raise RuntimeError("Not connected to ElastiCache")
            
            # Convert ttl to milliseconds for Redis
            ttl_ms = int(ttl * 1000) if ttl is not None else None
            
            # Set the value with TTL
            await self._client.set(
                prompt,
                response,
                px=ttl_ms if ttl_ms is not None else None,
                nx=False  # Overwrite if exists
            )
            
            # Store metadata for eviction policies
            now = time.time()
            metadata = {
                "created_at": str(now),
                "last_accessed": str(now),
                "access_count": "1"
            }
            await self._client.hset(f"{prompt}:meta", mapping=metadata)
            
        except RedisError as e:
            logger.error(f"Error adding item to ElastiCache: {e}")
            raise
    
    async def clear(self) -> None:
        """Clear all items from the cache."""
        try:
            await self._ensure_connected()
            if self._client:
                await self._client.flushdb()
        except RedisError as e:
            logger.error(f"Error clearing ElastiCache: {e}")
            raise
    
    async def size(self) -> int:
        """Get the number of items in the cache."""
        try:
            await self._ensure_connected()
            if not self._client:
                return 0
                
            # Get total keys (excluding metadata)
            keys = await self._client.keys("*")
            # Filter out metadata keys
            cache_keys = [k for k in keys if not k.endswith(":meta") and ":" not in k]
            return len(cache_keys)
            
        except RedisError as e:
            logger.error(f"Error getting ElastiCache size: {e}")
            return 0
    
    async def enforce_limits(self, resource_limits: Any) -> None:
        """Enforce cache size limits."""
        try:
            await self._ensure_connected()
            if not self._client:
                return
                
            current_size = await self.size()
            max_size = getattr(resource_limits, 'max_size', None)
            
            if max_size is not None and current_size > max_size:
                await self.eviction_policy.apply(self, max_size)
                
        except RedisError as e:
            logger.error(f"Error enforcing ElastiCache limits: {e}")
            raise
    
    def get_eviction_policy(self) -> EvictionPolicy:
        """Get the eviction policy for this cache store."""
        return self.eviction_policy
    
    async def close(self) -> None:
        """Close the ElastiCache connection."""
        if self._client:
            try:
                await self._client.close()
                self._connected = False
                logger.info("Closed ElastiCache connection")
            except Exception as e:
                logger.error(f"Error closing ElastiCache connection: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            await self._ensure_connected()
            if not self._client:
                return {"error": "Not connected to ElastiCache"}
                
            stats = {
                "backend": "elasticache",
                "endpoint": f"{self.endpoint}:{self.port}",
                "protocol": "redis" if self.use_redis else "memcached",
                "connected": self._connected,
                "ssl": self.ssl,
            }
            
            if self._client and self.use_redis:
                try:
                    # Get Redis info
                    info = await self._client.info()
                    stats.update({
                        "version": info.get("redis_version"),
                        "uptime_seconds": info.get("uptime_in_seconds"),
                        "total_connections_received": info.get("total_connections_received"),
                        "total_commands_processed": info.get("total_commands_processed"),
                        "keyspace_hits": info.get("keyspace_hits"),
                        "keyspace_misses": info.get("keyspace_misses"),
                        "used_memory_human": info.get("used_memory_human"),
                        "used_memory_peak_human": info.get("used_memory_peak_human"),
                        "connected_clients": info.get("connected_clients"),
                        "blocked_clients": info.get("blocked_clients"),
                    })
                except Exception as e:
                    logger.warning(f"Could not get Redis info: {e}")
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting ElastiCache stats: {e}")
            return {"error": str(e)}
    
    def __del__(self) -> None:
        """Ensure connection is closed when the object is garbage collected."""
        if hasattr(self, '_client') and self._client:
            try:
                asyncio.get_event_loop().run_until_complete(self.close())
            except Exception:
                pass
