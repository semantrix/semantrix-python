"""
Google Cloud Memorystore for Redis implementation for Semantrix.

This module provides a cache store implementation using Google Cloud Memorystore for Redis.
It's optimized for Google Cloud Platform (GCP) environments.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, cast

from google.cloud import redis_v1
from google.cloud.redis_v1.types import Instance
from google.api_core.exceptions import GoogleAPICallError, RetryError
from redis.asyncio import Redis
from redis.exceptions import RedisError

from semantrix.cache_store.base import BaseCacheStore, EvictionPolicy, NoOpEvictionPolicy, DeletionMode
from semantrix.exceptions import CacheOperationError
from semantrix.utils.logging import get_logger

# Configure logging
logger = get_logger(__name__)

class GoogleMemorystoreCacheStore(BaseCacheStore):
    """
    Google Cloud Memorystore for Redis implementation.
    
    This implementation uses Google Cloud Memorystore for Redis as the backend.
    It provides a fully managed Redis service on Google Cloud Platform.
    
    Args:
        project_id: Google Cloud project ID
        region: Google Cloud region (e.g., 'us-central1')
        instance_id: Memorystore instance ID
        location: Location of the instance (defaults to region if not specified)
        network: VPC network name (e.g., 'projects/{project}/global/networks/default')
        redis_version: Redis version (e.g., 'REDIS_6_X')
        tier: Instance tier (e.g., 'BASIC', 'STANDARD_HA')
        memory_size_gb: Memory size in GB (default: 1)
        connect_timeout: Connection timeout in seconds (default: 5.0)
        **kwargs: Additional arguments for BaseCacheStore
    """
    
    def __init__(
        self,
        project_id: str,
        region: str,
        instance_id: str,
        location: Optional[str] = None,
        network: Optional[str] = None,
        redis_version: str = "REDIS_6_X",
        tier: str = "BASIC",
        memory_size_gb: int = 1,
        connect_timeout: float = 5.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.project_id = project_id
        self.region = region
        self.instance_id = instance_id
        self.location = location or region
        self.network = network or f"projects/{project_id}/global/networks/default"
        self.redis_version = redis_version
        self.tier = tier
        self.memory_size_gb = memory_size_gb
        self.connect_timeout = connect_timeout
        
        # Initialize clients
        self._redis_client: Optional[Redis] = None
        self._cloud_redis_client: Optional[redis_v1.CloudRedisClient] = None
        self._instance: Optional[Instance] = None
        self._connected = False
        self._lock = asyncio.Lock()
    
    async def _ensure_connected(self) -> None:
        """Ensure we have an active connection to Google Memorystore."""
        if self._connected and self._redis_client is not None:
            try:
                # Simple ping to check connection
                await self._redis_client.ping()
                return
            except RedisError:
                self._connected = False
                if self._redis_client:
                    await self._redis_client.close()
        
        async with self._lock:
            if not self._connected:
                try:
                    # Initialize Cloud Redis client
                    self._cloud_redis_client = redis_v1.CloudRedisClient()
                    
                    # Get or create instance
                    instance_name = self._cloud_redis_client.instance_path(
                        self.project_id, self.region, self.instance_id
                    )
                    
                    try:
                        # Try to get existing instance
                        self._instance = await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: self._cloud_redis_client.get_instance(name=instance_name)
                        )
                        logger.info(f"Connected to existing Memorystore instance: {self.instance_id}")
                    except GoogleAPICallError as e:
                        if e.code == 5:  # NOT_FOUND
                            # Create new instance if it doesn't exist
                            logger.info(f"Creating new Memorystore instance: {self.instance_id}")
                            parent = f"projects/{self.project_id}/locations/{self.location}"
                            
                            instance = {
                                "display_name": f"semantrix-cache-{int(time.time())}",
                                "tier": self.tier,
                                "memory_size_gb": self.memory_size_gb,
                                "redis_version": self.redis_version,
                                "authorized_network": self.network,
                            }
                            
                            operation = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: self._cloud_redis_client.create_instance(
                                    parent=parent,
                                    instance_id=self.instance_id,
                                    instance=instance,
                                )
                            )
                            
                            # Wait for operation to complete
                            await asyncio.get_event_loop().run_in_executor(
                                None,
                                operation.result,
                            )
                            
                            self._instance = await asyncio.get_event_loop().run_in_executor(
                                None,
                                lambda: self._cloud_redis_client.get_instance(name=instance_name)
                            )
                            logger.info(f"Created Memorystore instance: {self.instance_id}")
                        else:
                            raise
                    
                    # Connect to Redis instance
                    redis_host = self._instance.host
                    redis_port = self._instance.port
                    
                    # Initialize Redis client
                    self._redis_client = Redis(
                        host=redis_host,
                        port=redis_port,
                        socket_connect_timeout=self.connect_timeout,
                        socket_timeout=self.connect_timeout,
                        retry_on_timeout=True,
                        decode_responses=True,
                    )
                    
                    # Test the connection
                    await self._redis_client.ping()
                    self._connected = True
                    logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
                    
                except (RedisError, GoogleAPICallError, RetryError) as e:
                    self._connected = False
                    logger.error(f"Failed to connect to Google Memorystore: {e}")
                    raise CacheOperationError("Failed to connect to Google Memorystore", original_exception=e) from e
    
    async def get_exact(self, prompt: str) -> Optional[str]:
        """Get a cached response if it exists and is not expired."""
        try:
            await self._ensure_connected()
            if not self._redis_client:
                return None
                
            # Check if item is tombstoned
            tombstoned = await self._redis_client.hget(f"{prompt}:meta", "tombstoned")
            if tombstoned == "1":
                return None
                
            value = await self._redis_client.get(prompt)
            if value and self._redis_client:
                # Update last accessed time (for LRU eviction)
                await self._redis_client.hset(f"{prompt}:meta", "last_accessed", str(time.time()))
                
            return value
            
        except (RedisError, GoogleAPICallError) as e:
            logger.error(f"Error getting item from Google Memorystore: {e}")
            raise CacheOperationError("Failed to get item from Google Memorystore", original_exception=e) from e
    
    async def add(self, prompt: str, response: str, ttl: Optional[float] = None) -> None:
        """Add a response to the cache."""
        try:
            await self._ensure_connected()
            if not self._redis_client:
                raise CacheOperationError("Not connected to Google Memorystore")
            
            # Convert ttl to milliseconds for Redis
            ttl_ms = int(ttl * 1000) if ttl is not None else None
            
            # Set the value with TTL
            await self._redis_client.set(
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
            await self._redis_client.hset(f"{prompt}:meta", mapping=metadata)
            
        except (RedisError, GoogleAPICallError) as e:
            logger.error(f"Error adding item to Google Memorystore: {e}")
            raise CacheOperationError("Failed to add item to Google Memorystore", original_exception=e) from e
            
    async def delete(self, key: str, mode: DeletionMode = DeletionMode.DIRECT) -> bool:
        """
        Delete a key from the Google Memorystore cache.
        
        Args:
            key: The key to delete
            mode: Deletion mode (DIRECT or TOMBSTONE)
            
        Returns:
            bool: True if the key was found and deleted, False otherwise
        """
        if mode == DeletionMode.TOMBSTONE:
            logger.info(f"Tombstoning requested for Google Memorystore, using direct deletion for key: {key}")
            # Fall back to direct deletion for external stores
            return await self._direct_delete(key)
        
        return await self._direct_delete(key)
    
    async def _direct_delete(self, key: str) -> bool:
        """Direct deletion implementation."""
        
        try:
            await self._ensure_connected()
            if not self._redis_client:
                return False
                
            # Delete both the key and its metadata
            pipeline = self._redis_client.pipeline()
            pipeline.delete(key)
            pipeline.delete(f"{key}:meta")
            results = await pipeline.execute()
            
            # If either the key or its metadata was deleted, return True
            return any(results)
            
        except (RedisError, GoogleAPICallError) as e:
            logger.error(f"Error deleting key from Google Memorystore: {e}")
            raise CacheOperationError(f"Failed to delete key from Google Memorystore: {key}", original_exception=e) from e
    
    async def tombstone(self, key: str) -> bool:
        """
        Mark a key as deleted (tombstoning) without removing it from storage.
        
        Args:
            key: The key to tombstone
            
        Returns:
            bool: True if the key was found and tombstoned, False otherwise
        """
        try:
            await self._ensure_connected()
            if not self._redis_client:
                return False
                
            # Check if key exists
            exists = await self._redis_client.exists(key)
            if not exists:
                return False
                
            # Add tombstone flag to metadata
            metadata = {
                "tombstoned": "1",
                "tombstoned_at": str(time.time())
            }
            await self._redis_client.hset(f"{key}:meta", mapping=metadata)
            
            return True
            
        except (RedisError, GoogleAPICallError) as e:
            logger.error(f"Error tombstoning key in Google Memorystore: {e}")
            raise CacheOperationError(f"Failed to tombstone key in Google Memorystore: {key}", original_exception=e) from e
    
    async def is_tombstoned(self, key: str) -> bool:
        """
        Check if a key is tombstoned.
        
        Args:
            key: The key to check
            
        Returns:
            bool: True if the key is tombstoned, False otherwise
        """
        try:
            await self._ensure_connected()
            if not self._redis_client:
                return False
                
            tombstoned = await self._redis_client.hget(f"{key}:meta", "tombstoned")
            return tombstoned == "1"
            
        except (RedisError, GoogleAPICallError) as e:
            logger.error(f"Error checking tombstone status in Google Memorystore: {e}")
            return False
    
    async def purge_tombstones(self) -> int:
        """
        Permanently remove all tombstoned keys from storage.
        
        Returns:
            int: Number of tombstoned keys that were purged
        """
        try:
            await self._ensure_connected()
            if not self._redis_client:
                return 0
                
            # Get all keys
            keys = await self._redis_client.keys("*")
            cache_keys = [k for k in keys if not k.endswith(":meta") and ":" not in k]
            
            purged_count = 0
            pipeline = self._redis_client.pipeline()
            
            for key in cache_keys:
                # Check if this key is tombstoned
                tombstoned = await self._redis_client.hget(f"{key}:meta", "tombstoned")
                if tombstoned == "1":
                    # Delete both the key and its metadata
                    pipeline.delete(key)
                    pipeline.delete(f"{key}:meta")
                    purged_count += 1
            
            if purged_count > 0:
                await pipeline.execute()
            
            return purged_count
            
        except (RedisError, GoogleAPICallError) as e:
            logger.error(f"Error purging tombstones from Google Memorystore: {e}")
            raise CacheOperationError("Failed to purge tombstones from Google Memorystore", original_exception=e) from e
    
    async def clear(self) -> None:
        """Clear all items from the cache."""
        try:
            await self._ensure_connected()
            if self._redis_client:
                await self._redis_client.flushdb()
        except (RedisError, GoogleAPICallError) as e:
            logger.error(f"Error clearing Google Memorystore: {e}")
            raise CacheOperationError("Failed to clear Google Memorystore", original_exception=e) from e
    
    async def size(self) -> int:
        """Get the number of items in the cache."""
        try:
            await self._ensure_connected()
            if not self._redis_client:
                return 0
                
            # Get total keys (excluding metadata)
            keys = await self._redis_client.keys("*")
            # Filter out metadata keys
            cache_keys = [k for k in keys if not k.endswith(":meta") and ":" not in k]
            return len(cache_keys)
            
        except (RedisError, GoogleAPICallError) as e:
            logger.error(f"Error getting Google Memorystore size: {e}")
            raise CacheOperationError("Failed to get Google Memorystore size", original_exception=e) from e
    
    async def enforce_limits(self, resource_limits: Any) -> None:
        """Enforce cache size limits."""
        try:
            await self._ensure_connected()
            if not self._redis_client:
                return
                
            current_size = await self.size()
            max_size = getattr(resource_limits, 'max_size', None)
            
            if max_size is not None and current_size > max_size:
                await self.eviction_policy.apply(self, max_size)
                
        except (RedisError, GoogleAPICallError) as e:
            logger.error(f"Error enforcing Google Memorystore limits: {e}")
            raise CacheOperationError("Failed to enforce Google Memorystore limits", original_exception=e) from e
    
    def get_eviction_policy(self) -> EvictionPolicy:
        """Get the eviction policy for this cache store."""
        return self.eviction_policy
    
    async def close(self) -> None:
        """Close the Google Memorystore connection."""
        if self._redis_client:
            try:
                await self._redis_client.close()
                self._connected = False
                logger.info("Closed Google Memorystore connection")
            except RedisError as e:
                logger.error(f"Error closing Google Memorystore connection: {e}")
                raise CacheOperationError("Failed to close Google Memorystore connection", original_exception=e) from e
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            await self._ensure_connected()
            if not self._redis_client or not self._instance:
                return {"error": "Not connected to Google Memorystore"}
                
            # Get Redis info
            info = await self._redis_client.info()
            
            stats = {
                "backend": "google_memorystore",
                "instance_id": self.instance_id,
                "project_id": self.project_id,
                "region": self.region,
                "connected": self._connected,
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
                "instance_tier": self._instance.tier.name,
                "memory_size_gb": self._instance.memory_size_gb,
                "redis_version": self._instance.redis_version,
                "authorized_network": self._instance.authorized_network.split("/")[-1],
                "create_time": self._instance.create_time.isoformat(),
            }
            
            return stats
            
        except (RedisError, GoogleAPICallError) as e:
            logger.error(f"Error getting Google Memorystore stats: {e}")
            raise CacheOperationError("Failed to get Google Memorystore stats", original_exception=e) from e
    
    def __del__(self) -> None:
        """Ensure connection is closed when the object is garbage collected."""
        if hasattr(self, '_redis_client') and self._redis_client:
            try:
                asyncio.get_event_loop().run_until_complete(self.close())
            except Exception:
                pass

    @classmethod
    async def list_instances(
        cls, 
        project_id: str, 
        region: str,
        show_deleted: bool = False
    ) -> List[Dict[str, Any]]:
        """List all Memorystore instances in the specified project and region.
        
        Args:
            project_id: Google Cloud project ID
            region: Google Cloud region
            show_deleted: Whether to include deleted instances
            
        Returns:
            List of instance information dictionaries
        """
        try:
            client = redis_v1.CloudRedisClient()
            parent = f"projects/{project_id}/locations/{region}"
            
            # List instances
            instances = []
            page_result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.list_instances(parent=parent)
            )
            
            for instance in page_result.instances:
                if not show_deleted and instance.state != Instance.State.RUNNING:
                    continue
                    
                instances.append({
                    "name": instance.name.split("/")[-1],
                    "display_name": instance.display_name,
                    "state": instance.State(instance.state).name,
                    "host": instance.host,
                    "port": instance.port,
                    "memory_size_gb": instance.memory_size_gb,
                    "tier": instance.tier.name,
                    "redis_version": instance.redis_version,
                    "create_time": instance.create_time.isoformat(),
                })
            
            return instances
            
        except (GoogleAPICallError, RetryError) as e:
            logger.error(f"Error listing Google Memorystore instances: {e}")
            raise CacheOperationError("Failed to list Google Memorystore instances", original_exception=e) from e
