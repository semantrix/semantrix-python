"""
MongoDB cache store implementation for Semantrix.

Provides a persistent, document-based cache using MongoDB with async support.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from pymongo import MongoClient, errors
from pymongo.database import Database
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from semantrix.cache_store.base import BaseCacheStore, EvictionPolicy, NoOpEvictionPolicy

# Configure logging
logger = logging.getLogger(__name__)

class MongoDBCacheStore(BaseCacheStore):
    """
    MongoDB-based cache store implementation.

    Features:
    - Persistent storage in MongoDB
    - Built-in TTL index support
    - Connection pooling
    - Document-based storage with flexible schema
    - Automatic collection creation and indexing

    Args:
        connection_string: MongoDB connection string (default: 'mongodb://localhost:27017/')
        db_name: Name of the database to use (default: 'semantrix_cache')
        collection_name: Name of the collection to use (default: 'cache')
        eviction_policy: Eviction policy to use
        ttl_seconds: Default TTL in seconds for cache entries (None for no expiration)
        **kwargs: Additional keyword arguments for BaseCacheStore
    """

    def __init__(
        self,
        connection_string: str = "mongodb://localhost:27017/",
        db_name: str = "semantrix_cache",
        collection_name: str = "cache",
        eviction_policy: Optional[EvictionPolicy] = None,
        ttl_seconds: Optional[int] = 3600,  # 1 hour default TTL
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.connection_string = connection_string
        self.db_name = db_name
        self.collection_name = collection_name
        self.eviction_policy = eviction_policy or NoOpEvictionPolicy()
        self.ttl_seconds = ttl_seconds
        
        # Will be initialized in _ensure_connected
        self._client: Optional[MongoClient] = None
        self._db: Optional[Database] = None
        self._collection: Optional[Collection] = None
        self._connected = False
        self._lock = asyncio.Lock()

    async def _ensure_connected(self) -> None:
        """Ensure we have an active MongoDB connection."""
        if self._connected and self._client is not None:
            try:
                # Ping the server to check connection
                await asyncio.get_event_loop().run_in_executor(
                    None, self._client.admin.command, 'ping'
                )
                return
            except PyMongoError:
                # Connection lost, will reconnect
                self._connected = False
                if self._client:
                    self._client.close()
        
        async with self._lock:
            if not self._connected:
                try:
                    # Create a new client with connection pooling
                    self._client = MongoClient(
                        self.connection_string,
                        connectTimeoutMS=5000,
                        socketTimeoutMS=30000,
                        serverSelectionTimeoutMS=5000,
                        maxPoolSize=100,
                        minPoolSize=1,
                        retryWrites=True,
                        retryReads=True
                    )
                    
                    # Test the connection
                    await asyncio.get_event_loop().run_in_executor(
                        None, self._client.admin.command, 'ping'
                    )
                    
                    self._db = self._client[self.db_name]
                    self._collection = self._db[self.collection_name]
                    
                    # Create TTL index if TTL is enabled
                    if self.ttl_seconds is not None:
                        await self._ensure_ttl_index()
                    
                    self._connected = True
                    logger.info(f"Connected to MongoDB at {self.connection_string}")
                    
                except PyMongoError as e:
                    self._connected = False
                    logger.error(f"Failed to connect to MongoDB: {e}")
                    raise

    async def _ensure_ttl_index(self) -> None:
        """Ensure TTL index exists on the collection."""
        if self._collection is None:
            return
            
        try:
            # Check if TTL index already exists
            indexes = await asyncio.get_event_loop().run_in_executor(
                None, self._collection.index_information
            )
            
            ttl_index_exists = any(
                'expireAfterSeconds' in idx_info 
                for idx_info in indexes.values()
            )
            
            if not ttl_index_exists and self.ttl_seconds is not None:
                # Create TTL index on 'expires_at' field
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._collection.create_index(
                        [("expires_at", 1)],
                        expireAfterSeconds=0,  # Documents expire when expires_at < current time
                        background=True
                    )
                )
                logger.debug(f"Created TTL index on 'expires_at' field")
                
        except PyMongoError as e:
            logger.error(f"Error ensuring TTL index: {e}")
            raise

    async def get_exact(self, prompt: str) -> Optional[str]:
        """Get a cached response if it exists and is not expired."""
        try:
            await self._ensure_connected()
            if not self._collection:
                return None
                
            # Find the document by key
            doc = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._collection.find_one(
                    {"key": prompt, "$or": [
                        {"expires_at": {"$exists": False}},
                        {"expires_at": {"$gt": datetime.utcnow()}}
                    ]}
                )
            )
            
            if doc:
                # Update last accessed time
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._collection.update_one(
                        {"_id": doc["_id"]},
                        {
                            "$inc": {"access_count": 1},
                            "$set": {"last_accessed_at": datetime.utcnow()}
                        }
                    )
                )
                return doc.get("value")
                
            return None
            
        except PyMongoError as e:
            logger.error(f"Error getting item from MongoDB cache: {e}")
            return None

    async def add(self, prompt: str, response: str, ttl: Optional[float] = None) -> None:
        """Add a response to the cache."""
        try:
            await self._ensure_connected()
            if not self._collection:
                raise RuntimeError("MongoDB collection not available")
                
            now = datetime.utcnow()
            expires_at = (
                now + timedelta(seconds=ttl) 
                if ttl is not None 
                else (now + timedelta(seconds=self.ttl_seconds) 
                      if self.ttl_seconds is not None 
                      else None)
            )
            
            document = {
                "key": prompt,
                "value": response,
                "created_at": now,
                "last_accessed_at": now,
                "access_count": 1,
            }
            
            if expires_at:
                document["expires_at"] = expires_at
            
            # Use upsert to update if exists, insert if not
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._collection.update_one(
                    {"key": prompt},
                    {"$set": document},
                    upsert=True
                )
            )
            
        except PyMongoError as e:
            logger.error(f"Error adding item to MongoDB cache: {e}")
            raise

    async def clear(self) -> None:
        """Clear all items from the cache."""
        try:
            await self._ensure_connected()
            if self._collection:
                await asyncio.get_event_loop().run_in_executor(
                    None, self._collection.drop
                )
                # Recreate the collection and indexes
                self._collection = self._db[self.collection_name]
                if self.ttl_seconds is not None:
                    await self._ensure_ttl_index()
        except PyMongoError as e:
            logger.error(f"Error clearing MongoDB cache: {e}")
            raise

    async def size(self) -> int:
        """Get the number of items in the cache."""
        try:
            await self._ensure_connected()
            if not self._collection:
                return 0
                
            return await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._collection.count_documents({
                    "$or": [
                        {"expires_at": {"$exists": False}},
                        {"expires_at": {"$gt": datetime.utcnow()}}
                    ]
                })
            )
            
        except PyMongoError as e:
            logger.error(f"Error getting MongoDB cache size: {e}")
            return 0

    async def enforce_limits(self, resource_limits: Any) -> None:
        """Enforce cache size limits."""
        try:
            await self._ensure_connected()
            if not self._collection:
                return
                
            # MongoDB TTL index will handle expired documents
            # Apply eviction policy if needed
            current_size = await self.size()
            max_size = getattr(resource_limits, 'max_size', None)
            
            if max_size is not None and current_size > max_size:
                await self.eviction_policy.apply(self, max_size)
                
        except PyMongoError as e:
            logger.error(f"Error enforcing MongoDB cache limits: {e}")
            raise

    def get_eviction_policy(self) -> EvictionPolicy:
        """Get the eviction policy for this cache store."""
        return self.eviction_policy

    async def close(self) -> None:
        """Close the MongoDB connection."""
        if self._client:
            try:
                self._client.close()
                self._connected = False
                logger.info("Closed MongoDB connection")
            except Exception as e:
                logger.error(f"Error closing MongoDB connection: {e}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            await self._ensure_connected()
            if not self._collection:
                return {"error": "Not connected to MongoDB"}
                
            stats = {
                "backend": "mongodb",
                "db": self.db_name,
                "collection": self.collection_name,
                "connected": self._connected,
                "ttl_seconds": self.ttl_seconds,
            }
            
            if self._db:
                # Get collection stats
                stats.update(await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._db.command("collstats", self.collection_name)
                ))
                
                # Get document count (excluding expired)
                stats["active_documents"] = await self.size()
                
            return stats
            
        except PyMongoError as e:
            logger.error(f"Error getting MongoDB cache stats: {e}")
            return {"error": str(e)}

    def __del__(self) -> None:
        """Ensure connection is closed when the object is garbage collected."""
        if hasattr(self, '_client') and self._client:
            try:
                self._client.close()
            except Exception:
                pass
