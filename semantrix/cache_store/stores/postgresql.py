"""
PostgreSQL cache store implementation for Semantrix.

Provides a persistent, SQL-based cache using asyncpg with PostgreSQL.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union

import asyncpg
from asyncpg import Connection, Pool, create_pool
from asyncpg.pool import PoolAcquireContext

from semantrix.cache_store.base import BaseCacheStore, EvictionPolicy, NoOpEvictionPolicy, DeletionMode
from semantrix.exceptions import CacheOperationError

# Configure logging
logger = logging.getLogger(__name__)

class PostgreSQLCacheStore(BaseCacheStore):
    """
    PostgreSQL-based cache store implementation.

    Features:
    - Persistent storage in PostgreSQL
    - Connection pooling
    - JSONB for flexible metadata storage
    - Automatic table and index creation
    - TTL support
    - Full-text search capabilities

    Args:
        dsn: PostgreSQL connection string
        table_name: Name of the table to use (default: 'semantrix_cache')
        eviction_policy: Eviction policy to use
        ttl_seconds: Default TTL in seconds for cache entries (None for no expiration)
        pool_min_size: Minimum number of connections in the pool
        pool_max_size: Maximum number of connections in the pool
        **kwargs: Additional keyword arguments for BaseCacheStore
    """

    def __init__(
        self,
        dsn: str = "postgresql://postgres:postgres@localhost:5432/semantrix_cache",
        table_name: str = "semantrix_cache",
        eviction_policy: Optional[EvictionPolicy] = None,
        ttl_seconds: Optional[int] = 3600,  # 1 hour default TTL
        pool_min_size: int = 1,
        pool_max_size: int = 10,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.dsn = dsn
        self.table_name = table_name
        self.eviction_policy = eviction_policy or NoOpEvictionPolicy()
        self.ttl_seconds = ttl_seconds
        self.pool_min_size = pool_min_size
        self.pool_max_size = pool_max_size
        self._pool: Optional[Pool] = None
        self._lock = asyncio.Lock()
        self._connected = False

    async def _ensure_connected(self) -> None:
        """Ensure we have an active connection pool."""
        if self._connected and self._pool:
            try:
                async with self._pool.acquire() as conn:
                    await conn.execute("SELECT 1")
                return
            except asyncpg.PostgresError as e:
                self._connected = False
                if self._pool:
                    await self._pool.close()
                raise CacheOperationError("Failed to verify PostgreSQL connection", original_exception=e) from e
        
        async with self._lock:
            if not self._connected:
                try:
                    self._pool = await create_pool(
                        dsn=self.dsn,
                        min_size=self.pool_min_size,
                        max_size=self.pool_max_size,
                        command_timeout=30,
                        server_settings={
                            'application_name': 'semantrix-cache',
                            'statement_timeout': '30000',  # 30 seconds
                        },
                    )
                    
                    # Create tables and indexes
                    await self._initialize_schema()
                    self._connected = True
                    logger.info(f"Connected to PostgreSQL at {self.dsn}")
                    
                except asyncpg.PostgresError as e:
                    self._connected = False
                    logger.error(f"Failed to connect to PostgreSQL: {e}")
                    raise CacheOperationError("Failed to connect to PostgreSQL", original_exception=e) from e

    async def _initialize_schema(self) -> None:
        """Initialize database tables and indexes."""
        if not self._pool:
            return
            
        async with self._pool.acquire() as conn:
            # Enable UUID extension if not exists
            await conn.execute("""
                CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
            """)
            
            # Create cache table
            await conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                    expires_at TIMESTAMP WITH TIME ZONE,
                    last_accessed_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
                    access_count INTEGER NOT NULL DEFAULT 1,
                    metadata JSONB,
                    tombstoned BOOLEAN NOT NULL DEFAULT FALSE,
                    tombstoned_at TIMESTAMP WITH TIME ZONE,
                    
                    -- Indexes
                    CONSTRAINT {self.table_name}_key_unique UNIQUE (key)
                );
                
                -- Index for faster lookups
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_key 
                ON {self.table_name} (key);
                
                -- Index for TTL cleanup
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_expires 
                ON {self.table_name} (expires_at) 
                WHERE expires_at IS NOT NULL;
                
                -- Index for full-text search if needed
                CREATE INDEX IF NOT EXISTS idx_{self.table_name}_fts 
                ON {self.table_name} USING GIN (to_tsvector('english', key));
            """)
            
            # Add any missing columns (for schema migrations)
            await self._migrate_schema(conn)

    async def _migrate_schema(self, conn: Connection) -> None:
        """Handle schema migrations."""
        try:
            # Example migration: Add metadata column if it doesn't exist
            await conn.execute(f"""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 
                        FROM information_schema.columns 
                        WHERE table_name = '{self.table_name}' 
                        AND column_name = 'metadata'
                    ) THEN
                        ALTER TABLE {self.table_name} 
                        ADD COLUMN metadata JSONB;
                    END IF;
                END $$;
            """)
        except asyncpg.PostgresError as e:
            logger.warning(f"Schema migration failed: {e}")

    async def get_exact(self, prompt: str) -> Optional[str]:
        """Get a cached response if it exists and is not expired."""
        try:
            await self._ensure_connected()
            if not self._pool:
                return None
                
            async with self._pool.acquire() as conn:
                # Use NOW() AT TIME ZONE 'UTC' for consistent time comparison
                row = await conn.fetchrow(
                    f"""
                    UPDATE {self.table_name}
                    SET 
                        last_accessed_at = NOW() AT TIME ZONE 'UTC',
                        access_count = access_count + 1
                    WHERE 
                        key = $1 AND 
                        (expires_at IS NULL OR expires_at > NOW() AT TIME ZONE 'UTC') AND
                        tombstoned = FALSE
                    RETURNING value
                    """,
                    prompt
                )
                
                return row['value'] if row else None
                
        except asyncpg.PostgresError as e:
            logger.error(f"Error getting item from PostgreSQL cache: {e}")
            raise CacheOperationError(f"Failed to get item from PostgreSQL for key: {prompt}", original_exception=e) from e

    async def add(self, prompt: str, response: str, ttl: Optional[float] = None) -> None:
        """Add a response to the cache."""
        try:
            await self._ensure_connected()
            if not self._pool:
                raise CacheOperationError("PostgreSQL connection pool not available")
                
            expires_at = (
                f"(NOW() AT TIME ZONE 'UTC' + INTERVAL '{ttl} seconds')" 
                if ttl is not None 
                else (
                    f"(NOW() AT TIME ZONE 'UTC' + INTERVAL '{self.ttl_seconds} seconds')" 
                    if self.ttl_seconds is not None 
                    else "NULL"
                )
            )
            
            async with self._pool.acquire() as conn:
                await conn.execute(
                    f"""
                    INSERT INTO {self.table_name} 
                        (key, value, expires_at, last_accessed_at)
                    VALUES 
                        ($1, $2, {expires_at}, NOW() AT TIME ZONE 'UTC')
                    ON CONFLICT (key) 
                    DO UPDATE SET
                        value = EXCLUDED.value,
                        expires_at = EXCLUDED.expires_at,
                        last_accessed_at = EXCLUDED.last_accessed_at,
                        access_count = {self.table_name}.access_count + 1
                    """,
                    prompt, response
                )
                
        except asyncpg.PostgresError as e:
            logger.error(f"Error adding item to PostgreSQL cache: {e}")
            raise CacheOperationError(f"Failed to add item to PostgreSQL for key: {prompt}", original_exception=e) from e
            
    async def delete(self, key: str, mode: DeletionMode = DeletionMode.DIRECT) -> bool:
        """
        Delete a key from the PostgreSQL cache.
        
        Args:
            key: The key to delete
            mode: Deletion mode (DIRECT or TOMBSTONE)
            
        Returns:
            bool: True if the key was found and deleted, False otherwise
        """
        if mode == DeletionMode.TOMBSTONE:
            logger.info(f"Tombstoning requested for PostgreSQL store, using direct deletion for key: {key}")
            # Fall back to direct deletion for external stores
            return await self._direct_delete(key)
        
        return await self._direct_delete(key)
    
    async def _direct_delete(self, key: str) -> bool:
        """Direct deletion implementation."""
        
        try:
            await self._ensure_connected()
            if not self._pool:
                return False
                
            async with self._pool.acquire() as conn:
                result = await conn.execute(
                    f"""
                    DELETE FROM {self.table_name}
                    WHERE key = $1
                    RETURNING key
                    """,
                    key
                )
                
                # If rows were affected, the key existed and was deleted
                return result.split()[-1] == '1'  # Returns 'DELETE 1' if successful
                
        except asyncpg.PostgresError as e:
            logger.error(f"Error deleting key from PostgreSQL cache: {e}")
            raise CacheOperationError(f"Failed to delete key from PostgreSQL: {key}", original_exception=e) from e
    
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
            if not self._pool:
                return False
                
            async with self._pool.acquire() as conn:
                result = await conn.execute(
                    f"""
                    UPDATE {self.table_name}
                    SET 
                        tombstoned = TRUE,
                        tombstoned_at = NOW() AT TIME ZONE 'UTC'
                    WHERE key = $1
                    RETURNING key
                    """,
                    key
                )
                
                # If rows were affected, the key existed and was tombstoned
                return result.split()[-1] == '1'  # Returns 'UPDATE 1' if successful
                
        except asyncpg.PostgresError as e:
            logger.error(f"Error tombstoning key in PostgreSQL cache: {e}")
            raise CacheOperationError(f"Failed to tombstone key in PostgreSQL: {key}", original_exception=e) from e
    
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
            if not self._pool:
                return False
                
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"""
                    SELECT tombstoned 
                    FROM {self.table_name}
                    WHERE key = $1
                    """,
                    key
                )
                
                return row['tombstoned'] if row else False
                
        except asyncpg.PostgresError as e:
            logger.error(f"Error checking tombstone status in PostgreSQL cache: {e}")
            return False
    
    async def purge_tombstones(self) -> int:
        """
        Permanently remove all tombstoned keys from storage.
        
        Returns:
            int: Number of tombstoned keys that were purged
        """
        try:
            await self._ensure_connected()
            if not self._pool:
                return 0
                
            async with self._pool.acquire() as conn:
                # Count tombstoned items before deletion
                count_row = await conn.fetchrow(
                    f"""
                    SELECT COUNT(*) as count
                    FROM {self.table_name}
                    WHERE tombstoned = TRUE
                    """
                )
                count = count_row['count'] if count_row else 0
                
                # Delete tombstoned items
                await conn.execute(
                    f"""
                    DELETE FROM {self.table_name}
                    WHERE tombstoned = TRUE
                    """
                )
                
                return count
                
        except asyncpg.PostgresError as e:
            logger.error(f"Error purging tombstones from PostgreSQL cache: {e}")
            raise CacheOperationError("Failed to purge tombstones from PostgreSQL", original_exception=e) from e

    async def clear(self) -> None:
        """Clear all items from the cache."""
        try:
            await self._ensure_connected()
            if self._pool:
                async with self._pool.acquire() as conn:
                    await conn.execute(f"TRUNCATE TABLE {self.table_name}")
        except asyncpg.PostgresError as e:
            logger.error(f"Error clearing PostgreSQL cache: {e}")
            raise CacheOperationError("Failed to clear PostgreSQL cache", original_exception=e) from e

    async def size(self) -> int:
        """Get the number of items in the cache."""
        try:
            await self._ensure_connected()
            if not self._pool:
                return 0
                
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    f"""
                    SELECT COUNT(*) as count 
                    FROM {self.table_name}
                    WHERE expires_at IS NULL OR expires_at > NOW() AT TIME ZONE 'UTC'
                    """
                )
                return row['count'] if row else 0
                
        except asyncpg.PostgresError as e:
            logger.error(f"Error getting PostgreSQL cache size: {e}")
            raise CacheOperationError("Failed to get PostgreSQL cache size", original_exception=e) from e

    async def enforce_limits(self, resource_limits: Any) -> None:
        """Enforce cache size limits."""
        try:
            await self._ensure_connected()
            if not self._pool:
                return
                
            # Clean up expired entries first
            async with self._pool.acquire() as conn:
                await conn.execute(
                    f"""
                    DELETE FROM {self.table_name}
                    WHERE expires_at IS NOT NULL 
                    AND expires_at <= NOW() AT TIME ZONE 'UTC'
                    """
                )
                
                # Apply eviction policy if needed
                current_size = await self.size()
                max_size = getattr(resource_limits, 'max_size', None)
                
                if max_size is not None and current_size > max_size:
                    await self.eviction_policy.apply(self, max_size)
                    
        except asyncpg.PostgresError as e:
            logger.error(f"Error enforcing PostgreSQL cache limits: {e}")
            raise CacheOperationError("Failed to enforce PostgreSQL cache limits", original_exception=e) from e

    def get_eviction_policy(self) -> EvictionPolicy:
        """Get the eviction policy for this cache store."""
        return self.eviction_policy

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            try:
                await self._pool.close()
                self._connected = False
                logger.info("Closed PostgreSQL connection pool")
            except asyncpg.PostgresError as e:
                logger.error(f"Error closing PostgreSQL connection pool: {e}")
                raise CacheOperationError("Failed to close PostgreSQL connection pool", original_exception=e) from e

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            await self._ensure_connected()
            if not self._pool:
                return {"error": "Not connected to PostgreSQL"}
                
            async with self._pool.acquire() as conn:
                # Get basic stats
                stats = {
                    "backend": "postgresql",
                    "table": self.table_name,
                    "connected": self._connected,
                    "ttl_seconds": self.ttl_seconds,
                }
                
                # Get row count
                row = await conn.fetchrow(
                    f"""
                    SELECT 
                        COUNT(*) as total_count,
                        COUNT(CASE WHEN expires_at <= NOW() AT TIME ZONE 'UTC' THEN 1 END) as expired_count,
                        pg_size_pretty(pg_total_relation_size('{self.table_name}')) as size
                    FROM {self.table_name}
                    """
                )
                
                if row:
                    stats.update({
                        "total_entries": row["total_count"],
                        "expired_entries": row["expired_count"],
                        "table_size": row["size"],
                    })
                
                return stats
                
        except asyncpg.PostgresError as e:
            logger.error(f"Error getting PostgreSQL cache stats: {e}")
            raise CacheOperationError("Failed to get PostgreSQL stats", original_exception=e) from e

    def __del__(self) -> None:
        """Ensure connection pool is closed when the object is garbage collected."""
        if hasattr(self, '_pool') and self._pool:
            try:
                asyncio.get_event_loop().run_until_complete(self._pool.close())
            except Exception:
                pass
