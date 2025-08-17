"""
SQLite cache store implementation for Semantrix.

Provides a persistent, file-based cache using SQLite with async support.
"""

import asyncio
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

from semantrix.cache_store.base import BaseCacheStore, EvictionPolicy, NoOpEvictionPolicy, DeletionMode
from semantrix.exceptions import CacheOperationError

# Configure logging
logger = logging.getLogger(__name__)

class SQLiteCacheStore(BaseCacheStore):
    """
    SQLite-based cache store implementation.

    Features:
    - Persistent storage in a single SQLite file
    - Thread-safe operations
    - TTL support
    - Configurable database path
    - Automatic table creation
    - Connection pooling

    Args:
        db_path: Path to SQLite database file (default: ':memory:' for in-memory)
        table_name: Name of the table to use for cache storage
        eviction_policy: Eviction policy to use
        max_connections: Maximum number of database connections in the pool
        **kwargs: Additional keyword arguments for BaseCacheStore
    """

    def __init__(
        self,
        db_path: Union[str, Path] = ":memory:",
        table_name: str = "semantrix_cache",
        eviction_policy: Optional[EvictionPolicy] = None,
        max_connections: int = 5,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.db_path = str(db_path)
        self.table_name = table_name
        self.eviction_policy = eviction_policy or NoOpEvictionPolicy()
        self.max_connections = max_connections
        self._connection_pool = []
        self._connection_lock = asyncio.Lock()
        self._initialized = False
        self._closed = False

    async def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection from the pool or create a new one."""
        async with self._connection_lock:
            if self._connection_pool:
                return self._connection_pool.pop()
            
            if len(self._connection_pool) >= self.max_connections:
                # Wait for a connection to become available
                while not self._connection_pool:
                    await asyncio.sleep(0.1)
                return self._connection_pool.pop()
            
            # Create a new connection
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            
            # Initialize the database if needed
            if not self._initialized:
                await self._initialize_db(conn)
                self._initialized = True
                
            return conn

    def _return_connection(self, conn: sqlite3.Connection) -> None:
        """Return a connection to the pool."""
        if not self._closed:
            self._connection_pool.append(conn)

    async def _initialize_db(self, conn: sqlite3.Connection) -> None:
        """Initialize the database tables if they don't exist."""
        cursor = conn.cursor()
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL,
            created_at REAL NOT NULL,
            expires_at REAL,
            metadata TEXT,
            access_count INTEGER DEFAULT 0,
            last_accessed_at REAL NOT NULL,
            tombstoned INTEGER DEFAULT 0,
            tombstoned_at REAL
        )
        """)
        
        # Create indexes for faster lookups
        cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_expires 
        ON {self.table_name}(expires_at)
        """)
        
        cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_{self.table_name}_last_accessed 
        ON {self.table_name}(last_accessed_at)
        """)
        
        conn.commit()

    async def get_exact(self, prompt: str) -> Optional[str]:
        """Get a cached response if it exists and is not expired."""
        if self._closed:
            raise CacheOperationError("Cache store is closed")
            
        conn = await self._get_connection()
        try:
            cursor = conn.cursor()
            now = time.time()
            
            cursor.execute(
                f"""
                SELECT value, expires_at 
                FROM {self.table_name} 
                WHERE key = ? AND (expires_at IS NULL OR expires_at > ?) AND tombstoned = 0
                """,
                (prompt, now)
            )
            
            if row := cursor.fetchone():
                # Update access time and count
                cursor.execute(
                    f"""
                    UPDATE {self.table_name} 
                    SET access_count = access_count + 1, last_accessed_at = ?
                    WHERE key = ?
                    """,
                    (now, prompt)
                )
                conn.commit()
                return row[0]
            return None
        except sqlite3.Error as e:
            raise CacheOperationError("Failed to get item from SQLite cache", original_exception=e) from e
        finally:
            self._return_connection(conn)

    async def add(self, prompt: str, response: str, ttl: Optional[float] = None) -> None:
        """Add a response to the cache."""
        if self._closed:
            raise CacheOperationError("Cache store is closed")
            
        conn = await self._get_connection()
        try:
            cursor = conn.cursor()
            now = time.time()
            expires_at = now + ttl if ttl is not None else None
            
            cursor.execute(
                f"""
                INSERT OR REPLACE INTO {self.table_name} 
                (key, value, created_at, expires_at, last_accessed_at, access_count)
                VALUES (?, ?, ?, ?, ?, 1)
                """,
                (prompt, response, now, expires_at, now)
            )
            conn.commit()
        except sqlite3.Error as e:
            raise CacheOperationError("Failed to add item to SQLite cache", original_exception=e) from e
        finally:
            self._return_connection(conn)
            
    async def delete(self, key: str, mode: DeletionMode = DeletionMode.DIRECT) -> bool:
        """
        Delete a key from the SQLite cache.
        
        Args:
            key: The key to delete
            mode: Deletion mode (DIRECT or TOMBSTONE)
            
        Returns:
            bool: True if the key was found and deleted, False otherwise
        """
        if mode == DeletionMode.TOMBSTONE:
            logger.info(f"Tombstoning requested for SQLite store, using direct deletion for key: {key}")
            # Fall back to direct deletion for external stores
            return await self._direct_delete(key)
        
        return await self._direct_delete(key)
    
    async def _direct_delete(self, key: str) -> bool:
        """Direct deletion implementation."""
        
        if self._closed:
            raise CacheOperationError("Cache store is closed")
            
        conn = await self._get_connection()
        try:
            cursor = conn.cursor()
            
            # First check if the key exists
            cursor.execute(
                f"SELECT 1 FROM {self.table_name} WHERE key = ?",
                (key,)
            )
            exists = cursor.fetchone() is not None
            
            if not exists:
                return False
                
            # Delete the key
            cursor.execute(
                f"DELETE FROM {self.table_name} WHERE key = ?",
                (key,)
            )
            conn.commit()
            return True
            
        except sqlite3.Error as e:
            raise CacheOperationError(f"Failed to delete key '{key}' from SQLite cache", original_exception=e) from e
        finally:
            self._return_connection(conn)
    
    async def tombstone(self, key: str) -> bool:
        """
        Mark a key as deleted (tombstoning) without removing it from storage.
        
        Args:
            key: The key to tombstone
            
        Returns:
            bool: True if the key was found and tombstoned, False otherwise
        """
        if self._closed:
            raise CacheOperationError("Cache store is closed")
            
        conn = await self._get_connection()
        try:
            cursor = conn.cursor()
            
            # First check if the key exists
            cursor.execute(
                f"SELECT 1 FROM {self.table_name} WHERE key = ?",
                (key,)
            )
            exists = cursor.fetchone() is not None
            
            if not exists:
                return False
                
            # Mark as tombstoned
            cursor.execute(
                f"UPDATE {self.table_name} SET tombstoned = 1, tombstoned_at = ? WHERE key = ?",
                (time.time(), key)
            )
            conn.commit()
            return True
            
        except sqlite3.Error as e:
            raise CacheOperationError(f"Failed to tombstone key '{key}' in SQLite cache", original_exception=e) from e
        finally:
            self._return_connection(conn)
    
    async def is_tombstoned(self, key: str) -> bool:
        """
        Check if a key is tombstoned.
        
        Args:
            key: The key to check
            
        Returns:
            bool: True if the key is tombstoned, False otherwise
        """
        if self._closed:
            raise CacheOperationError("Cache store is closed")
            
        conn = await self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"SELECT tombstoned FROM {self.table_name} WHERE key = ?",
                (key,)
            )
            result = cursor.fetchone()
            return result is not None and result[0] == 1
            
        except sqlite3.Error as e:
            logger.error(f"SQLite error checking tombstone status for key '{key}': {e}")
            return False
        finally:
            self._return_connection(conn)
    
    async def purge_tombstones(self) -> int:
        """
        Permanently remove all tombstoned keys from storage.
        
        Returns:
            int: Number of tombstoned keys that were purged
        """
        if self._closed:
            raise CacheOperationError("Cache store is closed")
            
        conn = await self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Count tombstoned items before deletion
            cursor.execute(f"SELECT COUNT(*) FROM {self.table_name} WHERE tombstoned = 1")
            count = cursor.fetchone()[0] or 0
            
            # Delete tombstoned items
            cursor.execute(f"DELETE FROM {self.table_name} WHERE tombstoned = 1")
            conn.commit()
            
            return count
            
        except sqlite3.Error as e:
            raise CacheOperationError("Failed to purge tombstones from SQLite cache", original_exception=e) from e
        finally:
            self._return_connection(conn)

    async def clear(self) -> None:
        """Clear all items from the cache."""
        if self._closed:
            raise CacheOperationError("Cache store is closed")
            
        conn = await self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(f"DELETE FROM {self.table_name}")
            conn.commit()
        except sqlite3.Error as e:
            raise CacheOperationError("Failed to clear SQLite cache", original_exception=e) from e
        finally:
            self._return_connection(conn)

    async def size(self) -> int:
        """Get the number of items in the cache."""
        if self._closed:
            raise CacheOperationError("Cache store is closed")
            
        conn = await self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) as count FROM {self.table_name}")
            return cursor.fetchone()[0] or 0
        except sqlite3.Error as e:
            raise CacheOperationError("Failed to get SQLite cache size", original_exception=e) from e
        finally:
            self._return_connection(conn)

    async def enforce_limits(self, resource_limits: Any) -> None:
        """Enforce cache size limits."""
        if self._closed:
            raise CacheOperationError("Cache store is closed")
            
        conn = await self._get_connection()
        try:
            # First, clean up expired items
            cursor = conn.cursor()
            cursor.execute(
                f"DELETE FROM {self.table_name} WHERE expires_at IS NOT NULL AND expires_at <= ?",
                (time.time(),)
            )
            
            # Then apply eviction policy if needed
            current_size = await self.size()
            max_size = getattr(resource_limits, 'max_size', None)
            
            if max_size is not None and current_size > max_size:
                await self.eviction_policy.apply(self, max_size)
                
            conn.commit()
        except sqlite3.Error as e:
            raise CacheOperationError("Failed to enforce SQLite cache limits", original_exception=e) from e
        finally:
            self._return_connection(conn)

    def get_eviction_policy(self) -> EvictionPolicy:
        """Get the eviction policy for this cache store."""
        return self.eviction_policy

    async def close(self) -> None:
        """Close all database connections."""
        if self._closed:
            return
            
        async with self._connection_lock:
            for conn in self._connection_pool:
                try:
                    conn.close()
                except Exception as e:
                    logger.error(f"Error closing SQLite connection: {e}")
            self._connection_pool.clear()
            self._closed = True

    def __del__(self) -> None:
        """Ensure connections are closed when the object is garbage collected."""
        if not self._closed:
            self._closed = True
            for conn in self._connection_pool:
                try:
                    conn.close()
                except Exception:
                    pass
            self._connection_pool.clear()

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if self._closed:
            raise CacheOperationError("Cache store is closed")
            
        conn = await self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Get basic stats
            cursor.execute(f"SELECT COUNT(*) as count FROM {self.table_name}")
            total_items = cursor.fetchone()[0] or 0
            
            cursor.execute(f"""
                SELECT COUNT(*) as expired 
                FROM {self.table_name} 
                WHERE expires_at IS NOT NULL AND expires_at <= ?
            """, (time.time(),))
            expired_items = cursor.fetchone()[0] or 0
            
            return {
                "backend": "sqlite",
                "path": self.db_path,
                "table_name": self.table_name,
                "total_items": total_items,
                "expired_items": expired_items,
                "connection_pool_size": len(self._connection_pool),
                "max_connections": self.max_connections,
                "closed": self._closed
            }
        except sqlite3.Error as e:
            raise CacheOperationError("Failed to get SQLite cache stats", original_exception=e) from e
        finally:
            self._return_connection(conn)
