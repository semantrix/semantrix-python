# Tombstoning Implementation TODO List

## Overview
Implement hybrid tombstoning support where:
- **In-Memory Stores**: Full tombstoning support (InMemory, FAISS)
- **External Stores**: Graceful fallback to direct deletion with logging for visibility
- **API Consistency**: All stores accept `DeletionMode.TOMBSTONE` but external stores log and use direct deletion

## Cache Stores Implementation Status

### ✅ FULL TOMBSTONING SUPPORT (In-Memory)
1. **In-Memory Store** (`semantrix/cache_store/stores/in_memory.py`)
   - ✅ `delete()` with mode parameter
   - ✅ `tombstone()` method
   - ✅ `is_tombstoned()` method
   - ✅ `purge_tombstones()` method
   - ✅ `get_exact()` filters tombstoned items
   - ✅ `DeletionMode` import

### ✅ HYBRID SUPPORT (External Stores - Fallback to Direct Deletion)
2. **Redis Store** (`semantrix/cache_store/stores/redis.py`)
   - ✅ `delete()` with mode parameter (logs and falls back to direct deletion)
   - ✅ `_direct_delete()` method
   - ✅ `DeletionMode` import
   - ✅ Logging for visibility when tombstoning is requested

3. **SQLite Store** (`semantrix/cache_store/stores/sqlite.py`)
   - ✅ `delete()` with mode parameter (logs and falls back to direct deletion)
   - ✅ `_direct_delete()` method
   - ✅ `DeletionMode` import
   - ✅ Logging for visibility when tombstoning is requested

4. **PostgreSQL Store** (`semantrix/cache_store/stores/postgresql.py`)
   - ✅ `delete()` with mode parameter (logs and falls back to direct deletion)
   - ✅ `_direct_delete()` method
   - ✅ `DeletionMode` import
   - ✅ Logging for visibility when tombstoning is requested

5. **MongoDB Store** (`semantrix/cache_store/stores/mongodb.py`)
   - ✅ `delete()` with mode parameter (logs and falls back to direct deletion)
   - ✅ `_direct_delete()` method
   - ✅ `DeletionMode` import
   - ✅ Logging for visibility when tombstoning is requested

6. **Google Memorystore Store** (`semantrix/cache_store/stores/google_memorystore.py`)
   - ✅ `delete()` with mode parameter (logs and falls back to direct deletion)
   - ✅ `_direct_delete()` method
   - ✅ `DeletionMode` import
   - ✅ Logging for visibility when tombstoning is requested

7. **ElastiCache Store** (`semantrix/cache_store/stores/elasticache.py`)
   - ✅ `delete()` with mode parameter (logs and falls back to direct deletion)
   - ✅ `_direct_delete()` method
   - ✅ `DeletionMode` import
   - ✅ Logging for visibility when tombstoning is requested

8. **DynamoDB Store** (`semantrix/cache_store/stores/dynamodb.py`)
   - ✅ `delete()` with mode parameter (logs and falls back to direct deletion)
   - ✅ `_direct_delete()` method
   - ✅ `DeletionMode` import
   - ✅ Logging for visibility when tombstoning is requested

9. **DocumentDB Store** (`semantrix/cache_store/stores/documentdb.py`)
   - ✅ Inherits MongoDB hybrid support
   - ✅ `delete()` with mode parameter (logs and falls back to direct deletion)
   - ✅ `_direct_delete()` method
   - ✅ `DeletionMode` import
   - ✅ Logging for visibility when tombstoning is requested

10. **Memcached Store** (`semantrix/cache_store/stores/memcached.py`)
    - ✅ `delete()` with mode parameter (logs and falls back to direct deletion)
    - ✅ `_direct_delete()` method
    - ✅ `DeletionMode` import
    - ✅ Logging for visibility when tombstoning is requested

## Vector Stores Implementation Status

### ✅ FULL TOMBSTONING SUPPORT (In-Memory)
1. **FAISS Store** (`semantrix/vector_store/stores/faiss.py`)
   - ✅ `tombstone()` method
   - ✅ `purge_tombstones()` method
   - ✅ `search()` filters tombstoned vectors
   - ✅ `delete()` supports documents parameter

### ✅ HYBRID SUPPORT (External Stores - Fallback to Direct Deletion)
2. **Chroma Store** (`semantrix/vector_store/stores/chroma.py`)
   - ✅ `delete()` with documents parameter (logs and falls back to direct deletion)
   - ✅ `_direct_delete()` method
   - ✅ Logging for visibility when tombstoning is requested

3. **Redis Vector Store** (`semantrix/vector_store/stores/redis.py`)
   - ✅ `delete()` with documents parameter (logs and falls back to direct deletion)
   - ✅ `_direct_delete()` method
   - ✅ Logging for visibility when tombstoning is requested

4. **Qdrant Store** (`semantrix/vector_store/stores/qdrant.py`)
   - ✅ `delete()` with documents parameter (logs and falls back to direct deletion)
   - ✅ `_direct_delete()` method
   - ✅ Logging for visibility when tombstoning is requested

5. **Pinecone Store** (`semantrix/vector_store/stores/pinecone.py`)
   - ✅ `delete()` with documents parameter (logs and falls back to direct deletion)
   - ✅ `_direct_delete()` method
   - ✅ Logging for visibility when tombstoning is requested

6. **PGVector Store** (`semantrix/vector_store/stores/pgvector.py`)
   - ✅ `delete()` with documents parameter (logs and falls back to direct deletion)
   - ✅ `_direct_delete()` method
   - ✅ Logging for visibility when tombstoning is requested

7. **Milvus Store** (`semantrix/vector_store/stores/milvus.py`)
   - ✅ `delete()` with documents parameter (logs and falls back to direct deletion)
   - ✅ `_direct_delete()` method
   - ✅ Logging for visibility when tombstoning is requested

## Implementation Strategy

### For In-Memory Stores (Full Tombstoning):
- Complete tombstoning implementation with metadata tracking
- Background cleanup processes
- Filtering of tombstoned items from all operations

### For External Stores (Hybrid Approach):
- Accept `DeletionMode.TOMBSTONE` for API consistency
- Log informative message when tombstoning is requested
- Fall back to direct deletion implementation
- No complex metadata tracking or background cleanup needed

## Benefits of Hybrid Approach:
1. **Performance**: External stores use optimized native deletion
2. **Simplicity**: Reduced implementation complexity
3. **API Consistency**: All stores accept the same interface
4. **Visibility**: Logging provides transparency about behavior
5. **Real-World Alignment**: Matches typical usage patterns

## Progress Summary
- **In-Memory Stores with Full Tombstoning:** 2/2 completed (100%)
- **External Stores with Hybrid Support:** 15/15 completed (100%)
- **Overall:** 17/17 stores completed (100%)

## Implementation Complete! 🎉
All stores now support the unified deletion API with appropriate behavior for their storage type.
