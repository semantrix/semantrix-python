"""Tests for the Semantrix cache module."""

import asyncio
import pytest
import numpy as np
from typing import Optional, Dict

from semantrix.core.cache import Semantrix
from semantrix.embedding import BaseEmbedder
from semantrix.vector_store.vector_store import BaseVectorStore
from semantrix.cache_store import BaseCacheStore

class MockEmbedder(BaseEmbedder):
    """Mock embedder for testing."""
    def __init__(self, dim=16):
        self._dim = dim
    async def aencode(self, text: str) -> np.ndarray:
        return np.random.rand(self._dim).astype(np.float32)
    def encode(self, text: str) -> np.ndarray:
        return asyncio.run(self.aencode(text))
    def get_dimension(self) -> int:
        return self._dim

class MockVectorStore(BaseVectorStore):
    """Mock vector store for testing."""
    def __init__(self, dim=16):
        self.embeddings = {}
    def add(self, emb: np.ndarray, resp: str) -> None:
        self.embeddings[resp] = emb
    def search(self, emb: np.ndarray, **kwargs) -> Optional[str]:
        return next(iter(self.embeddings.keys()), None) if self.embeddings else None
    def clear(self) -> None:
        self.embeddings = {}
    def size(self) -> int:
        return len(self.embeddings)

class MockCacheStore(BaseCacheStore):
    """Mock cache store for testing eviction."""
    def __init__(self, max_size=2):
        self.cache: Dict[str, str] = {}
        self.max_size = max_size
    async def get_exact(self, p: str) -> Optional[str]:
        return self.cache.get(p)
    async def add(self, p: str, r: str) -> None:
        if len(self.cache) >= self.max_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[p] = r
    async def enforce_limits(self, _) -> None:
        while len(self.cache) > self.max_size:
            self.cache.pop(next(iter(self.cache)))
    async def clear(self) -> None:
        self.cache = {}
    async def size(self) -> int:
        return len(self.cache)

@pytest.fixture
def cache():
    """Create a test cache instance."""
    return Semantrix(
        embedder=MockEmbedder(),
        vector_store=MockVectorStore(),
        cache_store=MockCacheStore(max_size=2),
        similarity_threshold=0.8
    )

@pytest.mark.asyncio
async def test_async_eviction(cache):
    """Test that cache evicts oldest items when full."""
    # Add items up to capacity
    await cache.set("p1", "r1")
    await cache.set("p2", "r2")
    assert await cache.cache_store.size() == 2
    
    # Add one more to trigger eviction
    await cache.set("p3", "r3")
    assert await cache.cache_store.size() == 2
    
    # First item should be evicted
    assert await cache.get_exact("p1") is None
    assert await cache.get_exact("p3") == "r3"

@pytest.mark.asyncio
async def test_async_get_set(cache):
    """Test basic async get/set operations."""
    # Test cache miss
    assert await cache.get("test") is None
    
    # Test set and get
    await cache.set("test", "response")
    assert await cache.get("test") == "response"
    assert await cache.get_exact("test") == "response"

@pytest.mark.asyncio
async def test_async_clear(cache):
    """Test cache clearing."""
    await cache.set("test", "response")
    assert await cache.cache_store.size() == 1
    
    await cache.clear()
    assert await cache.cache_store.size() == 0
    assert await cache.get("test") is None