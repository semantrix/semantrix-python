"""
Unit tests for ElastiCacheStore.

These tests use mocks to test the ElastiCacheStore without requiring
an actual Redis or Memcached instance.
"""

import asyncio
import os
import pytest
from unittest.mock import MagicMock, patch, AsyncMock

from semantrix.cache_store.stores import ElastiCacheStore

# Skip tests if redis is not installed
pytest.importorskip("redis")

# Test configuration
TEST_ENDPOINT = "test.xxxxx.ng.0001.aps1.cache.amazonaws.com:6379"
TEST_REDIS_URL = f"redis://{TEST_ENDPOINT}"

# Check if we should run integration tests
RUN_INTEGRATION_TESTS = os.getenv("RUN_INTEGRATION_TESTS", "false").lower() == "true"


class TestElastiCacheStore:
    @pytest.fixture
    def mock_redis(self):
        with patch('redis.asyncio.Redis') as mock_redis_class:
            # Create a mock Redis instance
            mock_redis = AsyncMock()
            mock_redis_class.return_value = mock_redis
            
            # Configure mock methods
            mock_redis.ping = AsyncMock()
            mock_redis.get = AsyncMock(return_value="test_value")
            mock_redis.set = AsyncMock()
            mock_redis.hset = AsyncMock()
            mock_redis.flushdb = AsyncMock()
            mock_redis.keys = AsyncMock(return_value=[b"key1", b"key2"])
            mock_redis.info = AsyncMock(return_value={
                "redis_version": "6.2.6",
                "uptime_in_seconds": 12345,
                "used_memory_human": "1.5M",
                "connected_clients": 5
            })
            
            yield mock_redis

    @pytest.fixture
    def cache_store(self, mock_redis):
        return ElastiCacheStore(
            endpoint=TEST_ENDPOINT,
            ssl=False  # Disable SSL for testing
        )

    @pytest.mark.asyncio
    async def test_add_and_get(self, cache_store, mock_redis):
        # Test add
        await cache_store.add("test_key", "test_value", ttl=3600)
        
        # Verify set was called with correct parameters
        mock_redis.set.assert_awaited_once_with(
            "test_key",
            "test_value",
            px=3600000,  # TTL in milliseconds
            nx=False
        )
        
        # Verify hset was called for metadata
        mock_redis.hset.assert_awaited_once()
        
        # Test get
        result = await cache_store.get_exact("test_key")
        assert result == "test_value"
        
        # Verify get was called with the correct key
        mock_redis.get.assert_awaited_once_with("test_key")

    @pytest.mark.asyncio
    async def test_clear(self, cache_store, mock_redis):
        await cache_store.clear()
        mock_redis.flushdb.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_size(self, cache_store, mock_redis):
        # Mock keys to return some test keys
        mock_redis.keys.return_value = [b"key1", b"key2", b"key1:meta"]
        
        size = await cache_store.size()
        assert size == 2  # Should exclude the metadata key
        
        # Verify keys was called with the correct pattern
        mock_redis.keys.assert_awaited_once_with("*")

    @pytest.mark.asyncio
    async def test_enforce_limits(self, cache_store, mock_redis):
        # Mock size to return a large number
        mock_redis.keys.return_value = [f"key{i}".encode() for i in range(150)]
        
        # Create a mock eviction policy
        class MockEvictionPolicy:
            async def apply(self, store, max_size):
                # Simulate eviction by reducing the number of keys
                mock_redis.keys.return_value = [f"key{i}".encode() for i in range(90)]
        
        cache_store.eviction_policy = MockEvictionPolicy()
        
        # Create a simple object with max_size attribute
        class MockLimits:
            max_size = 100
        
        await cache_store.enforce_limits(MockLimits())
        
        # Verify the policy was applied
        assert (await cache_store.size()) == 90

    @pytest.mark.asyncio
    async def test_stats(self, cache_store, mock_redis):
        stats = await cache_store.get_stats()
        
        # Verify info was called
        mock_redis.info.assert_awaited_once()
        
        # Check some stats
        assert stats["backend"] == "elasticache"
        assert stats["endpoint"] == TEST_ENDPOINT
        assert stats["version"] == "6.2.6"


# Integration tests (only run when explicitly requested)
if RUN_INTEGRATION_TESTS:
    import redis.asyncio as redis
    
    @pytest.mark.integration
    class TestElastiCacheStoreIntegration:
        @pytest.fixture
        def cache_store(self):
            return ElastiCacheStore(
                endpoint=os.getenv("ELASTICACHE_ENDPOINT", "localhost:6379"),
                ssl=False,
                password=os.getenv("ELASTICACHE_PASSWORD")
            )
        
        @pytest.mark.asyncio
        async def test_integration_lifecycle(self, cache_store):
            test_key = f"test_key_{int(time.time())}"
            
            # Test add
            await cache_store.add(test_key, "integration_test_value", ttl=60)
            
            # Test get
            value = await cache_store.get_exact(test_key)
            assert value == "integration_test_value"
            
            # Test size
            size = await cache_store.size()
            assert size > 0
            
            # Test clear
            await cache_store.clear()
            assert await cache_store.size() == 0


if __name__ == "__main__":
    pytest.main([__file__])
