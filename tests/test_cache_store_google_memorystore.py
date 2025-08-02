"""
Unit tests for GoogleMemorystoreCacheStore.

These tests use mocks to test the GoogleMemorystoreCacheStore without requiring
an actual Google Cloud Memorystore instance.
"""

import asyncio
import os
import pytest
from unittest.mock import MagicMock, patch, AsyncMock, ANY

from semantrix.cache_store.stores import GoogleMemorystoreCacheStore

# Skip tests if google-cloud-redis is not installed
pytest.importorskip("google.cloud.redis_v1")
pytest.importorskip("redis")

# Test configuration
TEST_PROJECT_ID = "test-project"
TEST_REGION = "us-central1"
TEST_INSTANCE_ID = "test-instance"
TEST_INSTANCE_NAME = f"projects/{TEST_PROJECT_ID}/locations/{TEST_REGION}/instances/{TEST_INSTANCE_ID}"

# Check if we should run integration tests
RUN_INTEGRATION_TESTS = os.getenv("RUN_INTEGRATION_TESTS", "false").lower() == "true"


class TestGoogleMemorystoreCacheStore:
    @pytest.fixture
    def mock_cloud_redis(self):
        with patch('google.cloud.redis_v1.CloudRedisClient') as mock_client_class, \
             patch('redis.asyncio.Redis') as mock_redis_class:
            
            # Create mock Cloud Redis client
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client
            
            # Create mock Redis instance
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
            
            # Mock instance response
            mock_instance = MagicMock()
            mock_instance.name = TEST_INSTANCE_NAME
            mock_instance.host = "10.0.0.1"
            mock_instance.port = 6379
            mock_instance.tier = 1  # BASIC
            mock_instance.memory_size_gb = 1
            mock_instance.redis_version = "REDIS_6_X"
            mock_instance.authorized_network = "projects/test-project/global/networks/default"
            
            # Mock get_instance
            mock_client.get_instance.return_value = mock_instance
            
            # Mock list_instances
            mock_list_response = MagicMock()
            mock_list_response.instances = [mock_instance]
            mock_client.list_instances.return_value = mock_list_response
            
            yield mock_client, mock_redis, mock_instance

    @pytest.fixture
    def cache_store(self, mock_cloud_redis):
        return GoogleMemorystoreCacheStore(
            project_id=TEST_PROJECT_ID,
            region=TEST_REGION,
            instance_id=TEST_INSTANCE_ID,
            memory_size_gb=1
        )

    @pytest.mark.asyncio
    async def test_add_and_get(self, cache_store, mock_cloud_redis):
        _, mock_redis, _ = mock_cloud_redis
        
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
    async def test_clear(self, cache_store, mock_cloud_redis):
        _, mock_redis, _ = mock_cloud_redis
        await cache_store.clear()
        mock_redis.flushdb.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_size(self, cache_store, mock_cloud_redis):
        _, mock_redis, _ = mock_cloud_redis
        
        # Mock keys to return some test keys
        mock_redis.keys.return_value = [b"key1", b"key2", b"key1:meta"]
        
        size = await cache_store.size()
        assert size == 2  # Should exclude the metadata key
        
        # Verify keys was called with the correct pattern
        mock_redis.keys.assert_awaited_once_with("*")

    @pytest.mark.asyncio
    async def test_list_instances(self, mock_cloud_redis):
        mock_client, _, _ = mock_cloud_redis
        
        # Call the class method
        instances = await GoogleMemorystoreCacheStore.list_instances(
            project_id=TEST_PROJECT_ID,
            region=TEST_REGION
        )
        
        # Verify the client was called with the correct parameters
        mock_client.list_instances.assert_called_once_with(
            parent=f"projects/{TEST_PROJECT_ID}/locations/{TEST_REGION}"
        )
        
        # Check the result
        assert len(instances) == 1
        assert instances[0]["name"] == TEST_INSTANCE_ID

    @pytest.mark.asyncio
    async def test_stats(self, cache_store, mock_cloud_redis):
        _, mock_redis, _ = mock_cloud_redis
        
        stats = await cache_store.get_stats()
        
        # Verify info was called
        mock_redis.info.assert_awaited_once()
        
        # Check some stats
        assert stats["backend"] == "google_memorystore"
        assert stats["instance_id"] == TEST_INSTANCE_ID
        assert stats["project_id"] == TEST_PROJECT_ID
        assert stats["version"] == "6.2.6"


# Integration tests (only run when explicitly requested)
if RUN_INTEGRATION_TESTS:
    import redis.asyncio as redis
    
    @pytest.mark.integration
    class TestGoogleMemorystoreCacheStoreIntegration:
        @pytest.fixture
        def cache_store(self):
            return GoogleMemorystoreCacheStore(
                project_id=os.getenv("GOOGLE_CLOUD_PROJECT"),
                region=os.getenv("GOOGLE_CLOUD_REGION", "us-central1"),
                instance_id=os.getenv("GOOGLE_MEMORYSTORE_INSTANCE", "semantrix-test"),
                memory_size_gb=1
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
