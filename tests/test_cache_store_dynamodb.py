"""
Unit tests for DynamoDBCacheStore.

These tests require a local DynamoDB instance or a mock.
For local development, you can use DynamoDB Local.
"""

import asyncio
import os
import pytest
from unittest.mock import MagicMock, patch

from semantrix.cache_store.stores import DynamoDBCacheStore

# Skip tests if boto3 is not installed
pytest.importorskip("boto3")

# Test configuration
TEST_TABLE_NAME = "test-semantrix-cache"
TEST_REGION = "us-west-2"

# Check if we should run integration tests
RUN_INTEGRATION_TESTS = os.getenv("RUN_INTEGRATION_TESTS", "false").lower() == "true"


class TestDynamoDBCacheStore:
    @pytest.fixture
    def mock_boto3(self):
        with patch('boto3.resource') as mock_resource, \
             patch('boto3.client') as mock_client:
            # Mock DynamoDB Table
            mock_table = MagicMock()
            mock_resource.return_value.Table.return_value = mock_table
            
            # Mock DynamoDB client
            mock_dynamodb = MagicMock()
            mock_client.return_value = mock_dynamodb
            
            # Mock describe_table to simulate table exists
            mock_dynamodb.describe_table.return_value = {
                'Table': {
                    'TableStatus': 'ACTIVE',
                    'TableName': TEST_TABLE_NAME
                }
            }
            
            yield mock_table, mock_dynamodb

    @pytest.fixture
    def cache_store(self, mock_boto3):
        return DynamoDBCacheStore(
            table_name=TEST_TABLE_NAME,
            region_name=TEST_REGION,
            endpoint_url="http://localhost:8000"  # For local testing
        )

    @pytest.mark.asyncio
    async def test_add_and_get(self, cache_store, mock_boto3):
        mock_table, _ = mock_boto3
        
        # Mock get_item response
        mock_table.get_item.return_value = {
            'Item': {
                'key': 'test_key',
                'value': 'test_value',
                'created_at': '2023-01-01T00:00:00',
                'last_accessed': '2023-01-01T00:00:00',
                'access_count': 1
            }
        }
        
        # Test add
        await cache_store.add("test_key", "test_value", ttl=3600)
        
        # Test get
        result = await cache_store.get_exact("test_key")
        assert result == "test_value"
        
        # Verify table operations were called
        mock_table.put_item.assert_called_once()
        mock_table.get_item.assert_called_once_with(
            Key={'key': 'test_key'},
            ConsistentRead=True
        )

    @pytest.mark.asyncio
    async def test_clear(self, cache_store, mock_boto3):
        mock_table, _ = mock_boto3
        
        # Mock scan response
        mock_table.scan.return_value = {
            'Items': [{'key': 'key1'}, {'key': 'key2'}],
            'LastEvaluatedKey': None
        }
        
        await cache_store.clear()
        
        # Verify scan and batch_writer were used
        mock_table.scan.assert_called_once_with(
            ProjectionExpression='key'
        )
        
    @pytest.mark.asyncio
    async def test_size(self, cache_store, mock_boto3):
        mock_table, _ = mock_boto3
        
        # Mock scan response with count
        mock_table.scan.return_value = {
            'Count': 42,
            'ScannedCount': 42
        }
        
        size = await cache_store.size()
        assert size == 42
        
        # Verify scan was called with the right parameters
        mock_table.scan.assert_called_once()
        call_args = mock_table.scan.call_args[1]
        assert call_args['Select'] == 'COUNT'
        
    @pytest.mark.asyncio
    async def test_enforce_limits(self, cache_store, mock_boto3):
        # This is a basic test - actual implementation would depend on the eviction policy
        mock_table, _ = mock_boto3
        
        # Mock size to return a large number
        mock_table.scan.return_value = {'Count': 1000}
        
        # Create a mock eviction policy
        class MockEvictionPolicy:
            async def apply(self, store, max_size):
                # Verify the store and max_size are passed correctly
                assert max_size == 100
                # Simulate eviction by reducing the size
                mock_table.scan.return_value = {'Count': 90}
        
        cache_store.eviction_policy = MockEvictionPolicy()
        
        # Create a simple object with max_size attribute
        class MockLimits:
            max_size = 100
        
        await cache_store.enforce_limits(MockLimits())
        
        # Verify the policy was applied
        assert (await cache_store.size()) == 90


# Integration tests (only run when explicitly requested)
if RUN_INTEGRATION_TESTS:
    @pytest.mark.integration
    class TestDynamoDBCacheStoreIntegration:
        @pytest.fixture
        def cache_store(self):
            return DynamoDBCacheStore(
                table_name=TEST_TABLE_NAME,
                region_name=TEST_REGION,
                endpoint_url=os.getenv("DYNAMODB_ENDPOINT_URL", "http://localhost:8000")
            )
        
        @pytest.mark.asyncio
        async def test_integration_lifecycle(self, cache_store):
            # Test add
            await cache_store.add("int_test_key", "int_test_value", ttl=60)
            
            # Test get
            value = await cache_store.get_exact("int_test_key")
            assert value == "int_test_value"
            
            # Test size
            size = await cache_store.size()
            assert size > 0
            
            # Test clear
            await cache_store.clear()
            assert await cache_store.size() == 0


if __name__ == "__main__":
    pytest.main([__file__])
