"""
DynamoDB cache store implementation for Semantrix.

This module provides a cache store implementation using Amazon DynamoDB.
It's optimized for high-performance, serverless applications on AWS.
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

import boto3
from boto3.dynamodb.conditions import Key
from botocore.config import Config
from botocore.exceptions import ClientError

from semantrix.cache_store.base import BaseCacheStore, EvictionPolicy, NoOpEvictionPolicy

# Configure logging
logger = logging.getLogger(__name__)

class DynamoDBCacheStore(BaseCacheStore):
    """
    DynamoDB-based cache store implementation.
    
    Features:
    - Serverless, fully managed NoSQL database
    - Single-digit millisecond performance
    - Built-in TTL support
    - Auto-scaling
    - Global tables for multi-region deployment
    
    Args:
        table_name: Name of the DynamoDB table (default: 'SemantrixCache')
        region_name: AWS region name (default: 'ap-south-1')
        endpoint_url: Custom endpoint URL (for local testing)
        ttl_attribute: Name of the TTL attribute (default: 'ttl')
        read_capacity_units: Read capacity units (default: 5)
        write_capacity_units: Write capacity units (default: 5)
        **kwargs: Additional arguments for BaseCacheStore
    """
    
    def __init__(
        self,
        table_name: str = "SemantrixCache",
        region_name: str = "ap-south-1",
        endpoint_url: Optional[str] = None,
        ttl_attribute: str = "ttl",
        read_capacity_units: int = 5,
        write_capacity_units: int = 5,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.table_name = table_name
        self.region_name = region_name
        self.endpoint_url = endpoint_url
        self.ttl_attribute = ttl_attribute
        self.read_capacity_units = read_capacity_units
        self.write_capacity_units = write_capacity_units
        
        # Initialize boto3 resources
        self._dynamodb = None
        self._table = None
        self._client = None
        self._lock = asyncio.Lock()
        self._connected = False
    
    async def _ensure_connected(self) -> None:
        """Ensure we have an active connection to DynamoDB."""
        if self._connected and self._table is not None:
            try:
                # Simple check to verify connection
                await asyncio.get_event_loop().run_in_executor(
                    None, self._table.table_status
                )
                return
            except Exception:
                self._connected = False
        
        async with self._lock:
            if not self._connected:
                try:
                    # Configure boto3 client
                    session = boto3.Session(region_name=self.region_name)
                    config = Config(
                        max_pool_connections=100,
                        retries={
                            'max_attempts': 10,
                            'mode': 'adaptive'
                        }
                    )
                    
                    # Initialize resources
                    self._dynamodb = session.resource(
                        'dynamodb',
                        endpoint_url=self.endpoint_url,
                        config=config
                    )
                    self._client = session.client(
                        'dynamodb',
                        endpoint_url=self.endpoint_url,
                        config=config
                    )
                    
                    # Ensure table exists
                    await self._ensure_table_exists()
                    self._table = self._dynamodb.Table(self.table_name)
                    self._connected = True
                    logger.info(f"Connected to DynamoDB table: {self.table_name}")
                    
                except Exception as e:
                    self._connected = False
                    logger.error(f"Failed to connect to DynamoDB: {e}")
                    raise
    
    async def _ensure_table_exists(self) -> None:
        """Ensure the DynamoDB table exists and is properly configured."""
        try:
            # Check if table exists
            await asyncio.get_event_loop().run_in_executor(
                None, self._client.describe_table, {'TableName': self.table_name}
            )
            logger.debug(f"Using existing DynamoDB table: {self.table_name}")
            
        except self._client.exceptions.ResourceNotFoundException:
            # Table doesn't exist, create it
            logger.info(f"Creating DynamoDB table: {self.table_name}")
            
            table_params = {
                'TableName': self.table_name,
                'KeySchema': [
                    {'AttributeName': 'key', 'KeyType': 'HASH'}  # Partition key
                ],
                'AttributeDefinitions': [
                    {'AttributeName': 'key', 'AttributeType': 'S'}
                ],
                'BillingMode': 'PROVISIONED',
                'ProvisionedThroughput': {
                    'ReadCapacityUnits': self.read_capacity_units,
                    'WriteCapacityUnits': self.write_capacity_units
                },
                'Tags': [
                    {
                        'Key': 'CreatedBy',
                        'Value': 'SemantrixCacheStore'
                    }
                ]
            }
            
            # Create the table
            await asyncio.get_event_loop().run_in_executor(
                None, self._client.create_table, table_params
            )
            
            # Wait for table to be active
            waiter = self._client.get_waiter('table_exists')
            waiter.wait(
                TableName=self.table_name,
                WaiterConfig={
                    'Delay': 1,
                    'MaxAttempts': 30
                }
            )
            
            # Enable TTL if specified
            if self.ttl_attribute:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._client.update_time_to_live,
                    {
                        'TableName': self.table_name,
                        'TimeToLiveSpecification': {
                            'Enabled': True,
                            'AttributeName': self.ttl_attribute
                        }
                    }
                )
            
            logger.info(f"Created DynamoDB table: {self.table_name}")
    
    async def get_exact(self, prompt: str) -> Optional[str]:
        """Get a cached response if it exists and is not expired."""
        try:
            await self._ensure_connected()
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._table.get_item(
                    Key={'key': prompt},
                    ConsistentRead=True
                )
            )
            
            item = response.get('Item')
            if not item:
                return None
                
            # Check if item is expired
            current_time = int(time.time())
            if 'ttl' in item and item['ttl'] < current_time:
                return None
                
            # Update last accessed time
            await self._update_last_accessed(prompt)
            
            return item.get('value')
            
        except Exception as e:
            logger.error(f"Error getting item from DynamoDB cache: {e}")
            return None
    
    async def _update_last_accessed(self, key: str) -> None:
        """Update the last accessed time for an item."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._table.update_item(
                    Key={'key': key},
                    UpdateExpression='SET last_accessed = :now, access_count = if_not_exists(access_count, :zero) + :incr',
                    ExpressionAttributeValues={
                        ':now': datetime.utcnow().isoformat(),
                        ':zero': 0,
                        ':incr': 1
                    }
                )
            )
        except Exception as e:
            logger.warning(f"Failed to update last accessed time: {e}")
    
    async def add(self, prompt: str, response: str, ttl: Optional[float] = None) -> None:
        """Add a response to the cache."""
        try:
            await self._ensure_connected()
            
            # Calculate TTL (in seconds since epoch)
            ttl_seconds = ttl if ttl is not None else self.ttl_seconds
            ttl_value = None
            if ttl_seconds is not None:
                ttl_value = int(time.time() + ttl_seconds)
            
            item = {
                'key': prompt,
                'value': response,
                'created_at': datetime.utcnow().isoformat(),
                'last_accessed': datetime.utcnow().isoformat(),
                'access_count': 1
            }
            
            if ttl_value is not None:
                item[self.ttl_attribute] = ttl_value
            
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._table.put_item(Item=item)
            )
            
        except Exception as e:
            logger.error(f"Error adding item to DynamoDB cache: {e}")
            raise
            
    async def delete(self, key: str) -> bool:
        """
        Delete a key from the DynamoDB cache.
        
        Args:
            key: The key to delete
            
        Returns:
            bool: True if the key was found and deleted, False otherwise
        """
        try:
            await self._ensure_connected()
            
            # Use delete_item which is idempotent
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._table.delete_item(
                    Key={'key': key},
                    ReturnValues='ALL_OLD'  # Returns the item as it appeared before deletion
                )
            )
            
            # If Attributes is in the response, the item existed and was deleted
            return 'Attributes' in response
            
        except Exception as e:
            logger.error(f"Error deleting key from DynamoDB cache: {e}")
            return False
    
    async def clear(self) -> None:
        """Clear all items from the cache."""
        try:
            await self._ensure_connected()
            
            # Scan and delete all items (for small to medium tables)
            scan_params = {
                'TableName': self.table_name,
                'ProjectionExpression': 'key'
            }
            
            while True:
                response = await asyncio.get_event_loop().run_in_executor(
                    None, self._client.scan, scan_params
                )
                
                with self._table.batch_writer() as batch:
                    for item in response.get('Items', []):
                        batch.delete_item(Key={'key': item['key']})
                
                # If there are more items to delete
                if 'LastEvaluatedKey' not in response:
                    break
                scan_params['ExclusiveStartKey'] = response['LastEvaluatedKey']
                
        except Exception as e:
            logger.error(f"Error clearing DynamoDB cache: {e}")
            raise
    
    async def size(self) -> int:
        """Get the number of items in the cache."""
        try:
            await self._ensure_connected()
            
            # Note: This performs a full table scan which can be expensive for large tables
            # For production, consider using a counter or CloudWatch metrics
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._table.scan(
                    Select='COUNT',
                    FilterExpression='attribute_not_exists(#ttl) OR #ttl > :now',
                    ExpressionAttributeNames={'#ttl': self.ttl_attribute},
                    ExpressionAttributeValues={':now': int(time.time())}
                )
            )
            
            return response.get('Count', 0)
            
        except Exception as e:
            logger.error(f"Error getting DynamoDB cache size: {e}")
            return 0
    
    async def enforce_limits(self, resource_limits: Any) -> None:
        """Enforce cache size limits."""
        # DynamoDB handles scaling automatically, but we can implement custom eviction here
        # For now, just clean up expired items
        await self._cleanup_expired()
        
        # Apply eviction policy if needed
        current_size = await self.size()
        max_size = getattr(resource_limits, 'max_size', None)
        
        if max_size is not None and current_size > max_size:
            await self.eviction_policy.apply(self, max_size)
    
    async def _cleanup_expired(self) -> None:
        """Clean up expired items."""
        # Note: This is handled automatically by DynamoDB TTL, but this can be used for manual cleanup
        pass
    
    def get_eviction_policy(self) -> EvictionPolicy:
        """Get the eviction policy for this cache store."""
        return self.eviction_policy
    
    async def close(self) -> None:
        """Close the DynamoDB connection."""
        self._connected = False
        self._table = None
        self._client = None
        self._dynamodb = None
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            await self._ensure_connected()
            
            stats = {
                'backend': 'dynamodb',
                'table': self.table_name,
                'region': self.region_name,
                'connected': self._connected,
                'ttl_attribute': self.ttl_attribute
            }
            
            # Get table description
            table_info = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.describe_table(TableName=self.table_name)
            )
            
            if 'Table' in table_info:
                stats.update({
                    'item_count': table_info['Table'].get('ItemCount', 0),
                    'status': table_info['Table'].get('TableStatus'),
                    'size_bytes': table_info['Table'].get('TableSizeBytes', 0),
                    'creation_date': table_info['Table'].get('CreationDateTime', '').isoformat(),
                    'provisioned_throughput': table_info['Table'].get('ProvisionedThroughput', {})
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting DynamoDB cache stats: {e}")
            return {'error': str(e)}
    
    def __del__(self) -> None:
        """Ensure resources are cleaned up."""
        if hasattr(self, '_client') and self._client:
            try:
                asyncio.get_event_loop().run_until_complete(self.close())
            except Exception:
                pass
