"""
Amazon DocumentDB cache store implementation for Semantrix.

This module provides a cache store implementation using Amazon DocumentDB (with MongoDB compatibility).
It extends the MongoDBCacheStore with AWS-specific optimizations and configurations.
"""

import asyncio
import logging
import ssl
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union

from boto3 import client as boto3_client
from botocore.exceptions import ClientError
from pymongo import MongoClient, errors
from pymongo.database import Database
from pymongo.collection import Collection

from semantrix.exceptions import CacheOperationError

from .mongodb import MongoDBCacheStore

# Configure logging
logger = logging.getLogger(__name__)

class DocumentDBCacheStore(MongoDBCacheStore):
    """
    Amazon DocumentDB cache store implementation.
    
    This class extends MongoDBCacheStore with AWS-specific features:
    - Automatic IAM authentication
    - TLS configuration
    - AWS region-specific optimizations
    - Cluster endpoint discovery
    
    Args:
        cluster_identifier: The DocumentDB cluster identifier
        region_name: AWS region name (default: 'us-east-1')
        db_name: Database name (default: 'semantrix_cache')
        collection_name: Collection name (default: 'cache')
        tls_enabled: Whether to enable TLS (default: True)
        tls_ca_file: Path to CA certificate file (default: 'rds-combined-ca-bundle.pem')
        retry_writes: Whether to enable retryable writes (default: False)
        **kwargs: Additional arguments passed to MongoDBCacheStore
    """
    
    def __init__(
        self,
        cluster_identifier: str,
        region_name: str = "ap-south-1",
        db_name: str = "semantrix_cache",
        collection_name: str = "cache",
        tls_enabled: bool = True,
        tls_ca_file: str = "rds-combined-ca-bundle.pem",
        retry_writes: bool = False,
        **kwargs: Any,
    ):
        self.cluster_identifier = cluster_identifier
        self.region_name = region_name
        self.tls_enabled = tls_enabled
        self.tls_ca_file = tls_ca_file
        self.retry_writes = retry_writes
        
        # Will be set during connection
        self._aws_rds_client = None
        self._connection_string = None
        
        # Initialize with a dummy connection string, will be updated in _ensure_connected
        super().__init__(
            connection_string="mongodb://localhost:27017/",
            db_name=db_name,
            collection_name=collection_name,
            **kwargs
        )
    
    async def _get_auth_token(self) -> str:
        """Generate an authentication token for DocumentDB."""
        if not self._aws_rds_client:
            self._aws_rds_client = boto3_client('rds', region_name=self.region_name)
        
        try:
            auth_token = self._aws_rds_client.generate_db_auth_token(
                DBHostname=self.cluster_identifier,
                Port=27017,  # Default MongoDB port
                DBUsername=self.username,
                Region=self.region_name
            )
            return auth_token
        except ClientError as e:
            logger.error(f"Failed to generate DocumentDB auth token: {e}")
            raise CacheOperationError("Failed to generate DocumentDB auth token", original_exception=e) from e
    
    async def _get_cluster_endpoints(self) -> Dict[str, str]:
        """Discover DocumentDB cluster endpoints."""
        if not self._aws_rds_client:
            self._aws_rds_client = boto3_client('rds', region_name=self.region_name)
        
        try:
            response = self._aws_rds_client.describe_db_clusters(
                DBClusterIdentifier=self.cluster_identifier
            )
            
            if not response.get('DBClusters'):
                raise ValueError(f"No DocumentDB cluster found with identifier: {self.cluster_identifier}")
            
            cluster = response['DBClusters'][0]
            return {
                'primary': cluster.get('ReaderEndpoint'),
                'readers': [m.get('Endpoint') for m in cluster.get('DBClusterMembers', []) 
                           if m.get('IsClusterWriter') is False],
                'writer': cluster.get('Endpoint')
            }
        except ClientError as e:
            logger.error(f"Failed to discover DocumentDB endpoints: {e}")
            raise CacheOperationError("Failed to discover DocumentDB endpoints", original_exception=e) from e
    
    async def _build_connection_string(self) -> str:
        """Build the MongoDB connection string for DocumentDB."""
        try:
            # Get cluster endpoints
            endpoints = await self._get_cluster_endpoints()
            
            # Use the writer endpoint for now (simplified)
            host = endpoints.get('writer')
            if not host:
                raise ValueError("No writer endpoint found for DocumentDB cluster")
            
            # Build connection string with TLS options
            connection_string = f"mongodb://{host}:27017/"
            
            # Add TLS options if enabled
            tls_opts = ""
            if self.tls_enabled:
                tls_opts = "&tls=true"
                if self.tls_ca_file:
                    tls_opts += f"&tlsCAFile={self.tls_ca_file}"
            
            # Add retryWrites if enabled
            retry_writes = "&retryWrites=true" if self.retry_writes else ""
            
            # Add read preference and other options
            connection_string += f"?ssl=true{tls_opts}{retry_writes}&readPreference=secondaryPreferred"
            
            return connection_string
            
        except (ValueError, ClientError) as e:
            logger.error(f"Failed to build DocumentDB connection string: {e}")
            raise CacheOperationError("Failed to build DocumentDB connection string", original_exception=e) from e
    
    async def _get_mongo_client(self) -> MongoClient:
        """Create and configure a MongoDB client for DocumentDB."""
        if not self._connection_string:
            self._connection_string = await self._build_connection_string()
        
        # Get auth token if using IAM authentication
        auth_token = await self._get_auth_token() if hasattr(self, 'username') and self.username else None
        
        # Configure TLS/SSL
        ssl_context = None
        if self.tls_enabled:
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            if self.tls_ca_file:
                ssl_context.load_verify_locations(self.tls_ca_file)
            else:
                ssl_context.verify_mode = ssl.CERT_NONE
        
        # Create client with AWS-specific settings
        client = MongoClient(
            self._connection_string,
            ssl=bool(ssl_context),
            ssl_cert_reqs=ssl.CERT_NONE if ssl_context else None,
            ssl_ca_certs=self.tls_ca_file if ssl_context and self.tls_ca_file else None,
            retryWrites=self.retry_writes,
            readPreference='secondaryPreferred',
            connectTimeoutMS=10000,
            socketTimeoutMS=30000,
            serverSelectionTimeoutMS=10000,
            maxPoolSize=100,
            minPoolSize=5,
            authMechanism='MONGODB-AWS' if auth_token else None,
            username=self.username if hasattr(self, 'username') else None,
            password=auth_token if auth_token else None,
            authSource='$external' if auth_token else None,
            tlsInsecure=not bool(ssl_context) if ssl_context else None
        )
        
        return client
    
    async def _ensure_connected(self) -> None:
        """Ensure we have an active connection to DocumentDB."""
        if self._connected and self._client is not None:
            try:
                # Ping the server to check connection
                await asyncio.get_event_loop().run_in_executor(
                    None, self._client.admin.command, 'ping'
                )
                return
            except errors.PyMongoError:
                # Connection lost, will reconnect
                self._connected = False
                if self._client:
                    self._client.close()
        
        async with self._lock:
            if not self._connected:
                try:
                    # Get a new client
                    self._client = await self._get_mongo_client()
                    
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
                    logger.info(f"Connected to DocumentDB cluster: {self.cluster_identifier}")
                    
                except (errors.PyMongoError, ClientError) as e:
                    self._connected = False
                    logger.error(f"Failed to connect to DocumentDB: {e}")
                    raise CacheOperationError("Failed to connect to DocumentDB", original_exception=e) from e
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics with DocumentDB-specific metrics."""
        try:
            await self._ensure_connected()
            if not self._collection:
                return {"error": "Not connected to DocumentDB"}
            
            # Get basic stats from parent
            stats = await super().get_stats()
            
            # Add DocumentDB-specific stats
            if self._client:
                try:
                    # Get server status
                    server_status = await asyncio.get_event_loop().run_in_executor(
                        None, self._client.admin.command, 'serverStatus'
                    )
                    
                    # Add relevant metrics
                    stats.update({
                        "documentdb": {
                            "host": self._client.HOST,
                            "version": server_status.get('version'),
                            "connections": server_status.get('connections', {}).get('current'),
                            "opcounters": server_status.get('opcounters', {}),
                            "network": server_status.get('network', {}),
                            "mem": server_status.get('mem', {})
                        }
                    })
                except Exception as e:
                    logger.warning(f"Could not get DocumentDB server status: {e}")
            
            return stats
            
        except errors.PyMongoError as e:
            logger.error(f"Error getting DocumentDB cache stats: {e}")
            raise CacheOperationError("Failed to get DocumentDB stats", original_exception=e) from e
