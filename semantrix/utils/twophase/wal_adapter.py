"""
WAL adapter for flexible integration with different Write-Ahead Log implementations.

This module provides adapters and factories for integrating different WAL
implementations with the 2PC system.
"""

import logging
from typing import Any, Dict, List, Optional, Union
from abc import ABC, abstractmethod

from .interfaces import WALInterface, WALFactory

logger = logging.getLogger(__name__)


class WALAdapter(WALInterface):
    """
    Adapter that wraps different WAL implementations to provide a consistent interface.
    
    This adapter allows the 2PC system to work with different WAL implementations
    without tight coupling to any specific one.
    """
    
    def __init__(self, wal_implementation: Any):
        """
        Initialize the adapter with a WAL implementation.
        
        Args:
            wal_implementation: The actual WAL implementation to wrap
        """
        self._wal = wal_implementation
        self._validate_implementation()
    
    def _validate_implementation(self) -> None:
        """Validate that the WAL implementation has the required methods."""
        required_methods = ['log_operation', 'commit_operation', 'fail_operation', 'get_pending_operations']
        
        for method in required_methods:
            if not hasattr(self._wal, method):
                raise ValueError(f"WAL implementation must have method: {method}")
    
    async def log_operation(
        self, 
        op_type: Any, 
        data: Dict[str, Any], 
        request_id: str
    ) -> None:
        """Log an operation to the WAL."""
        try:
            await self._wal.log_operation(op_type, data, request_id)
        except Exception as e:
            logger.error(f"Error logging operation to WAL: {e}", exc_info=True)
            raise
    
    async def commit_operation(self, request_id: str) -> None:
        """Mark an operation as committed in the WAL."""
        try:
            await self._wal.commit_operation(request_id)
        except Exception as e:
            logger.error(f"Error committing operation in WAL: {e}", exc_info=True)
            raise
    
    async def fail_operation(self, request_id: str, error_message: str) -> None:
        """Mark an operation as failed in the WAL."""
        try:
            await self._wal.fail_operation(request_id, error_message)
        except Exception as e:
            logger.error(f"Error failing operation in WAL: {e}", exc_info=True)
            raise
    
    async def get_pending_operations(self) -> Dict[str, Dict[str, Any]]:
        """Get all pending operations from the WAL."""
        try:
            return await self._wal.get_pending_operations()
        except Exception as e:
            logger.error(f"Error getting pending operations from WAL: {e}", exc_info=True)
            raise


class NoOpWAL(WALInterface):
    """
    No-operation WAL implementation for testing or when WAL is disabled.
    
    This implementation provides a WAL interface that does nothing,
    useful for testing or when WAL functionality is not needed.
    """
    
    async def log_operation(
        self, 
        op_type: Any, 
        data: Dict[str, Any], 
        request_id: str
    ) -> None:
        """No-op: do nothing."""
        logger.debug(f"NoOpWAL: Would log operation {request_id} of type {op_type}")
    
    async def commit_operation(self, request_id: str) -> None:
        """No-op: do nothing."""
        logger.debug(f"NoOpWAL: Would commit operation {request_id}")
    
    async def fail_operation(self, request_id: str, error_message: str) -> None:
        """No-op: do nothing."""
        logger.debug(f"NoOpWAL: Would fail operation {request_id} with error: {error_message}")
    
    async def get_pending_operations(self) -> Dict[str, Dict[str, Any]]:
        """No-op: return empty dict."""
        logger.debug("NoOpWAL: Would get pending operations")
        return {}


class MemoryWAL(WALInterface):
    """
    In-memory WAL implementation for testing.
    
    This implementation stores WAL data in memory, useful for testing
    scenarios where persistence is not required.
    """
    
    def __init__(self):
        self._operations: Dict[str, Dict[str, Any]] = {}
        self._committed: set = set()
        self._failed: Dict[str, str] = {}
    
    async def log_operation(
        self, 
        op_type: Any, 
        data: Dict[str, Any], 
        request_id: str
    ) -> None:
        """Log an operation to memory."""
        self._operations[request_id] = {
            'operation_type': op_type,
            'data': data,
            'status': 'pending'
        }
        logger.debug(f"MemoryWAL: Logged operation {request_id}")
    
    async def commit_operation(self, request_id: str) -> None:
        """Mark an operation as committed."""
        if request_id in self._operations:
            self._operations[request_id]['status'] = 'committed'
            self._committed.add(request_id)
            logger.debug(f"MemoryWAL: Committed operation {request_id}")
    
    async def fail_operation(self, request_id: str, error_message: str) -> None:
        """Mark an operation as failed."""
        if request_id in self._operations:
            self._operations[request_id]['status'] = 'failed'
            self._failed[request_id] = error_message
            logger.debug(f"MemoryWAL: Failed operation {request_id}: {error_message}")
    
    async def get_pending_operations(self) -> Dict[str, Dict[str, Any]]:
        """Get all pending operations."""
        pending = {
            op_id: op_data for op_id, op_data in self._operations.items()
            if op_data['status'] == 'pending'
        }
        logger.debug(f"MemoryWAL: Returning {len(pending)} pending operations")
        return pending


class DefaultWALFactory(WALFactory):
    """
    Default factory for creating WAL instances.
    
    This factory supports creating different types of WAL implementations
    based on configuration.
    """
    
    def create_wal(self, config: Dict[str, Any]) -> WALInterface:
        """
        Create a WAL instance based on configuration.
        
        Args:
            config: Configuration dictionary with 'type' and other parameters
            
        Returns:
            WALInterface instance
        """
        wal_type = config.get('type', 'default')
        
        if wal_type == 'none' or wal_type == 'noop':
            return NoOpWAL()
        elif wal_type == 'memory':
            return MemoryWAL()
        elif wal_type == 'default':
            # Import the default WAL implementation
            try:
                from semantrix.utils.wal import WriteAheadLog, create_wal
                
                # Extract WAL-specific configuration
                wal_config = config.get('wal_config', {})
                return WALAdapter(create_wal(wal_config))
            except ImportError as e:
                logger.warning(f"Default WAL not available: {e}, falling back to NoOpWAL")
                return NoOpWAL()
        else:
            raise ValueError(f"Unknown WAL type: {wal_type}")


# Global factory instance
default_wal_factory = DefaultWALFactory()


def create_wal_adapter(wal_implementation: Any) -> WALInterface:
    """
    Create a WAL adapter for the given implementation.
    
    Args:
        wal_implementation: The WAL implementation to wrap
        
    Returns:
        WALInterface adapter
    """
    return WALAdapter(wal_implementation)


def create_wal_from_config(config: Dict[str, Any]) -> WALInterface:
    """
    Create a WAL instance from configuration using the default factory.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        WALInterface instance
    """
    return default_wal_factory.create_wal(config)
