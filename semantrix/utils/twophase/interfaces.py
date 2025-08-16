"""
Abstract interfaces for Two-Phase Commit system components.

This module defines the core interfaces that allow for flexible
implementations and better testability of the 2PC system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Protocol
from datetime import datetime


class WALInterface(Protocol):
    """Interface for Write-Ahead Log implementations."""
    
    async def log_operation(
        self, 
        op_type: Any, 
        data: Dict[str, Any], 
        request_id: str
    ) -> None:
        """Log an operation to the WAL."""
        ...
    
    async def commit_operation(self, request_id: str) -> None:
        """Mark an operation as committed in the WAL."""
        ...
    
    async def fail_operation(self, request_id: str, error_message: str) -> None:
        """Mark an operation as failed in the WAL."""
        ...
    
    async def get_pending_operations(self) -> Dict[str, Dict[str, Any]]:
        """Get all pending operations from the WAL."""
        ...


class ParticipantInterface(ABC):
    """Abstract interface for 2PC participants."""
    
    @abstractmethod
    async def prepare(self, operation: 'OperationInterface') -> Tuple[bool, Optional[str]]:
        """
        Prepare for the operation.
        
        Args:
            operation: The 2PC operation to prepare for
            
        Returns:
            Tuple of (success, error_message)
        """
        raise NotImplementedError
    
    @abstractmethod
    async def commit(self, operation: 'OperationInterface') -> Tuple[bool, Optional[str]]:
        """
        Commit the prepared operation.
        
        Args:
            operation: The 2PC operation to commit
            
        Returns:
            Tuple of (success, error_message)
        """
        raise NotImplementedError
    
    @abstractmethod
    async def rollback(self, operation: 'OperationInterface') -> Tuple[bool, Optional[str]]:
        """
        Rollback the prepared operation.
        
        Args:
            operation: The 2PC operation to rollback
            
        Returns:
            Tuple of (success, error_message)
        """
        raise NotImplementedError


class OperationInterface(ABC):
    """Abstract interface for 2PC operations."""
    
    @property
    @abstractmethod
    def operation_id(self) -> str:
        """Get the operation ID."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def operation_type(self) -> str:
        """Get the operation type."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def data(self) -> Dict[str, Any]:
        """Get the operation data."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def participants(self) -> List[ParticipantInterface]:
        """Get the operation participants."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def state(self) -> 'StateInterface':
        """Get the operation state."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def prepare_results(self) -> List[Tuple[bool, Optional[str]]]:
        """Get the prepare phase results."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def commit_results(self) -> List[Tuple[bool, Optional[str]]]:
        """Get the commit phase results."""
        raise NotImplementedError
    
    @property
    @abstractmethod
    def rollback_results(self) -> List[Tuple[bool, Optional[str]]]:
        """Get the rollback phase results."""
        raise NotImplementedError
    
    @abstractmethod
    def mark_prepared(self) -> None:
        """Mark the operation as prepared."""
        raise NotImplementedError
    
    @abstractmethod
    def mark_committed(self) -> None:
        """Mark the operation as committed."""
        raise NotImplementedError
    
    @abstractmethod
    def mark_aborted(self, error_message: Optional[str] = None) -> None:
        """Mark the operation as aborted."""
        raise NotImplementedError
    
    @abstractmethod
    def mark_failed(self, error_message: str) -> None:
        """Mark the operation as failed."""
        raise NotImplementedError
    
    @abstractmethod
    def is_final_state(self) -> bool:
        """Check if the operation is in a final state."""
        raise NotImplementedError
    
    @abstractmethod
    def can_prepare(self) -> bool:
        """Check if the operation can be prepared."""
        raise NotImplementedError
    
    @abstractmethod
    def can_commit(self) -> bool:
        """Check if the operation can be committed."""
        raise NotImplementedError
    
    @abstractmethod
    def can_abort(self) -> bool:
        """Check if the operation can be aborted."""
        raise NotImplementedError


class StateInterface(ABC):
    """Abstract interface for operation states."""
    
    @property
    @abstractmethod
    def value(self) -> str:
        """Get the state value."""
        raise NotImplementedError


class CoordinatorInterface(ABC):
    """Abstract interface for 2PC coordinators."""
    
    @abstractmethod
    async def create_operation(
        self,
        operation_type: Any,
        data: Dict[str, Any],
        participants: List[ParticipantInterface],
        operation_id: Optional[str] = None
    ) -> OperationInterface:
        """Create a new Two-Phase Commit operation."""
        raise NotImplementedError
    
    @abstractmethod
    async def get_operation(self, operation_id: str) -> Optional[OperationInterface]:
        """Get an active operation by ID."""
        raise NotImplementedError
    
    @abstractmethod
    async def execute_operation(self, operation: OperationInterface) -> bool:
        """Execute a Two-Phase Commit operation."""
        raise NotImplementedError
    
    @abstractmethod
    async def recover_operations(self) -> List[OperationInterface]:
        """Recover in-doubt operations from WAL."""
        raise NotImplementedError


class ParticipantFactory(ABC):
    """Abstract factory for creating participants."""
    
    @abstractmethod
    def create_cache_participant(self, cache_store: Any) -> ParticipantInterface:
        """Create a cache store participant."""
        raise NotImplementedError
    
    @abstractmethod
    def create_vector_participant(self, vector_store: Any) -> ParticipantInterface:
        """Create a vector store participant."""
        raise NotImplementedError


class WALFactory(ABC):
    """Abstract factory for creating WAL instances."""
    
    @abstractmethod
    def create_wal(self, config: Dict[str, Any]) -> WALInterface:
        """Create a WAL instance with the given configuration."""
        raise NotImplementedError
