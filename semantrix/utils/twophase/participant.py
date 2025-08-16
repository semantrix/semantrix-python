"""
Base participant interface for Two-Phase Commit operations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Generic

T = TypeVar('T')


class Participant(Generic[T], ABC):
    """
    Interface for participants in a Two-Phase Commit.
    
    Participants are responsible for executing operations atomically
    across different resources (cache stores, vector stores, etc.).
    """
    
    @abstractmethod
    async def prepare(self, operation: 'TwoPhaseOperation') -> Tuple[bool, Optional[str]]:
        """
        Prepare for the operation.
        
        This phase should validate the operation and ensure it can be
        executed, but should not make any permanent changes.
        
        Args:
            operation: The 2PC operation to prepare for
            
        Returns:
            Tuple of (success, error_message)
        """
        raise NotImplementedError
    
    @abstractmethod
    async def commit(self, operation: 'TwoPhaseOperation') -> Tuple[bool, Optional[str]]:
        """
        Commit the prepared operation.
        
        This phase should execute the actual operation and make
        permanent changes to the resource.
        
        Args:
            operation: The 2PC operation to commit
            
        Returns:
            Tuple of (success, error_message)
        """
        raise NotImplementedError
    
    @abstractmethod
    async def rollback(self, operation: 'TwoPhaseOperation') -> Tuple[bool, Optional[str]]:
        """
        Rollback the prepared operation.
        
        This phase should undo any changes made during the commit phase
        if the operation needs to be aborted.
        
        Args:
            operation: The 2PC operation to rollback
            
        Returns:
            Tuple of (success, error_message)
        """
        raise NotImplementedError


# Import here to avoid circular imports
from .operation import TwoPhaseOperation
