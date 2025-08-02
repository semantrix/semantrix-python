"""
Two-Phase Commit (2PC) coordinator for atomic operations across multiple resources.
"""
import asyncio
import logging
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Generic, Callable, Awaitable

from semantrix.utils.wal import WriteAheadLog, OperationType, OperationStatus

logger = logging.getLogger(__name__)

T = TypeVar('T')

class TwoPhaseState(Enum):
    """State of a Two-Phase Commit operation."""
    INITIALIZING = auto()
    PREPARING = auto()
    PREPARED = auto()
    COMMITTING = auto()
    COMMITTED = auto()
    ABORTING = auto()
    ABORTED = auto()
    FAILED = auto()

class Participant(Generic[T]):
    """Interface for participants in a Two-Phase Commit."""
    
    async def prepare(self, operation: 'TwoPhaseOperation') -> Tuple[bool, Optional[str]]:
        """
        Prepare for the operation.
        
        Returns:
            Tuple of (success, error_message)
        """
        raise NotImplementedError
    
    async def commit(self, operation: 'TwoPhaseOperation') -> Tuple[bool, Optional[str]]:
        """
        Commit the prepared operation.
        
        Returns:
            Tuple of (success, error_message)
        """
        raise NotImplementedError
    
    async def rollback(self, operation: 'TwoPhaseOperation') -> Tuple[bool, Optional[str]]:
        """
        Rollback the prepared operation.
        
        Returns:
            Tuple of (success, error_message)
        """
        raise NotImplementedError

class TwoPhaseOperation:
    """Represents an operation in a Two-Phase Commit."""
    
    def __init__(
        self,
        operation_type: OperationType,
        operation_id: str,
        data: Dict[str, Any],
        participants: List[Participant],
        wal: Optional[WriteAheadLog] = None
    ):
        self.operation_type = operation_type
        self.operation_id = operation_id
        self.data = data
        self.participants = participants
        self.wal = wal
        self.state = TwoPhaseState.INITIALIZING
        self.prepare_results: List[Tuple[bool, Optional[str]]] = []
        self.commit_results: List[Tuple[bool, Optional[str]]] = []
        self.rollback_results: List[Tuple[bool, Optional[str]]] = []
        self._lock = asyncio.Lock()
    
    async def execute(self) -> bool:
        """Execute the operation using Two-Phase Commit protocol."""
        try:
            # Phase 1: Prepare
            await self._prepare()
            
            # Check if all participants prepared successfully
            if all(success for success, _ in self.prepare_results):
                # Phase 2: Commit
                await self._commit()
                return True
            else:
                # At least one participant failed to prepare, abort
                await self._abort()
                return False
        except Exception as e:
            logger.error(f"Error during 2PC execution: {e}", exc_info=True)
            await self._abort()
            return False
    
    async def _prepare(self) -> None:
        """Execute the prepare phase."""
        async with self._lock:
            self.state = TwoPhaseState.PREPARING
            
            # Log the operation to WAL if available
            if self.wal:
                await self.wal.log_operation(
                    operation_type=self.operation_type,
                    data=self.data,
                    request_id=self.operation_id
                )
            
            # Prepare all participants
            prepare_tasks = [
                participant.prepare(self) 
                for participant in self.participants
            ]
            self.prepare_results = await asyncio.gather(
                *prepare_tasks, 
                return_exceptions=True
            )
            
            # Convert exceptions to failure results
            for i, result in enumerate(self.prepare_results):
                if isinstance(result, Exception):
                    self.prepare_results[i] = (False, str(result))
            
            self.state = TwoPhaseState.PREPARED
    
    async def _commit(self) -> None:
        """Execute the commit phase."""
        async with self._lock:
            self.state = TwoPhaseState.COMMITTING
            
            # Commit all participants
            commit_tasks = [
                participant.commit(self)
                for i, participant in enumerate(self.participants)
                if self.prepare_results[i][0]  # Only commit if prepare was successful
            ]
            
            # Wait for all commits to complete
            commit_results = await asyncio.gather(
                *commit_tasks,
                return_exceptions=True
            )
            
            # Process commit results
            result_idx = 0
            for i in range(len(self.participants)):
                if self.prepare_results[i][0]:
                    # This participant was committed
                    result = commit_results[result_idx]
                    if isinstance(result, Exception):
                        self.commit_results.append((False, str(result)))
                    else:
                        self.commit_results.append(result)
                    result_idx += 1
                else:
                    # This participant wasn't committed (failed prepare)
                    self.commit_results.append((False, "Prepare failed"))
            
            # Update WAL if available
            if self.wal:
                if all(success for success, _ in self.commit_results):
                    await self.wal.commit_operation(self.operation_id)
                else:
                    await self.wal.fail_operation(
                        self.operation_id, 
                        "Failed to commit all participants"
                    )
            
            self.state = TwoPhaseState.COMMITTED
    
    async def _abort(self) -> None:
        """Abort the operation."""
        async with self._lock:
            if self.state in (TwoPhaseState.ABORTING, TwoPhaseState.ABORTED, TwoPhaseState.FAILED):
                return
                
            self.state = TwoPhaseState.ABORTING
            
            # Rollback all participants that were prepared
            rollback_tasks = []
            for i, participant in enumerate(self.participants):
                if i < len(self.prepare_results) and self.prepare_results[i][0]:
                    rollback_tasks.append(participant.rollback(self))
            
            # Wait for rollbacks to complete
            rollback_results = await asyncio.gather(
                *rollback_tasks,
                return_exceptions=True
            )
            
            # Process rollback results
            result_idx = 0
            for i in range(len(self.participants)):
                if i < len(self.prepare_results) and self.prepare_results[i][0]:
                    result = rollback_results[result_idx]
                    if isinstance(result, Exception):
                        self.rollback_results.append((False, str(result)))
                    else:
                        self.rollback_results.append(result)
                    result_idx += 1
            
            # Update WAL if available
            if self.wal:
                await self.wal.fail_operation(
                    self.operation_id,
                    "Operation aborted"
                )
            
            self.state = TwoPhaseState.ABORTED

class TwoPhaseCoordinator:
    """Coordinates Two-Phase Commit operations."""
    
    def __init__(self, wal: Optional[WriteAheadLog] = None):
        self.wal = wal
        self.active_operations: Dict[str, TwoPhaseOperation] = {}
        self._lock = asyncio.Lock()
    
    async def create_operation(
        self,
        operation_type: OperationType,
        data: Dict[str, Any],
        participants: List[Participant],
        operation_id: Optional[str] = None
    ) -> TwoPhaseOperation:
        """Create a new Two-Phase Commit operation."""
        op_id = operation_id or str(uuid.uuid4())
        operation = TwoPhaseOperation(
            operation_type=operation_type,
            operation_id=op_id,
            data=data,
            participants=participants,
            wal=self.wal
        )
        
        async with self._lock:
            self.active_operations[op_id] = operation
        
        return operation
    
    async def get_operation(self, operation_id: str) -> Optional[TwoPhaseOperation]:
        """Get an active operation by ID."""
        async with self._lock:
            return self.active_operations.get(operation_id)
    
    async def recover_operations(self) -> List[TwoPhaseOperation]:
        """Recover in-doubt operations from WAL."""
        if not self.wal:
            return []
        
        # Get all pending operations from WAL
        pending_ops = await self.wal.get_pending_operations()
        recovered_ops = []
        
        for op_id, op_data in pending_ops.items():
            # Try to recover the operation state
            operation = TwoPhaseOperation(
                operation_type=op_data['operation_type'],
                operation_id=op_id,
                data=op_data['data'],
                participants=[],  # Participants need to be re-attached by the caller
                wal=self.wal
            )
            operation.state = TwoPhaseState.PREPARED  # Assume prepared if in WAL
            
            async with self._lock:
                self.active_operations[op_id] = operation
            
            recovered_ops.append(operation)
        
        return recovered_ops
