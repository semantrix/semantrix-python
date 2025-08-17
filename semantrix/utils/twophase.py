"""
Two-Phase Commit (2PC) coordinator for atomic operations across multiple resources.
"""
import asyncio
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Generic, Callable, Awaitable

from semantrix.utils.wal import WriteAheadLog, OperationType, LogStatus as OperationStatus
from semantrix.utils.logging import get_logger

logger = get_logger(__name__)

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
            logger.debug(f"Starting prepare phase for operation {self.operation_id}")
            await self._prepare()
            
            # Log prepare results
            logger.debug(f"Prepare results: {self.prepare_results}")
            
            # Check if all participants prepared successfully
            if all(success for success, _ in self.prepare_results):
                logger.debug("All participants prepared successfully, starting commit phase")
                # Phase 2: Commit
                await self._commit()
                
                # Log detailed commit results
                logger.info("[2PC] Final commit results:")
                for i, (success, msg) in enumerate(self.commit_results):
                    try:
                        participant_type = type(self.participants[i]).__name__ if i < len(self.participants) else "UNKNOWN"
                        logger.info(f"[2PC]   Participant {i} ({participant_type}): success={success}, msg={msg}")
                    except Exception as e:
                        logger.error(f"[2PC] Error logging commit result for participant {i}: {e}")
                        logger.info(f"[2PC]   Participant {i} (UNKNOWN): success={success}, msg={msg}")
                
                # Check if all participants committed successfully
                all_committed = True
                error_messages = []
                
                for i, (success, msg) in enumerate(self.commit_results):
                    if not success and msg != "Not committed":
                        all_committed = False
                        try:
                            participant_type = type(self.participants[i]).__name__ if i < len(self.participants) else "UNKNOWN"
                            error_messages.append(f"Participant {i} ({participant_type}): {msg}")
                        except Exception as e:
                            error_messages.append(f"Participant {i} (UNKNOWN): {msg} (Error getting type: {e})")
                
                if all_committed:
                    logger.info("[2PC] All participants committed successfully")
                else:
                    error_msg = "; ".join(error_messages)
                    logger.error(f"[2PC] Commit failed: {error_msg}")
                    raise RuntimeError(f"Commit failed: {error_msg}")
                return True
            else:
                # At least one participant failed to prepare, abort
                error_msgs = [
                    msg for i, (success, msg) in enumerate(self.prepare_results) 
                    if not success and msg
                ]
                error_msg = "Prepare failed: " + "; ".join(error_msgs)
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        except Exception as e:
            logger.error(f"Error during 2PC execution: {e}", exc_info=True)
            try:
                await self._abort()
            except Exception as abort_error:
                logger.error(f"Error during abort: {abort_error}", exc_info=True)
            # Re-raise the original exception to ensure it propagates to the caller
            raise
    
    async def _prepare(self) -> None:
        """Execute the prepare phase."""
        async with self._lock:
            self.state = TwoPhaseState.PREPARING
            
            # Log the operation to WAL if available
            if self.wal:
                await self.wal.log_operation(
                    op_type=self.operation_type,
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
            logger.debug(f"[2PC] Starting commit phase for operation {self.operation_id}")
            
            try:
                # Log detailed state information
                logger.debug(f"[2PC] Commit phase started with {len(self.participants)} total participants")
                logger.debug(f"[2PC] Prepare results (count={len(self.prepare_results)}): {self.prepare_results}")
                
                # Log each participant and their prepare status
                for i, (participant, (success, msg)) in enumerate(zip(self.participants, self.prepare_results)):
                    logger.debug(f"[2PC] Participant {i}: {type(participant).__name__}, "
                                f"prepare_success={success}, message={msg}")
                
                # Initialize commit_results with prepare results
                # We'll maintain a direct mapping from participant index to commit result
                self.commit_results = [(False, "Not committed")] * len(self.participants)
                
                # Track which participants to commit (those that prepared successfully)
                participants_to_commit = []
                
                # First, mark all participants that failed to prepare
                for i, (success, msg) in enumerate(self.prepare_results):
                    if success:
                        # This participant prepared successfully, add to commit list
                        participants_to_commit.append((i, self.participants[i]))
                        logger.debug(f"[2PC] Will commit participant {i} ({type(self.participants[i]).__name__})")
                    else:
                        # Mark failed participants in commit_results
                        self.commit_results[i] = (False, f"Prepare failed: {msg}")
                        logger.debug(f"[2PC] Skipping participant {i} ({type(self.participants[i]).__name__}) - prepare failed: {msg}")
                
                # Log the final state before starting commit
                logger.debug(f"[2PC] Total participants to commit: {len(participants_to_commit)} out of {len(self.participants)}")
                logger.debug(f"[2PC] Total prepare results: {len(self.prepare_results)}")
                logger.debug(f"[2PC] Total commit results: {len(self.commit_results)}")
                
                # Log detailed information about each participant
                for i, participant in enumerate(self.participants):
                    prep_success, prep_msg = self.prepare_results[i] if i < len(self.prepare_results) else (False, "No prepare result")
                    commit_success, commit_msg = self.commit_results[i] if i < len(self.commit_results) else (False, "No commit result")
                    logger.debug(
                        f"[2PC] Participant {i} ({type(participant).__name__}): "
                        f"prepare=({prep_success}, {prep_msg}), "
                        f"commit=({commit_success}, {commit_msg})"
                    )
                
                if not participants_to_commit:
                    logger.warning("[2PC] No participants to commit - all participants either failed or were skipped")
                    return
                
                # Process participants in small batches to avoid overwhelming the system
                BATCH_SIZE = 5  # Process 5 participants at a time
                total_batches = (len(participants_to_commit) + BATCH_SIZE - 1) // BATCH_SIZE
                logger.debug(f"[2PC] Processing {len(participants_to_commit)} participants in {total_batches} batches of up to {BATCH_SIZE}")
                
                for batch_num, batch_start in enumerate(range(0, len(participants_to_commit), BATCH_SIZE), 1):
                    batch_end = min(batch_start + BATCH_SIZE, len(participants_to_commit))
                    batch = participants_to_commit[batch_start:batch_end]
                    
                    logger.debug(f"[2PC] Processing batch {batch_num} with {len(batch)} participants")
                    
                    # Create commit tasks for this batch
                    commit_tasks = []
                    for participant_idx, participant in batch:
                        try:
                            logger.debug(f"[2PC] Creating commit task for participant {participant_idx} ({type(participant).__name__})")
                            task = asyncio.create_task(participant.commit(self))
                            commit_tasks.append((participant_idx, participant, task))
                        except Exception as e:
                            error_msg = f"Error creating commit task: {str(e)}"
                            logger.error(f"[2PC] {error_msg}", exc_info=True)
                            # Update the result for this participant
                            self.commit_results[participant_idx] = (False, error_msg)
                    
                    if not commit_tasks:
                        logger.debug("[2PC] No commit tasks to process in this batch")
                        continue
                    
                    # Process tasks as they complete
                    pending = {task for _, _, task in commit_tasks}
                    logger.debug(f"[2PC] Starting to process {len(pending)} commit tasks")
                    
                    # Create a mapping of task to (participant_idx, participant) for easy lookup
                    task_map = {task: (idx, p) for idx, p, task in commit_tasks}
                    
                    while pending:
                        done, pending = await asyncio.wait(
                            pending, 
                            return_when=asyncio.FIRST_COMPLETED,
                            timeout=30.0  # Add timeout to prevent hanging
                        )
                        logger.debug(f"[2PC] Completed {len(done)} tasks, {len(pending)} remaining")
                        
                        if not done:  # Timeout occurred
                            logger.warning("[2PC] Timeout waiting for commit tasks to complete")
                            break
                        
                        for completed_task in done:
                            if completed_task not in task_map:
                                logger.error("[2PC] Completed task not found in task map")
                                continue
                            
                            try:
                                participant_idx, participant = task_map[completed_task]
                                logger.debug(f"[2PC] Processing completed task for participant {participant_idx} ({type(participant).__name__})")
                                
                                # Get the result from the completed task
                                success, msg = await completed_task
                                logger.debug(f"[2PC] Commit result for participant {participant_idx} ({type(participant).__name__}): success={success}, msg={msg}")
                                
                                # Update the result directly using the participant's index
                                if 0 <= participant_idx < len(self.commit_results):
                                    self.commit_results[participant_idx] = (success, msg)
                                    logger.debug(f"[2PC] Updated commit_results[{participant_idx}] = ({success}, {msg})")
                                else:
                                    logger.error(f"[2PC] CRITICAL: Invalid participant index {participant_idx} for commit result")
                            except Exception as e:
                                error_msg = f"Error in commit task: {str(e)}"
                                logger.error(f"[2PC] {error_msg} for participant {participant_idx} ({type(participant).__name__})", exc_info=True)
                                if 0 <= participant_idx < len(self.commit_results):
                                    self.commit_results[participant_idx] = (False, error_msg)
                    
                    logger.debug(f"[2PC] Batch {batch_num} commit results: {[f'{i}:{r}' for i, r in enumerate(self.commit_results) if r[1] != 'Not committed']}")
                
                logger.debug(f"[2PC] Final commit results: {[f'{i}:{r}' for i, r in enumerate(self.commit_results)]}")
                
            except Exception as e:
                logger.error(f"[2PC] Critical error during commit phase: {e}", exc_info=True)
                # Ensure we have a consistent state even if an error occurs
                if not hasattr(self, 'commit_results') or not self.commit_results:
                    self.commit_results = [(False, f"Commit phase failed: {e}")] * len(self.participants)
                raise
            
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
