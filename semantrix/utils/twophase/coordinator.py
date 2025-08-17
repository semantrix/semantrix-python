"""
Two-Phase Commit coordinator for atomic operations across multiple resources.
"""

import asyncio
import uuid
from typing import Any, Dict, List, Optional

from semantrix.utils.wal import OperationType
from .operation import TwoPhaseOperation, TwoPhaseState
from .participant import Participant
from .interfaces import WALInterface
from semantrix.utils.logging import get_logger

logger = get_logger(__name__)


class TwoPhaseCoordinator:
    """Coordinates Two-Phase Commit operations."""
    
    def __init__(self, wal: Optional[WALInterface] = None):
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
            operation_id=op_id,
            operation_type=operation_type.value,
            data=data,
            participants=participants
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
                operation_id=op_id,
                operation_type=op_data['operation_type'],
                data=op_data['data'],
                participants=[]  # Participants need to be re-attached by the caller
            )
            operation.state = TwoPhaseState.PREPARED  # Assume prepared if in WAL
            
            async with self._lock:
                self.active_operations[op_id] = operation
            
            recovered_ops.append(operation)
        
        return recovered_ops
    
    async def execute_operation(self, operation: TwoPhaseOperation) -> bool:
        """Execute a Two-Phase Commit operation."""
        try:
            # Phase 1: Prepare
            logger.debug(f"Starting prepare phase for operation {operation.operation_id}")
            await self._prepare(operation)
            
            # Check if all participants prepared successfully
            if all(success for success, _ in operation.prepare_results):
                logger.debug("All participants prepared successfully, starting commit phase")
                # Phase 2: Commit
                await self._commit(operation)
                return True
            else:
                # At least one participant failed to prepare, abort
                error_msgs = [
                    msg for i, (success, msg) in enumerate(operation.prepare_results) 
                    if not success and msg
                ]
                error_msg = "Prepare failed: " + "; ".join(error_msgs)
                logger.error(error_msg)
                await self._abort(operation)
                raise RuntimeError(error_msg)
        except Exception as e:
            logger.error(f"Error during 2PC execution: {e}", exc_info=True)
            try:
                await self._abort(operation)
            except Exception as abort_error:
                logger.error(f"Error during abort: {abort_error}", exc_info=True)
            # Re-raise the original exception to ensure it propagates to the caller
            raise
    
    async def _prepare(self, operation: TwoPhaseOperation) -> None:
        """Execute the prepare phase."""
        async with operation._lock:
            operation.state = TwoPhaseState.PREPARING
            
            # Log the operation to WAL if available
            if self.wal:
                await self.wal.log_operation(
                    op_type=OperationType(operation.operation_type),
                    data=operation.data,
                    request_id=operation.operation_id
                )
            
            # Prepare all participants
            prepare_tasks = [
                participant.prepare(operation) 
                for participant in operation.participants
            ]
            operation.prepare_results = await asyncio.gather(
                *prepare_tasks, 
                return_exceptions=True
            )
            
            # Convert exceptions to failure results
            for i, result in enumerate(operation.prepare_results):
                if isinstance(result, Exception):
                    operation.prepare_results[i] = (False, str(result))
            
            operation.state = TwoPhaseState.PREPARED
    
    async def _commit(self, operation: TwoPhaseOperation) -> None:
        """Execute the commit phase."""
        async with operation._lock:
            operation.state = TwoPhaseState.COMMITTING
            logger.debug(f"[2PC] Starting commit phase for operation {operation.operation_id}")
            
            try:
                # Log detailed state information
                logger.debug(f"[2PC] Commit phase started with {len(operation.participants)} total participants")
                logger.debug(f"[2PC] Prepare results (count={len(operation.prepare_results)}): {operation.prepare_results}")
                
                # Log each participant and their prepare status
                for i, (participant, (success, msg)) in enumerate(zip(operation.participants, operation.prepare_results)):
                    logger.debug(f"[2PC] Participant {i}: {type(participant).__name__}, "
                                f"prepare_success={success}, message={msg}")
                
                # Initialize commit_results with prepare results
                # We'll maintain a direct mapping from participant index to commit result
                operation.commit_results = [(False, "Not committed")] * len(operation.participants)
                
                # Track which participants to commit (those that prepared successfully)
                participants_to_commit = []
                
                # First, mark all participants that failed to prepare
                for i, (success, msg) in enumerate(operation.prepare_results):
                    if success:
                        # This participant prepared successfully, add to commit list
                        participants_to_commit.append((i, operation.participants[i]))
                        logger.debug(f"[2PC] Will commit participant {i} ({type(operation.participants[i]).__name__})")
                    else:
                        # Mark failed participants in commit_results
                        operation.commit_results[i] = (False, f"Prepare failed: {msg}")
                        logger.debug(f"[2PC] Skipping participant {i} ({type(operation.participants[i]).__name__}) - prepare failed: {msg}")
                
                # Log the final state before starting commit
                logger.debug(f"[2PC] Total participants to commit: {len(participants_to_commit)} out of {len(operation.participants)}")
                logger.debug(f"[2PC] Total prepare results: {len(operation.prepare_results)}")
                logger.debug(f"[2PC] Total commit results: {len(operation.commit_results)}")
                
                # Log detailed information about each participant
                for i, participant in enumerate(operation.participants):
                    prep_success, prep_msg = operation.prepare_results[i] if i < len(operation.prepare_results) else (False, "No prepare result")
                    commit_success, commit_msg = operation.commit_results[i] if i < len(operation.commit_results) else (False, "No commit result")
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
                            task = asyncio.create_task(participant.commit(operation))
                            commit_tasks.append((participant_idx, participant, task))
                        except Exception as e:
                            error_msg = f"Error creating commit task: {str(e)}"
                            logger.error(f"[2PC] {error_msg}", exc_info=True)
                            # Update the result for this participant
                            operation.commit_results[participant_idx] = (False, error_msg)
                    
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
                                if 0 <= participant_idx < len(operation.commit_results):
                                    operation.commit_results[participant_idx] = (success, msg)
                                    logger.debug(f"[2PC] Updated commit_results[{participant_idx}] = ({success}, {msg})")
                                else:
                                    logger.error(f"[2PC] CRITICAL: Invalid participant index {participant_idx} for commit result")
                            except Exception as e:
                                error_msg = f"Error in commit task: {str(e)}"
                                logger.error(f"[2PC] {error_msg} for participant {participant_idx} ({type(participant).__name__})", exc_info=True)
                                if 0 <= participant_idx < len(operation.commit_results):
                                    operation.commit_results[participant_idx] = (False, error_msg)
                    
                    logger.debug(f"[2PC] Batch {batch_num} commit results: {[f'{i}:{r}' for i, r in enumerate(operation.commit_results) if r[1] != 'Not committed']}")
                
                logger.debug(f"[2PC] Final commit results: {[f'{i}:{r}' for i, r in enumerate(operation.commit_results)]}")
                
            except Exception as e:
                logger.error(f"[2PC] Critical error during commit phase: {e}", exc_info=True)
                # Ensure we have a consistent state even if an error occurs
                if not hasattr(operation, 'commit_results') or not operation.commit_results:
                    operation.commit_results = [(False, f"Commit phase failed: {e}")] * len(operation.participants)
                raise
            
            # Update WAL if available
            if self.wal:
                if all(success for success, _ in operation.commit_results):
                    await self.wal.commit_operation(operation.operation_id)
                else:
                    await self.wal.fail_operation(
                        operation.operation_id, 
                        "Failed to commit all participants"
                    )
            
            operation.state = TwoPhaseState.COMMITTED
    
    async def _abort(self, operation: TwoPhaseOperation) -> None:
        """Abort the operation."""
        async with operation._lock:
            if operation.state in (TwoPhaseState.ABORTING, TwoPhaseState.ABORTED, TwoPhaseState.FAILED):
                return
                
            operation.state = TwoPhaseState.ABORTING
            
            # Rollback all participants that were prepared
            rollback_tasks = []
            for i, participant in enumerate(operation.participants):
                if i < len(operation.prepare_results) and operation.prepare_results[i][0]:
                    rollback_tasks.append(participant.rollback(operation))
            
            # Wait for rollbacks to complete
            rollback_results = await asyncio.gather(
                *rollback_tasks,
                return_exceptions=True
            )
            
            # Process rollback results
            result_idx = 0
            for i in range(len(operation.participants)):
                if i < len(operation.prepare_results) and operation.prepare_results[i][0]:
                    result = rollback_results[result_idx]
                    if isinstance(result, Exception):
                        operation.rollback_results.append((False, str(result)))
                    else:
                        operation.rollback_results.append(result)
                    result_idx += 1
            
            # Update WAL if available
            if self.wal:
                await self.wal.fail_operation(
                    operation.operation_id,
                    "Operation aborted"
                )
            
            operation.state = TwoPhaseState.ABORTED
