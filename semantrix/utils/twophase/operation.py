"""
Two-Phase Commit operation and state definitions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional
from datetime import datetime


class TwoPhaseState(Enum):
    """States of a Two-Phase Commit operation."""
    INITIALIZED = "initialized"
    PREPARING = "preparing"
    PREPARED = "prepared"
    COMMITTING = "committing"
    COMMITTED = "committed"
    ABORTING = "aborting"
    ABORTED = "aborted"
    FAILED = "failed"


@dataclass
class TwoPhaseOperation:
    """
    Represents a Two-Phase Commit operation.
    
    This class encapsulates all the information needed to execute
    an atomic operation across multiple participants.
    """
    
    operation_id: str
    operation_type: str
    data: Dict[str, Any]
    participants: list
    state: TwoPhaseState = TwoPhaseState.INITIALIZED
    created_at: Optional[datetime] = None
    prepared_at: Optional[datetime] = None
    committed_at: Optional[datetime] = None
    aborted_at: Optional[datetime] = None
    error_message: Optional[str] = None
    prepare_results: list = None
    commit_results: list = None
    rollback_results: list = None
    _lock: Any = None
    
    def __post_init__(self):
        """Initialize timestamps and internal state after object creation."""
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        
        # Initialize internal state
        if self.prepare_results is None:
            self.prepare_results = []
        if self.commit_results is None:
            self.commit_results = []
        if self.rollback_results is None:
            self.rollback_results = []
        if self._lock is None:
            import asyncio
            self._lock = asyncio.Lock()
    
    def mark_prepared(self):
        """Mark the operation as prepared."""
        self.state = TwoPhaseState.PREPARED
        self.prepared_at = datetime.utcnow()
    
    def mark_committed(self):
        """Mark the operation as committed."""
        self.state = TwoPhaseState.COMMITTED
        self.committed_at = datetime.utcnow()
    
    def mark_aborted(self, error_message: Optional[str] = None):
        """Mark the operation as aborted."""
        self.state = TwoPhaseState.ABORTED
        self.aborted_at = datetime.utcnow()
        if error_message:
            self.error_message = error_message
    
    def mark_failed(self, error_message: str):
        """Mark the operation as failed."""
        self.state = TwoPhaseState.FAILED
        self.error_message = error_message
    
    def is_final_state(self) -> bool:
        """Check if the operation is in a final state."""
        return self.state in [TwoPhaseState.COMMITTED, TwoPhaseState.ABORTED, TwoPhaseState.FAILED]
    
    def can_prepare(self) -> bool:
        """Check if the operation can be prepared."""
        return self.state == TwoPhaseState.INITIALIZED
    
    def can_commit(self) -> bool:
        """Check if the operation can be committed."""
        return self.state == TwoPhaseState.PREPARED
    
    def can_abort(self) -> bool:
        """Check if the operation can be aborted."""
        return self.state in [TwoPhaseState.INITIALIZED, TwoPhaseState.PREPARING, TwoPhaseState.PREPARED]
