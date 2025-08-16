"""
Two-Phase Commit (2PC) implementation for Semantrix.

This module provides atomic operations across multiple resources using
the Two-Phase Commit protocol with optional Write-Ahead Logging for
crash recovery.
"""

from .coordinator import TwoPhaseCoordinator
from .participant import Participant
from .operation import TwoPhaseOperation, TwoPhaseState
from .participants import CacheStoreParticipant, VectorStoreParticipant
from .interfaces import (
    WALInterface,
    ParticipantInterface,
    OperationInterface,
    StateInterface,
    CoordinatorInterface,
    ParticipantFactory,
    WALFactory
)
from .wal_adapter import (
    WALAdapter,
    NoOpWAL,
    MemoryWAL,
    DefaultWALFactory,
    create_wal_adapter,
    create_wal_from_config
)

__all__ = [
    'TwoPhaseCoordinator',
    'Participant', 
    'TwoPhaseOperation',
    'TwoPhaseState',
    'CacheStoreParticipant',
    'VectorStoreParticipant',
    # Interfaces
    'WALInterface',
    'ParticipantInterface',
    'OperationInterface',
    'StateInterface',
    'CoordinatorInterface',
    'ParticipantFactory',
    'WALFactory',
    # WAL Adapters
    'WALAdapter',
    'NoOpWAL',
    'MemoryWAL',
    'DefaultWALFactory',
    'create_wal_adapter',
    'create_wal_from_config'
]
