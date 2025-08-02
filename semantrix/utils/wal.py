import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class OperationType(str, Enum):
    SET = "SET"
    DELETE = "DELETE"
    BULK_SET = "BULK_SET"
    BULK_DELETE = "BULK_DELETE"

class LogStatus(str, Enum):
    PENDING = "PENDING"
    COMMITTED = "COMMITTED"
    FAILED = "FAILED"

class WriteAheadLog:
    """Async Write-Ahead Log implementation with batching and persistence."""
    
    def __init__(
        self,
        log_dir: str = "./wal_logs",
        max_log_size_mb: int = 100,  # Max size before rotation
        fsync: bool = True,  # Whether to fsync after each write
        batch_size: int = 100,  # Max entries per batch
        batch_timeout_seconds: float = 1.0,  # Max time to wait before flushing batch
    ):
        self.log_dir = Path(log_dir)
        self.max_log_size = max_log_size_mb * 1024 * 1024  # Convert MB to bytes
        self.fsync = fsync
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout_seconds
        
        # Current log file and position
        self.current_log_file: Optional[Path] = None
        self.current_log_fd = None
        self.current_log_size = 0
        
        # Batch processing
        self.batch: List[Dict[str, Any]] = []
        self.batch_lock = asyncio.Lock()
        self.batch_flush_task: Optional[asyncio.Task] = None
        self.shutdown_event = asyncio.Event()
        
        # Track processed request IDs for idempotency
        self.processed_requests = set()
        
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Start background tasks
        self.background_tasks = set()
        
    async def initialize(self):
        """Initialize WAL and recover any pending operations."""
        # Rotate to a new log file
        await self._rotate_log()
        
        # Start batch flusher
        self.batch_flush_task = asyncio.create_task(self._batch_flusher())
        self.batch_flush_task.add_done_callback(self._handle_task_exception)
        
        # Recover any pending operations from previous run
        await self._recover()
    
    async def shutdown(self):
        """Gracefully shutdown WAL, ensuring all writes are flushed."""
        self.shutdown_event.set()
        if self.batch_flush_task:
            await self.batch_flush_task
            self.batch_flush_task = None
        
        if self.current_log_fd:
            self.current_log_fd.close()
            self.current_log_fd = None
    
    async def log_operation(
        self,
        op_type: OperationType,
        data: Dict[str, Any],
        request_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """Log an operation to the WAL."""
        request_id = request_id or str(uuid.uuid4())
        
        # Skip if we've already processed this request
        if request_id in self.processed_requests:
            logger.debug(f"Skipping duplicate request: {request_id}")
            return request_id
            
        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "operation": op_type.value,
            "status": LogStatus.PENDING.value,
            "data": data,
            **kwargs
        }
        
        # Add to batch
        async with self.batch_lock:
            self.batch.append(log_entry)
            
            # Flush if batch is full
            if len(self.batch) >= self.batch_size:
                await self._flush_batch()
        
        return request_id
    
    async def commit_operation(self, request_id: str) -> bool:
        """Mark an operation as committed."""
        return await self._update_status(request_id, LogStatus.COMMITTED)
    
    async def fail_operation(self, request_id: str, error: str) -> bool:
        """Mark an operation as failed."""
        return await self._update_status(request_id, LogStatus.FAILED, {"error": error})
    
    async def _update_status(
        self, 
        request_id: str, 
        status: LogStatus,
        extra_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update the status of a logged operation."""
        log_entry = {
            "request_id": request_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status": status.value,
            "operation": "STATUS_UPDATE",
            "original_request_id": request_id,
            "data": extra_data or {}
        }
        
        # Write status update to log
        await self._write_log_entry(log_entry)
        
        # Add to processed requests if committed
        if status == LogStatus.COMMITTED:
            self.processed_requests.add(request_id)
            
        return True
    
    async def _batch_flusher(self):
        """Background task to flush the batch periodically."""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(self.batch_timeout)
                await self._flush_batch()
            except Exception as e:
                logger.error(f"Error in batch flusher: {e}")
                await asyncio.sleep(1)  # Prevent tight loop on errors
    
    async def _flush_batch(self):
        """Flush the current batch to disk."""
        if not self.batch:
            return
            
        async with self.batch_lock:
            batch = self.batch
            self.batch = []
            
        for entry in batch:
            await self._write_log_entry(entry)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.1, max=1.0)
    )
    async def _write_log_entry(self, entry: Dict[str, Any]):
        """Write a single log entry to the current log file."""
        if not self.current_log_fd:
            await self._rotate_log()
            
        log_line = json.dumps(entry) + "\n"
        log_bytes = log_line.encode('utf-8')
        
        try:
            self.current_log_fd.write(log_bytes)
            if self.fsync:
                os.fsync(self.current_log_fd.fileno())
            self.current_log_size += len(log_bytes)
            
            # Rotate if needed
            if self.current_log_size >= self.max_log_size:
                await self._rotate_log()
                
        except IOError as e:
            logger.error(f"Failed to write to WAL: {e}")
            raise
    
    async def _rotate_log(self):
        """Rotate to a new log file."""
        if self.current_log_fd:
            self.current_log_fd.close()
            
        # Create new log file with timestamp
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.current_log_file = self.log_dir / f"wal_{timestamp}.log"
        
        # Open new log file
        self.current_log_fd = open(self.current_log_file, 'a', buffering=1)  # Line buffered
        self.current_log_size = os.fstat(self.current_log_fd.fileno()).st_size
    
    async def _recover(self):
        """Recover pending operations from WAL files."""
        # Sort log files by creation time (oldest first)
        log_files = sorted(self.log_dir.glob("wal_*.log"))
        
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        try:
                            entry = json.loads(line)
                            request_id = entry.get('request_id')
                            
                            # Track processed requests
                            if entry.get('status') == LogStatus.COMMITTED.value and request_id:
                                self.processed_requests.add(request_id)
                                
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON in WAL file {log_file}")
                            continue
                            
            except IOError as e:
                logger.error(f"Failed to read WAL file {log_file}: {e}")
    
    def _handle_task_exception(self, task):
        """Handle exceptions in background tasks."""
        try:
            task.result()  # This will raise any exceptions
        except asyncio.CancelledError:
            pass  # Task was cancelled
        except Exception as e:
            logger.error(f"Background task failed: {e}", exc_info=True)
            # Optionally restart the task
            if not self.shutdown_event.is_set():
                self.batch_flush_task = asyncio.create_task(self._batch_flusher())
                self.batch_flush_task.add_done_callback(self._handle_task_exception)

    async def __aenter__(self):
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()

# Helper function to create a WAL instance with default settings
async def create_wal(
    log_dir: str = "./wal_logs",
    max_log_size_mb: int = 100,
    fsync: bool = True,
    batch_size: int = 100,
    batch_timeout_seconds: float = 1.0,
) -> WriteAheadLog:
    """Create and initialize a new WriteAheadLog instance."""
    wal = WriteAheadLog(
        log_dir=log_dir,
        max_log_size_mb=max_log_size_mb,
        fsync=fsync,
        batch_size=batch_size,
        batch_timeout_seconds=batch_timeout_seconds,
    )
    await wal.initialize()
    return wal
