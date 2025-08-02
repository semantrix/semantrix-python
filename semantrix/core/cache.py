import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple, Union, AsyncGenerator

from semantrix.embedding import BaseEmbedder, get_embedder
from semantrix.vector_store import BaseVectorStore, FAISSVectorStore
from semantrix.cache_store import BaseCacheStore, InMemoryStore
from semantrix.utils.resource_limits import ResourceLimits
from semantrix.utils.profiling import Profiler
from semantrix.utils.wal import WriteAheadLog, OperationType, create_wal
from semantrix.utils.twophase import TwoPhaseCoordinator, Participant, TwoPhaseOperation, TwoPhaseState
from semantrix.models.explain import ExplainResult, CacheMatch, create_explain_result

# Set up logging
logger = logging.getLogger(__name__)

class CacheStoreParticipant(Participant):
    """Participant for cache store operations in 2PC."""
    
    def __init__(self, cache_store: BaseCacheStore):
        self.cache_store = cache_store
    
    async def prepare(self, operation: TwoPhaseOperation) -> Tuple[bool, Optional[str]]:
        try:
            # For cache store, prepare is a no-op as we can't lock individual keys
            return True, None
        except Exception as e:
            return False, str(e)
    
    async def commit(self, operation: TwoPhaseOperation) -> Tuple[bool, Optional[str]]:
        try:
            if operation.operation_type == OperationType.ADD:
                await self.cache_store.set(
                    operation.data['prompt'],
                    operation.data['response']
                )
            elif operation.operation_type == OperationType.DELETE:
                await self.cache_store.delete(operation.data['prompt'])
            return True, None
        except Exception as e:
            return False, str(e)
    
    async def rollback(self, operation: TwoPhaseOperation) -> Tuple[bool, Optional[str]]:
        # For cache store, we can't rollback a set/delete operation
        # The best we can do is log the issue
        return True, None

class VectorStoreParticipant(Participant):
    """Participant for vector store operations in 2PC."""
    
    def __init__(self, vector_store: BaseVectorStore):
        self.vector_store = vector_store
    
    async def prepare(self, operation: TwoPhaseOperation) -> Tuple[bool, Optional[str]]:
        try:
            # For vector store, prepare is a no-op as we can't lock individual vectors
            return True, None
        except Exception as e:
            return False, str(e)
    
    async def commit(self, operation: TwoPhaseOperation) -> Tuple[bool, Optional[str]]:
        try:
            if operation.operation_type == OperationType.ADD:
                await self.vector_store.add(
                    operation.data['embedding'],
                    operation.data['prompt']
                )
            elif operation.operation_type == OperationType.DELETE:
                await self.vector_store.delete(operation.data['prompt'])
            return True, None
        except Exception as e:
            return False, str(e)
    
    async def rollback(self, operation: TwoPhaseOperation) -> Tuple[bool, Optional[str]]:
        try:
            # Try to remove the vector if it was added
            if operation.operation_type == OperationType.ADD:
                await self.vector_store.delete(operation.data['prompt'])
            return True, None
        except Exception as e:
            return False, str(e)

class Semantrix:
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        resource_limits: ResourceLimits = ResourceLimits(),
        enable_profiling: bool = False,
        enable_wal: bool = True,
        wal_config: Optional[Dict[str, Any]] = None,
        enable_2pc: bool = True,
        embedder: Optional[BaseEmbedder] = None,
        vector_store: Optional[BaseVectorStore] = None,
        cache_store: Optional[BaseCacheStore] = None
    ):
        """
        Initialize Semantrix semantic cache.
        
        Args:
            similarity_threshold: Cosine similarity threshold for semantic match
            resource_limits: Resource limits configuration
            enable_profiling: Enable lightweight profiling
            enable_wal: Enable Write-Ahead Logging for crash recovery
            wal_config: Configuration for WAL (log_dir, max_log_size_mb, etc.)
            enable_2pc: Enable Two-Phase Commit for atomic operations
            embedder: Custom embedder implementation (defaults to sentence-transformers)
            vector_store: Custom vector store implementation (defaults to FAISS)
            cache_store: Custom cache store implementation (defaults to InMemoryStore)
        """
        # Initialize components with defaults if not provided
        self.embedder = embedder or get_embedder("sentence-transformers")
        self.vector_store = vector_store or FAISSVectorStore(dimension=self.embedder.get_dimension())
        self.cache_store = cache_store or InMemoryStore()
        
        # Initialize WAL if enabled
        self.wal = None
        self.enable_wal = enable_wal
        self.wal_config = wal_config or {}
        
        # Initialize 2PC coordinator if enabled
        self.enable_2pc = enable_2pc
        self.coordinator = None
        
        self.resource_limits = resource_limits
        self.profiler = Profiler(enabled=enable_profiling)
        self.similarity_threshold = similarity_threshold
        
        # Track in-progress operations for idempotency
        self._pending_operations: Dict[str, asyncio.Task] = {}
        self._operation_lock = asyncio.Lock()
        self._initialized = False

    async def initialize(self):
        """Initialize the cache and its components."""
        if self._initialized:
            return
            
        # Initialize WAL if enabled
        if self.enable_wal and self.wal is None:
            self.wal = await create_wal(**self.wal_config)
            
            # Initialize 2PC coordinator with WAL
            if self.enable_2pc:
                self.coordinator = TwoPhaseCoordinator(self.wal)
                
                # Recover any pending operations
                await self._recover_pending_operations()
        
        self._initialized = True
    
    async def shutdown(self):
        """Gracefully shutdown the cache and its components."""
        if self.wal:
            await self.wal.shutdown()
    
    async def __aenter__(self):
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.shutdown()
    
    @asynccontextmanager
    async def _operation_ctx(self, operation_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Context manager for operation tracking and idempotency."""
        op_id = operation_id or str(uuid.uuid4())
        
        async with self._operation_lock:
            if op_id in self._pending_operations:
                # If operation is already in progress, wait for it to complete
                existing_op = self._pending_operations[op_id]
                try:
                    await existing_op
                    yield op_id
                    return
                except Exception as e:
                    # If the existing operation failed, we'll retry
                    logger.warning("Previous operation %s failed: %s", op_id, e)
            
            # Create a new task for this operation
            task = asyncio.current_task()
            if task is None:
                raise RuntimeError("Operation must be run in an async context")
                
            self._pending_operations[op_id] = task
        
        try:
            yield op_id
        finally:
            async with self._operation_lock:
                self._pending_operations.pop(op_id, None)
    
    async def _recover_pending_operations(self):
        """Recover any pending operations from WAL."""
        if not self.wal or not self.coordinator:
            return
            
        # Recover operations from WAL
        recovered_ops = await self.coordinator.recover_operations()
        
        # Re-execute recovered operations
        for operation in recovered_ops:
            # Re-attach participants
            operation.participants = [
                CacheStoreParticipant(self.cache_store),
                VectorStoreParticipant(self.vector_store)
            ]
            
            # Re-execute the operation
            try:
                logger.info(f"Replaying operation: {operation.operation_id}")
                success = await operation.execute()
                if not success:
                    logger.error(f"Failed to recover operation: {operation.operation_id}")
            except Exception as e:
                logger.error(f"Error recovering operation {operation.operation_id}: {e}", exc_info=True)
    
    async def _execute_with_wal(
        self,
        operation_type: OperationType,
        data: Dict[str, Any],
        operation_id: Optional[str] = None,
        **kwargs
    ) -> str:
        """Execute an operation with WAL and 2PC support."""
        if not self.wal or not self.coordinator:
            # Fall back to non-atomic execution if WAL or 2PC is disabled
            return await self._execute_operation(operation_type, data, operation_id or str(uuid.uuid4()))
        
        async with self._operation_ctx(operation_id) as op_id:
            # Create participants
            participants = [
                CacheStoreParticipant(self.cache_store),
                VectorStoreParticipant(self.vector_store)
            ]
            
            # Create and execute 2PC operation
            operation = await self.coordinator.create_operation(
                operation_type=operation_type,
                data=data,
                participants=participants,
                operation_id=op_id
            )
            
            try:
                success = await operation.execute()
                if not success:
                    raise RuntimeError("Failed to execute operation atomically")
                return op_id
                
            except Exception as e:
                # Ensure operation is aborted on failure
                if operation.state not in (TwoPhaseState.ABORTED, TwoPhaseState.FAILED):
                    await operation._abort()
                raise
    
    async def _execute_operation(
        self,
        operation_type: OperationType,
        data: Dict[str, Any],
        operation_id: str
    ) -> Any:
        """Execute the actual operation logic without 2PC.
        
        This is used as a fallback when WAL or 2PC is disabled.
        
        Args:
            operation_type: Type of operation to execute
            data: Operation data
            operation_id: Unique ID for the operation
            
        Returns:
            Result of the operation
            
        Raises:
            ValueError: If the operation type is not supported
        """
        if operation_type == OperationType.ADD:
            # Add operation
            prompt = data['prompt']
            response = data['response']
            embedding = data.get('embedding')
            
            if embedding is None:
                # Generate embedding if not provided
                embedding = await self.embedder.embed(prompt)
            
            # Store in vector store and cache store
            # Note: This is not atomic!
            try:
                await self.vector_store.add(embedding, prompt)
                await self.cache_store.set(prompt, response)
                return {'success': True}
            except Exception as e:
                # Try to clean up if one operation succeeds and the other fails
                try:
                    await self.vector_store.delete(prompt)
                except Exception:
                    pass
                try:
                    await self.cache_store.delete(prompt)
                except Exception:
                    pass
                raise
            
        elif operation_type == OperationType.DELETE:
            # Delete operation
            prompt = data['prompt']
            
            # Note: This is not atomic!
            try:
                await self.vector_store.delete(prompt)
                await self.cache_store.delete(prompt)
                return {'success': True}
            except Exception as e:
                # If one operation fails, we can't do much
                logger.error("Error during delete operation: %s", e)
                raise
            
        else:
            raise ValueError(f"Unsupported operation type: {operation_type}")
    
    async def get(self, prompt: str, operation_id: Optional[str] = None) -> Optional[str]:
        """
        Get a cached response for the given prompt asynchronously.
        
        Args:
            prompt: The prompt to search for
            operation_id: Optional operation ID for idempotency
            
        Returns:
            Optional[str]: The cached response if found, None otherwise
        """
        # Check if we have an exact match first (fast path)
        if exact_match := await self.cache_store.get(prompt):
            return exact_match
            
        # If no exact match, try semantic search
        return await self._get_semantic(prompt)

    async def _get_semantic(self, prompt: str) -> Optional[str]:
        with self.profiler.record("get"):
            # 1. Check resource constraints
            if not self.resource_limits.allow_operation():
                return None
                
            # 2. Semantic search (if vector store is available)
            if hasattr(self.embedder, 'aencode') and hasattr(self.vector_store, 'search'):
                try:
                    # Use async embedding
                    embedding = await self.embedder.aencode(prompt)
                    
                    # Search in vector store
                    match = await self.vector_store.search(embedding, self.similarity_threshold)
                    if isinstance(match, str):
                        return match
                except Exception as e:
                    logger.error(f"Error during semantic search: {e}")
                    # Continue to return None if there's an error
                    
            return None

    async def set(self, prompt: str, response: str, operation_id: Optional[str] = None) -> None:
        """
        Add a prompt-response pair to the cache.
        
        Args:
            prompt: The prompt to cache
            response: The response to cache
            operation_id: Optional operation ID for idempotency
        """
        # Generate embedding for the prompt
        embedding = await self.embedder.embed(prompt)
        
        # Prepare operation data for WAL
        operation_data = {
            'prompt': prompt,
            'response': response,
            'embedding': embedding.tolist() if hasattr(embedding, 'tolist') else embedding
        }
        
        # Execute with WAL support
        await self._execute_with_wal(
            operation_type=OperationType.ADD,
            data=operation_data,
            operation_id=operation_id
        )

    async def explain(self, prompt: str) -> ExplainResult:
        """
        Explain why a prompt hit or missed the cache.
        
        Args:
            prompt: The prompt to explain
            
        Returns:
            ExplainResult: Explanation of the cache result
        """
        # Track timing and resource usage
        start_time = time.time()
        resource_warnings = []
        
        # Check for exact match first (fast path)
        exact_match_result = await self.cache_store.get(prompt)
        if exact_match_result is not None:
            return create_explain_result(
                hit=True,
                reason="Exact match found in cache",
                matches=[CacheMatch(prompt=prompt, similarity=1.0)],
                exact_match=True,
                cache_hit=True,
                similarity_threshold=self.similarity_threshold,
                total_time_ms=(time.time() - start_time) * 1000
            )
        
        # Generate embedding for semantic search
        embedding_start = time.time()
        embedding = await self.embedder.embed(prompt)
        embedding_time = (time.time() - embedding_start) * 1000
        
        # Perform semantic search
        search_start = time.time()
        matches = await self.vector_store.search(embedding, k=3)
        search_time = (time.time() - search_start) * 1000
        
        # Check resource limits
        if self.resource_limits.is_memory_high():
            resource_warnings.append("High memory usage detected")
        
        total_time = (time.time() - start_time) * 1000
        
        # Check for semantic matches above threshold
        if matches and matches[0].similarity >= self.similarity_threshold:
            return create_explain_result(
                hit=True,
                reason=f"Semantic match found (similarity: {matches[0].similarity:.2f})",
                matches=matches,
                exact_match=False,
                semantic_match=True,
                cache_hit=False,
                similarity_threshold=self.similarity_threshold,
                top_matches=len(matches),
                resource_limited=bool(resource_warnings),
                resource_warnings=resource_warnings,
                embedding_time_ms=embedding_time,
                search_time_ms=search_time,
                total_time_ms=total_time
            )
        
        # No matches found
        return create_explain_result(
            hit=False,
            reason="No match found above similarity threshold",
            matches=matches or [],
            exact_match=False,
            semantic_match=False,
            cache_hit=False,
            similarity_threshold=self.similarity_threshold,
            top_matches=len(matches) if matches else 0,
            resource_limited=bool(resource_warnings),
            resource_warnings=resource_warnings,
            embedding_time_ms=embedding_time,
            search_time_ms=search_time,
            total_time_ms=total_time
        ) 