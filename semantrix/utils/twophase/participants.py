"""
Two-Phase Commit (2PC) participants for Semantrix operations.

This module contains participant implementations for different Semantrix components
that need to participate in atomic operations.
"""

from typing import Any, Dict, List, Optional, Tuple

from semantrix.utils.twophase import Participant, TwoPhaseOperation
from semantrix.utils.wal import OperationType
from semantrix.cache_store import BaseCacheStore
from semantrix.vector_store import BaseVectorStore
from semantrix.exceptions import (
    CacheOperationError,
    VectorOperationError,
    ValidationError
)
from semantrix.utils.logging import get_logger

logger = get_logger(__name__)


class CacheStoreParticipant(Participant):
    """Participant for cache store operations in 2PC."""
    
    def __init__(self, cache_store: BaseCacheStore):
        self.cache_store = cache_store
    
    async def prepare(self, operation: TwoPhaseOperation) -> Tuple[bool, Optional[str]]:
        """
        Prepare for the cache store operation.
        
        For cache store, prepare is a no-op as we can't lock individual keys.
        The actual work happens during commit.
        
        Args:
            operation: The 2PC operation to prepare for
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # For cache store, prepare is a no-op as we can't lock individual keys
            return True, None
        except Exception as e:
            err = CacheOperationError(f"Failed to prepare cache store: {e}", original_exception=e)
            return False, str(err)
    
    async def commit(self, operation: TwoPhaseOperation) -> Tuple[bool, Optional[str]]:
        """
        Commit the cache store operation.
        
        Args:
            operation: The 2PC operation to commit
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            if operation.operation_type == OperationType.SET:
                await self.cache_store.add(
                    operation.data['prompt'],
                    operation.data['response']
                )
            elif operation.operation_type == OperationType.DELETE:
                await self.cache_store.delete(operation.data['prompt'])
            else:
                return False, f"Unsupported operation type: {operation.operation_type}"
                
            return True, None
        except Exception as e:
            logger.error(f"Error in CacheStoreParticipant.commit: {e}", exc_info=True)
            # Wrap in CacheOperationError before returning string representation
            err = CacheOperationError(f"Failed to commit to cache store: {e}", original_exception=e)
            return False, str(err)
    
    async def rollback(self, operation: TwoPhaseOperation) -> Tuple[bool, Optional[str]]:
        """
        Rollback the cache store operation.
        
        Args:
            operation: The 2PC operation to rollback
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            if operation.operation_type == OperationType.SET:
                # Try to delete the key that was set
                await self.cache_store.delete(operation.data['prompt'])
            elif operation.operation_type == OperationType.DELETE:
                # Can't recover a delete, but we can log it
                logger.warning("Cannot recover from a failed DELETE operation in cache store")
            return True, None
        except Exception as e:
            logger.error(f"Error during cache store rollback: {e}", exc_info=True)
            err = CacheOperationError(f"Failed to rollback cache store: {e}", original_exception=e)
            return False, str(err)


class VectorStoreParticipant(Participant):
    """Participant for vector store operations in 2PC."""
    
    def __init__(self, vector_store: BaseVectorStore):
        self.vector_store = vector_store
    
    async def prepare(self, operation: TwoPhaseOperation) -> Tuple[bool, Optional[str]]:
        """
        Prepare for the vector store operation.
        
        For vector store, prepare is a no-op as we can't lock individual vectors.
        The actual work happens during commit.
        
        Args:
            operation: The 2PC operation to prepare for
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # For vector store, prepare is a no-op as we can't lock individual vectors
            return True, None
        except Exception as e:
            err = VectorOperationError(f"Failed to prepare vector store: {e}", original_exception=e)
            return False, str(err)
    
    async def commit(self, operation: TwoPhaseOperation) -> Tuple[bool, Optional[str]]:
        """
        Commit the vector store operation.
        
        Args:
            operation: The 2PC operation to commit
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            logger.debug(f"[2PC] VectorStoreParticipant.commit - Operation: {operation.operation_type}, "
                       f"Data keys: {list(operation.data.keys())}")
            
            if operation.operation_type == OperationType.SET:
                # Get the embedding and prompt
                if 'embedding' not in operation.data:
                    error_msg = "Missing 'embedding' in operation data"
                    logger.error(f"[2PC] {error_msg}")
                    return False, str(ValidationError(error_msg))
                    
                if 'prompt' not in operation.data:
                    error_msg = "Missing 'prompt' in operation data"
                    logger.error(f"[2PC] {error_msg}")
                    return False, str(ValidationError(error_msg))
                
                embedding = operation.data['embedding']
                prompt = operation.data['prompt']
                
                logger.debug(f"[2PC] VectorStore commit - Operation: {operation.operation_type}, "
                           f"Prompt: {prompt}, Embedding type: {type(embedding).__name__}, "
                           f"Embedding length: {len(embedding) if hasattr(embedding, '__len__') else 'N/A'}")
                
                # Ensure we have a list of embeddings
                import numpy as np
                if isinstance(embedding, (list, np.ndarray)):
                    if not isinstance(embedding, list) or (isinstance(embedding, list) and len(embedding) > 0 and not isinstance(embedding[0], (list, np.ndarray))):
                        # Single embedding case
                        embedding = np.array(embedding, dtype=np.float32)
                        if embedding.ndim == 1:
                            embedding = embedding.reshape(1, -1)  # Ensure 2D array
                        embeddings = [embedding]
                    else:
                        # Multiple embeddings case
                        embeddings = []
                        for emb in embedding:
                            emb_array = np.array(emb, dtype=np.float32)
                            if emb_array.ndim == 1:
                                emb_array = emb_array.reshape(1, -1)  # Ensure 2D array
                            embeddings.append(emb_array)
                else:
                    # Single scalar embedding
                    embeddings = [np.array([[embedding]], dtype=np.float32)]
                
                if not embeddings:
                    error_msg = "No embeddings provided in operation data"
                    logger.error(f"[2PC] {error_msg}")
                    return False, str(ValidationError(error_msg))
                
                logger.debug(f"[2PC] Processed embeddings - Count: {len(embeddings)}, "
                           f"First embedding type: {type(embeddings[0]).__name__ if embeddings else 'N/A'}, "
                           f"First embedding length: {len(embeddings[0]) if embeddings and hasattr(embeddings[0], '__len__') else 'N/A'}")
                
                # Create a list with the same prompt for each embedding
                documents = [prompt] * len(embeddings)
                
                try:
                    logger.debug("[2PC] Calling vector_store.add with vectors and documents")
                    await self.vector_store.add(
                        vectors=embeddings,
                        documents=documents,
                        metadatas=None,
                        ids=None
                    )
                    logger.debug("[2PC] vector_store.add completed successfully")
                    return True, None
                except Exception as e:
                    error_msg = f"Error in vector_store.add: {str(e)}"
                    logger.error(f"[2PC] {error_msg}", exc_info=True)
                    err = VectorOperationError(error_msg, original_exception=e)
                    return False, str(err)
                    
            elif operation.operation_type == OperationType.DELETE:
                # Get the ID to delete from the operation data
                delete_id = operation.data.get('id') or operation.data.get('prompt')
                if not delete_id:
                    error_msg = "No ID provided for delete operation"
                    logger.error(f"[2PC] {error_msg}")
                    return False, str(ValidationError(error_msg))
                
                try:
                    logger.debug(f"[2PC] Deleting vector with ID: {delete_id}")
                    # Delete the vector by ID
                    await self.vector_store.delete(ids=delete_id)
                    logger.debug("[2PC] vector_store.delete completed successfully")
                    return True, None
                except Exception as e:
                    error_msg = f"Error in vector_store.delete: {str(e)}"
                    logger.error(f"[2PC] {error_msg}", exc_info=True)
                    err = VectorOperationError(error_msg, original_exception=e)
                    return False, str(err)
            else:
                error_msg = f"Unsupported operation type: {operation.operation_type}"
                logger.error(f"[2PC] {error_msg}")
                return False, error_msg
                
        except Exception as e:
            error_msg = f"Unexpected error in VectorStoreParticipant.commit: {str(e)}"
            logger.error(f"[2PC] {error_msg}", exc_info=True)
            err = VectorOperationError(error_msg, original_exception=e)
            return False, str(err)
    
    async def rollback(self, operation: TwoPhaseOperation) -> Tuple[bool, Optional[str]]:
        """
        Rollback the vector store operation.
        
        Args:
            operation: The 2PC operation to rollback
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            # Try to remove the vector if it was added
            if operation.operation_type == OperationType.SET:
                await self.vector_store.delete(operation.data['prompt'])
            return True, None
        except Exception as e:
            err = VectorOperationError(f"Failed to rollback vector store: {e}", original_exception=e)
            return False, str(err)
