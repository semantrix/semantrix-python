from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any, Dict, Generic, List, Literal, Optional, Tuple, Type, TypeVar, Union, 
    TypedDict, Callable, AsyncIterator, Sequence, overload
)
from uuid import UUID, uuid4
import asyncio
import json
import logging

import numpy as np
import numpy.typing as npt

from semantrix.utils.datetime_utils import utc_now

# Type aliases
Vector = Union[npt.NDArray[np.float32], List[float]]
Metadata = Dict[str, Union[str, int, float, bool, List[Union[str, int, float, bool]]]]
MetadataFilter = Dict[str, Any]  # Simplified filter for now, can be enhanced

class DistanceMetric(str, Enum):
    """Supported distance metrics for vector similarity."""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"
    MANHATTAN = "manhattan"

class IndexType(str, Enum):
    """Supported index types for vector stores."""
    FLAT = "flat"
    IVF = "ivf"
    HNSW = "hnsw"
    LSH = "lsh"
    SCANN = "scann"

class VectorStoreStatus(str, Enum):
    """Status of the vector store."""
    READY = "ready"
    INITIALIZING = "initializing"
    ERROR = "error"
    MAINTENANCE = "maintenance"

@dataclass
class VectorRecord:
    """A record containing a vector and its associated data."""
    id: str
    embedding: npt.NDArray[np.float32]
    metadata: Optional[Metadata] = None
    document: Optional[str] = None  # Original document/text
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    namespace: Optional[str] = None
    score: Optional[float] = None  # For search results

    def to_dict(self) -> Dict[str, Any]:
        """Convert the record to a dictionary."""
        return {
            'id': self.id,
            'embedding': self.embedding.tolist(),
            'metadata': self.metadata,
            'document': self.document,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'namespace': self.namespace,
            'score': self.score
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VectorRecord':
        """Create a record from a dictionary."""
        return cls(
            id=data['id'],
            embedding=np.array(data['embedding'], dtype=np.float32),
            metadata=data.get('metadata'),
            document=data.get('document'),
            created_at=datetime.fromisoformat(data['created_at']),
            updated_at=datetime.fromisoformat(data['updated_at']),
            namespace=data.get('namespace'),
            score=data.get('score')
        )

class QueryResult(TypedDict):
    """Result of a vector search query."""
    id: str
    document: Optional[str]
    metadata: Optional[Metadata]
    score: float
    vector: Optional[Vector]

class BaseVectorStore(ABC):
    """
    Abstract base class for vector storage and similarity search with async support.
    
    This class defines the interface that all vector store implementations must follow.
    It provides async methods for adding, searching, and managing vector embeddings.
    
    Features:
    - Metadata support for all operations
    - Advanced filtering and querying
    - Batch operations
    - Index management
    - Connection pooling and retries
    - Namespace support for multi-tenancy
    - Hybrid search (vector + keyword)
    - Custom distance metrics
    """
    
    def __init__(
        self,
        dimension: int,
        metric: DistanceMetric = DistanceMetric.COSINE,
        namespace: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the vector store.
        
        Args:
            dimension: Dimension of the vectors
            metric: Distance metric to use for similarity search
            namespace: Optional namespace for multi-tenancy
            **kwargs: Additional implementation-specific parameters
        """
        self.dimension = dimension
        self.metric = metric
        self.namespace = namespace
        self._executor = ThreadPoolExecutor()
        self._logger = logging.getLogger(self.__class__.__name__)
    
    # Core Operations
    
    @abstractmethod
    async def add(
        self,
        vectors: Union[Vector, List[Vector]],
        documents: Optional[Union[str, List[str]]] = None,
        metadatas: Optional[Union[Metadata, List[Metadata]]] = None,
        ids: Optional[Union[str, List[str]]] = None,
        **kwargs: Any
    ) -> List[str]:
        """
        Add vectors to the store.
        
        Args:
            vectors: Single vector or list of vectors to add
            documents: Optional document text(s) associated with the vector(s)
            metadatas: Optional metadata dict(s) to associate with the vector(s)
            ids: Optional ID(s) for the vector(s). If not provided, will be generated.
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            List of IDs for the added vectors
            
        Raises:
            ValueError: If inputs are invalid or dimensions don't match
        """
        pass
    
    @abstractmethod
    async def search(
        self,
        query_vector: Vector,
        k: int = 4,
        filter: Optional[MetadataFilter] = None,
        include_vectors: bool = False,
        include_metadata: bool = True,
        **kwargs: Any
    ) -> List[QueryResult]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: The query vector
            k: Number of results to return
            filter: Optional metadata filter to apply
            include_vectors: Whether to include vectors in the results
            include_metadata: Whether to include metadata in the results
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            List of query results, sorted by relevance
        """
        pass
    
    @abstractmethod
    async def get(
        self,
        ids: Optional[Union[str, List[str]]] = None,
        filter: Optional[MetadataFilter] = None,
        include_vectors: bool = False,
        **kwargs: Any
    ) -> List[VectorRecord]:
        """
        Get vectors by ID or filter.
        
        Args:
            ids: Single ID or list of IDs to retrieve
            filter: Optional metadata filter
            include_vectors: Whether to include vectors in the results
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            List of vector records matching the criteria
        """
        pass
    
    @abstractmethod
    async def delete(
        self,
        ids: Optional[Union[str, List[str]]] = None,
        filter: Optional[MetadataFilter] = None,
        **kwargs: Any
    ) -> bool:
        """
        Delete vectors by ID or filter.
        
        Args:
            ids: Single ID or list of IDs to delete
            filter: Optional metadata filter for bulk deletion
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            True if deletion was successful
            
        Note:
            At least one of ids or filter must be provided
        """
        pass
    
    @abstractmethod
    async def update(
        self,
        ids: Union[str, List[str]],
        vectors: Optional[Union[Vector, List[Vector]]] = None,
        metadatas: Optional[Union[Metadata, List[Metadata]]] = None,
        documents: Optional[Union[str, List[str]]] = None,
        **kwargs: Any
    ) -> bool:
        """
        Update existing vectors.
        
        Args:
            ids: Single ID or list of IDs to update
            vectors: New vectors (must match count of ids if provided)
            metadatas: New metadata (will be merged with existing)
            documents: New document text
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            True if update was successful
        """
        pass
    
    # Batch Operations
    
    async def batch_add(
        self,
        vectors: List[Vector],
        documents: Optional[List[Optional[str]]] = None,
        metadatas: Optional[List[Optional[Metadata]]] = None,
        ids: Optional[List[str]] = None,
        batch_size: int = 100,
        **kwargs: Any
    ) -> List[str]:
        """
        Add vectors in batches.
        
        Args:
            vectors: List of vectors to add
            documents: Optional list of document texts
            metadatas: Optional list of metadata dicts
            ids: Optional list of IDs
            batch_size: Number of vectors to add in each batch
            **kwargs: Additional parameters to pass to add()
            
        Returns:
            List of IDs for the added vectors
        """
        if ids is None:
            ids = [str(uuid4()) for _ in range(len(vectors))]
            
        if documents is None:
            documents = [None] * len(vectors)
            
        if metadatas is None:
            metadatas = [None] * len(vectors)
            
        all_ids = []
        for i in range(0, len(vectors), batch_size):
            batch_vectors = vectors[i:i+batch_size]
            batch_docs = documents[i:i+batch_size]
            batch_metas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            batch_ids = await self.add(
                vectors=batch_vectors,
                documents=batch_docs,
                metadatas=batch_metas,
                ids=batch_ids,
                **kwargs
            )
            all_ids.extend(batch_ids)
            
        return all_ids
    
    # Index Management
    
    @abstractmethod
    async def create_index(
        self,
        index_type: IndexType = IndexType.FLAT,
        metric: Optional[DistanceMetric] = None,
        **kwargs: Any
    ) -> bool:
        """
        Create or update the vector index.
        
        Args:
            index_type: Type of index to create
            metric: Distance metric to use (overrides default if provided)
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            True if index creation was successful
        """
        pass
    
    @abstractmethod
    async def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about the current index.
        
        Returns:
            Dictionary containing index information
        """
        pass
    
    # Collection/Namespace Management
    
    @abstractmethod
    async def list_collections(self) -> List[str]:
        """
        List all collections/namespaces in the store.
        
        Returns:
            List of collection/namespace names
        """
        pass
    
    @abstractmethod
    async def delete_collection(self, name: str, **kwargs: Any) -> bool:
        """
        Delete a collection/namespace.
        
        Args:
            name: Name of the collection to delete
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            True if deletion was successful
        """
        pass
    
    # Utility Methods
    
    @abstractmethod
    async def count(self, filter: Optional[MetadataFilter] = None) -> int:
        """
        Count the number of vectors matching the filter.
        
        Args:
            filter: Optional metadata filter
            
        Returns:
            Number of vectors matching the filter
        """
        pass
    
    async def clear(self) -> bool:
        """
        Clear all vectors in the current namespace.
        
        Returns:
            True if operation was successful
        """
        return await self.delete(filter={})
    
    async def ping(self) -> bool:
        """
        Check if the vector store is available.
        
        Returns:
            True if the store is available
        """
        try:
            await self.count()
            return True
        except Exception as e:
            self._logger.warning(f"Ping failed: {str(e)}")
            return False
    
    # Context Manager
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup resources."""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=True)
    
    # Hybrid Search (Optional - can be overridden by implementations)
    
    async def hybrid_search(
        self,
        query: str,
        query_vector: Optional[Vector] = None,
        k: int = 4,
        filter: Optional[MetadataFilter] = None,
        include_vectors: bool = False,
        include_metadata: bool = True,
        **kwargs: Any
    ) -> List[QueryResult]:
        """
        Perform a hybrid search using both vector similarity and keyword matching.
        
        Args:
            query: The search query string
            query_vector: Optional pre-computed query vector
            k: Number of results to return
            filter: Optional metadata filter
            include_vectors: Whether to include vectors in the results
            include_metadata: Whether to include metadata in the results
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            List of query results, sorted by relevance
        """
        # Default implementation falls back to vector search
        if query_vector is None:
            raise ValueError("query_vector is required for default hybrid search implementation")
        return await self.search(
            query_vector=query_vector,
            k=k,
            filter=filter,
            include_vectors=include_vectors,
            include_metadata=include_metadata,
            **kwargs
        )
