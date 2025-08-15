"""
Mock failing vector store implementation for testing error handling.
"""
from typing import Any, Dict, List, Optional, Union, cast
import numpy as np

from semantrix.vector_store.base import (
    BaseVectorStore, 
    Vector, 
    Metadata, 
    MetadataFilter, 
    QueryResult, 
    VectorRecord,
    DistanceMetric,
    IndexType
)
from .mock_vector_store import MockVectorStore

class MockFailingVectorStore(MockVectorStore):
    """
    Mock vector store that can be configured to fail operations for testing error handling.
    """
    
    def __init__(
        self,
        dimension: int,
        metric: DistanceMetric = DistanceMetric.COSINE,
        namespace: Optional[str] = None,
        fail_on_operation: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Initialize the mock failing vector store.
        
        Args:
            dimension: The dimension of the vectors
            metric: The distance metric to use
            namespace: Optional namespace for the store
            fail_on_operation: If set, the specified operation will always fail.
                              Can be one of: 'add', 'get', 'search', 'delete', 'update',
                              'create_index', 'delete_collection', 'list_collections'
            **kwargs: Additional arguments passed to the parent class
        """
        super().__init__(dimension=dimension, metric=metric, namespace=namespace, **kwargs)
        self.fail_on_operation = fail_on_operation
        self.operation_count: Dict[str, int] = {
            'add': 0,
            'get': 0,
            'search': 0,
            'delete': 0,
            'update': 0,
            'create_index': 0,
            'delete_collection': 0,
            'list_collections': 0
        }
    
    def _should_fail(self, operation: str) -> bool:
        """Check if the current operation should fail."""
        self.operation_count[operation] = self.operation_count.get(operation, 0) + 1
        return self.fail_on_operation == operation
    
    async def add(
        self,
        vectors: Union[Vector, List[Vector]],
        documents: Optional[Union[str, List[str]]] = None,
        metadatas: Optional[Union[Metadata, List[Metadata]]] = None,
        ids: Optional[Union[str, List[str]]] = None,
        **kwargs: Any
    ) -> List[str]:
        """Add vectors to the store with potential failure."""
        if self._should_fail('add'):
            raise RuntimeError("Mocked failure in add operation")
        return await super().add(vectors, documents, metadatas, ids, **kwargs)
    
    async def search(
        self,
        query_vector: Vector,
        k: int = 4,
        filter: Optional[MetadataFilter] = None,
        include_vectors: bool = False,
        include_metadata: bool = True,
        **kwargs: Any
    ) -> List[QueryResult]:
        """Search for similar vectors with potential failure."""
        if self._should_fail('search'):
            raise RuntimeError("Mocked failure in search operation")
        return await super().search(
            query_vector, k, filter, include_vectors, include_metadata, **kwargs
        )
    
    async def get(
        self,
        ids: Optional[Union[str, List[str]]] = None,
        filter: Optional[MetadataFilter] = None,
        include_vectors: bool = False,
        **kwargs: Any
    ) -> List[VectorRecord]:
        """Get vectors by ID or filter with potential failure."""
        if self._should_fail('get'):
            raise RuntimeError("Mocked failure in get operation")
        return await super().get(ids, filter, include_vectors, **kwargs)
    
    async def delete(
        self,
        ids: Optional[Union[str, List[str]]] = None,
        filter: Optional[MetadataFilter] = None,
        **kwargs: Any
    ) -> bool:
        """Delete vectors by ID or filter with potential failure."""
        if self._should_fail('delete'):
            raise RuntimeError("Mocked failure in delete operation")
        return await super().delete(ids, filter, **kwargs)
    
    async def update(
        self,
        ids: Union[str, List[str]],
        vectors: Optional[Union[Vector, List[Vector]]] = None,
        metadatas: Optional[Union[Metadata, List[Metadata]]] = None,
        documents: Optional[Union[str, List[str]]] = None,
        **kwargs: Any
    ) -> bool:
        """Update existing vectors with potential failure."""
        if self._should_fail('update'):
            raise RuntimeError("Mocked failure in update operation")
        return await super().update(ids, vectors, metadatas, documents, **kwargs)
    
    async def create_index(
        self,
        index_type: IndexType = IndexType.FLAT,
        metric: Optional[DistanceMetric] = None,
        **kwargs: Any
    ) -> bool:
        """Create or update the vector index with potential failure."""
        if self._should_fail('create_index'):
            raise RuntimeError("Mocked failure in create_index operation")
        return await super().create_index(index_type, metric, **kwargs)
    
    async def delete_collection(self, name: str, **kwargs: Any) -> bool:
        """Delete a collection/namespace with potential failure."""
        if self._should_fail('delete_collection'):
            raise RuntimeError("Mocked failure in delete_collection operation")
        return await super().delete_collection(name, **kwargs)
    
    async def list_collections(self) -> List[str]:
        """List all collections/namespaces with potential failure."""
        if self._should_fail('list_collections'):
            raise RuntimeError("Mocked failure in list_collections operation")
        return await super().list_collections()
    
    async def get_index_info(self) -> Dict[str, Any]:
        """Get information about the current index with potential failure."""
        if self._should_fail('get_index_info'):
            raise RuntimeError("Mocked failure in get_index_info operation")
        return await super().get_index_info()
    
    async def count(self, filter: Optional[MetadataFilter] = None) -> int:
        """Count the number of vectors with potential failure."""
        if self._should_fail('count'):
            raise RuntimeError("Mocked failure in count operation")
        return await super().count(filter)
