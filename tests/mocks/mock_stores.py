"""
Mock stores for testing purposes.
"""
from typing import Any, Dict, List, Optional, Union
import numpy as np

from semantrix.cache_store.base import BaseCacheStore, EvictionPolicy, NoOpEvictionPolicy
from semantrix.embedding.base import BaseEmbedder
from semantrix.vector_store.base import (
    BaseVectorStore, Vector, Metadata, MetadataFilter, QueryResult, VectorRecord,
    DistanceMetric, IndexType
)

# Mock Embedder

class MockEmbedder(BaseEmbedder):
    """Mock embedder for testing."""
    def __init__(self, dimension: int = 128):
        self._dimension = dimension

    async def encode(self, text: str) -> np.ndarray:
        """Encode text into a dummy vector."""
        return np.random.rand(self._dimension).astype(np.float32)

    def get_dimension(self) -> int:
        """Return the dimension of the embeddings."""
        return self._dimension

class MockFailingEmbedder(MockEmbedder):
    """Mock embedder that fails on encode."""
    async def encode(self, text: str) -> np.ndarray:
        """Simulate an encoding failure."""
        raise RuntimeError("Mocked embedding failure")

# Mock Cache Stores

class MockCacheStore(BaseCacheStore):
    """Mock cache store for testing."""
    def __init__(self):
        self._cache: Dict[str, str] = {}

    async def get_exact(self, prompt: str) -> Optional[str]:
        """Get a value from the cache."""
        return self._cache.get(prompt)

    async def get(self, prompt: str) -> Optional[str]:
        return self._cache.get(prompt)

    async def add(self, prompt: str, response: str) -> None:
        """Add a value to the cache."""
        self._cache[prompt] = response

    async def delete(self, key: str) -> bool:
        """Delete a key from the cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    async def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()

    async def enforce_limits(self, resource_limits: Any) -> None:
        """No-op for mock."""
        pass

    async def size(self) -> int:
        """Get the size of the cache."""
        return len(self._cache)

    def get_eviction_policy(self) -> EvictionPolicy:
        """Return a no-op eviction policy."""
        return NoOpEvictionPolicy()

class MockFailingCacheStore(MockCacheStore):
    """Mock cache store that can be configured to fail operations."""
    def __init__(self, fail_on_operation: Optional[str] = None):
        super().__init__()
        self.fail_on_operation = fail_on_operation

    def _should_fail(self, operation: str) -> bool:
        """Check if the current operation should fail."""
        return self.fail_on_operation == operation

    async def get_exact(self, prompt: str) -> Optional[str]:
        if self._should_fail('get_exact'):
            raise RuntimeError("Mocked failure in get_exact operation")
        return await super().get_exact(prompt)

    async def get(self, prompt: str) -> Optional[str]:
        if self.fail_on_operation == "get":
            raise RuntimeError("Simulated cache get failure")
        return self._cache.get(prompt)

    async def add(self, prompt: str, response: str) -> None:
        if self._should_fail('add'):
            raise RuntimeError("Mocked failure in add operation")
        await super().add(prompt, response)

# Mock Vector Stores (from existing files)

class MockVectorStore(BaseVectorStore):
    """
    Mock vector store for testing purposes.
    Implements required abstract methods from BaseVectorStore.
    """
    
    def __init__(
        self,
        dimension: int,
        metric: DistanceMetric = DistanceMetric.COSINE,
        namespace: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(dimension=dimension, metric=metric, namespace=namespace, **kwargs)
        self._vectors: Dict[str, VectorRecord] = {}
        self._next_id = 0
    
    async def add(
        self,
        vectors: Union[Vector, List[Vector]],
        documents: Optional[Union[str, List[str]]] = None,
        metadatas: Optional[Union[Metadata, List[Metadata]]] = None,
        ids: Optional[Union[str, List[str]]] = None,
        **kwargs: Any
    ) -> List[str]:
        """Add vectors to the store."""
        if isinstance(vectors, list):
            if documents is not None and not isinstance(documents, list):
                documents = [documents] * len(vectors)
            if metadatas is not None and not isinstance(metadatas, list):
                metadatas = [metadatas] * len(vectors)
            if ids is not None and not isinstance(ids, list):
                ids = [ids] * len(vectors)
                
            result = []
            for i, vector in enumerate(vectors):
                doc = documents[i] if documents and i < len(documents) else None
                meta = metadatas[i] if metadatas and i < len(metadatas) else None
                vec_id = ids[i] if ids and i < len(ids) else None
                result.append(await self._add_single(vector, doc, meta, vec_id))
            return result
        else:
            return [await self._add_single(vectors, documents, metadatas, ids)]
    
    async def _add_single(
        self,
        vector: Vector,
        document: Optional[str] = None,
        metadata: Optional[Metadata] = None,
        vec_id: Optional[str] = None
    ) -> str:
        """Add a single vector to the store."""
        if vec_id is None:
            vec_id = str(self._next_id)
            self._next_id += 1
        
        vector_array = np.array(vector, dtype=np.float32)
        if vector_array.ndim != 1 or vector_array.shape[0] != self.dimension:
            raise ValueError(f"Vector must be 1D with shape ({self.dimension},)")
        
        self._vectors[vec_id] = VectorRecord(
            id=vec_id,
            embedding=vector_array,
            document=document,
            metadata=metadata or {}
        )
        return vec_id
    
    async def search(
        self,
        query_vector: Vector,
        k: int = 4,
        filter: Optional[MetadataFilter] = None,
        include_vectors: bool = False,
        include_metadata: bool = True,
        **kwargs: Any
    ) -> List[QueryResult]:
        """Search for similar vectors."""
        results = []
        query = np.array(query_vector, dtype=np.float32)
        
        scored_vectors = []
        for vec_id, record in self._vectors.items():
            score = float(np.dot(query, record.embedding))
            scored_vectors.append((score, record))
        
        scored_vectors.sort(key=lambda x: x[0], reverse=True)
        
        for score, record in scored_vectors[:k]:
            results.append({
                'id': record.id,
                'document': record.document,
                'metadata': record.metadata if include_metadata else None,
                'score': score,
                'vector': record.embedding.tolist() if include_vectors else None
            })
        
        return results
    
    async def get(
        self,
        ids: Optional[Union[str, List[str]]] = None,
        filter: Optional[MetadataFilter] = None,
        include_vectors: bool = False,
        **kwargs: Any
    ) -> List[VectorRecord]:
        """Get vectors by ID or filter."""
        if ids is not None:
            if isinstance(ids, str):
                ids = [ids]
            return [
                VectorRecord(
                    id=vec_id,
                    embedding=record.embedding if include_vectors else None,
                    document=record.document,
                    metadata=record.metadata
                )
                for vec_id in ids 
                if (record := self._vectors.get(vec_id)) is not None
            ]
        return []
    
    async def delete(
        self,
        ids: Optional[Union[str, List[str]]] = None,
        filter: Optional[MetadataFilter] = None,
        **kwargs: Any
    ) -> bool:
        """Delete vectors by ID or filter."""
        if ids is not None:
            if isinstance(ids, str):
                ids = [ids]
            for vec_id in ids:
                self._vectors.pop(vec_id, None)
            return True
        return False
    
    async def count(self, filter: Optional[MetadataFilter] = None) -> int:
        """Count the number of vectors."""
        return len(self._vectors)
    
    async def create_index(
        self,
        index_type: IndexType = IndexType.FLAT,
        metric: Optional[DistanceMetric] = None,
        **kwargs: Any
    ) -> bool:
        """Create or update the vector index."""
        return True
    
    async def delete_collection(self, name: str, **kwargs: Any) -> bool:
        """Delete a collection/namespace."""
        self._vectors.clear()
        return True
    
    async def list_collections(self) -> List[str]:
        """List all collections/namespaces."""
        return ["default"]
    
    async def get_index_info(self) -> Dict[str, Any]:
        """Get information about the current index."""
        return {
            "type": "mock",
            "dimension": self.dimension,
            "metric": self.metric.value,
            "size": len(self._vectors)
        }
    
    async def update(
        self,
        ids: Union[str, List[str]],
        vectors: Optional[Union[Vector, List[Vector]]] = None,
        metadatas: Optional[Union[Metadata, List[Metadata]]] = None,
        documents: Optional[Union[str, List[str]]] = None,
        **kwargs: Any
    ) -> bool:
        """Update existing vectors."""
        if isinstance(ids, str):
            ids = [ids]
        
        for i, vec_id in enumerate(ids):
            if vec_id not in self._vectors:
                continue
                
            record = self._vectors[vec_id]
            
            if vectors is not None:
                vector = vectors[i] if isinstance(vectors, list) and i < len(vectors) else vectors
                if vector is not None:
                    record.embedding = np.array(vector, dtype=np.float32)
            
            if metadatas is not None:
                meta = metadatas[i] if isinstance(metadatas, list) and i < len(metadatas) else metadatas
                if meta is not None:
                    record.metadata = meta
            
            if documents is not None:
                doc = documents[i] if isinstance(documents, list) and i < len(documents) else documents
                if doc is not None:
                    record.document = doc
        
        return True

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
        super().__init__(dimension=dimension, metric=metric, namespace=namespace, **kwargs)
        self.fail_on_operation = fail_on_operation
        self.operation_count: Dict[str, int] = {
            'add': 0, 'get': 0, 'search': 0, 'delete': 0, 'update': 0,
            'create_index': 0, 'delete_collection': 0, 'list_collections': 0
        }
    
    def _should_fail(self, operation: str) -> bool:
        """Check if the current operation should fail."""
        self.operation_count[operation] = self.operation_count.get(operation, 0) + 1
        return self.fail_on_operation == operation
    
    async def add(self, *args, **kwargs) -> List[str]:
        if self._should_fail('add'):
            raise RuntimeError("Mocked failure in add operation")
        return await super().add(*args, **kwargs)
    
    async def search(self, *args, **kwargs) -> List[QueryResult]:
        if self._should_fail('search'):
            raise RuntimeError("Mocked failure in search operation")
        return await super().search(*args, **kwargs)
    
    async def get(self, *args, **kwargs) -> List[VectorRecord]:
        if self._should_fail('get'):
            raise RuntimeError("Mocked failure in get operation")
        return await super().get(*args, **kwargs)
    
    async def delete(self, *args, **kwargs) -> bool:
        if self._should_fail('delete'):
            raise RuntimeError("Mocked failure in delete operation")
        return await super().delete(*args, **kwargs)
    
    async def update(self, *args, **kwargs) -> bool:
        if self._should_fail('update'):
            raise RuntimeError("Mocked failure in update operation")
        return await super().update(*args, **kwargs)
    
    async def create_index(self, *args, **kwargs) -> bool:
        if self._should_fail('create_index'):
            raise RuntimeError("Mocked failure in create_index operation")
        return await super().create_index(*args, **kwargs)
    
    async def delete_collection(self, *args, **kwargs) -> bool:
        if self._should_fail('delete_collection'):
            raise RuntimeError("Mocked failure in delete_collection operation")
        return await super().delete_collection(*args, **kwargs)
    
    async def list_collections(self, *args, **kwargs) -> List[str]:
        if self._should_fail('list_collections'):
            raise RuntimeError("Mocked failure in list_collections operation")
        return await super().list_collections(*args, **kwargs)
    
    async def get_index_info(self, *args, **kwargs) -> Dict[str, Any]:
        if self._should_fail('get_index_info'):
            raise RuntimeError("Mocked failure in get_index_info operation")
        return await super().get_index_info(*args, **kwargs)
    
    async def count(self, *args, **kwargs) -> int:
        if self._should_fail('count'):
            raise RuntimeError("Mocked failure in count operation")
        return await super().count(*args, **kwargs)
