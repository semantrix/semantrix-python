"""
Mock vector store implementation for testing purposes.
"""
from typing import Any, Dict, List, Optional, Union
import numpy as np
import numpy.typing as npt

from semantrix.vector_store.base import BaseVectorStore, Vector, Metadata, MetadataFilter, QueryResult, VectorRecord
from semantrix.vector_store.base import DistanceMetric, IndexType

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
        # Simple implementation that returns the first k vectors
        results = []
        query = np.array(query_vector, dtype=np.float32)
        
        # Calculate scores (simple dot product for now)
        scored_vectors = []
        for vec_id, record in self._vectors.items():
            score = float(np.dot(query, record.embedding))
            scored_vectors.append((score, record))
        
        # Sort by score (descending)
        scored_vectors.sort(key=lambda x: x[0], reverse=True)
        
        # Return top k results
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
        return []  # Filter not implemented in this mock
    
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
        return True  # No-op for mock
    
    async def delete_collection(self, name: str, **kwargs: Any) -> bool:
        """Delete a collection/namespace."""
        self._vectors.clear()
        return True
    
    async def list_collections(self) -> List[str]:
        """List all collections/namespaces."""
        return ["default"]  # Mock implementation
    
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
