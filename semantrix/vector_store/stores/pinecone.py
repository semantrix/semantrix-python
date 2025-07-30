"""Pinecone vector store implementation.

Note: This module requires the 'pinecone-client' package to be installed.
Install it with: pip install pinecone-client
"""

import logging
from typing import Any, Dict, List, Optional, Union, cast

# Optional import - will raise ImportError if not available
try:
    import pinecone
    from pinecone import ServerlessSpec, PodSpec  # type: ignore
    PINE_AVAILABLE = True
except ImportError:
    PINE_AVAILABLE = False

from ..base import (
    BaseVectorStore,
    DistanceMetric,
    Metadata,
    MetadataFilter,
    QueryResult,
    Vector,
    VectorRecord,
)

# Type alias for Pinecone index
PineconeIndex = Any

class PineconeVectorStore(BaseVectorStore):
    """Pinecone-based vector store implementation."""
    
    def __init__(
        self,
        dimension: int,
        metric: DistanceMetric = DistanceMetric.COSINE,
        namespace: Optional[str] = None,
        environment: str = "asia-south1-gcp",  # Mumbai region
        index_name: Optional[str] = None,
        spec: Optional[Any] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the Pinecone vector store."""
        super().__init__(dimension=dimension, metric=metric, namespace=namespace)
        self._logger = logging.getLogger(__name__)
        
        if not PINE_AVAILABLE:
            raise ImportError(
                "The 'pinecone-client' package is required for PineconeVectorStore. "
                "Please install it with: pip install pinecone-client"
            )
        self.environment = environment
        self.index_name = index_name or f"semantrix-{namespace or 'default'}-{dimension}d"
        # Default to AWS Mumbai region for serverless
        self.spec = spec or ServerlessSpec(cloud="aws", region="ap-south-1")
        self._index: Optional[PineconeIndex] = None
        self._initialize_pinecone(**kwargs)
    
    def _initialize_pinecone(self, **kwargs: Any) -> None:
        """Initialize the Pinecone client and ensure the index exists."""
        try:
            # Initialize Pinecone client
            pinecone.init(
                api_key=kwargs.get("api_key"),
                environment=self.environment,
                **{k: v for k, v in kwargs.items() if k != "api_key"}
            )
            
            # Create index if it doesn't exist
            if self.index_name not in pinecone.list_indexes():
                self._logger.info(f"Creating Pinecone index: {self.index_name}")
                pinecone.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self._get_pinecone_metric(),
                    spec=self.spec
                )
            
            # Connect to the index
            self._index = pinecone.Index(self.index_name)
            
        except Exception as e:
            self._logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise
    
    def _get_pinecone_metric(self) -> str:
        """Convert our DistanceMetric to Pinecone's metric string."""
        metric_map = {
            DistanceMetric.COSINE: 'cosine',
            DistanceMetric.EUCLIDEAN: 'euclidean',
            DistanceMetric.DOT_PRODUCT: 'dotproduct',
        }
        return metric_map.get(self.metric, 'cosine')
    
    @classmethod
    def from_api_key(
        cls,
        api_key: str,
        dimension: int,
        environment: str = "asia-south1-gcp",  # Default to Mumbai region
        **kwargs: Any
    ) -> 'PineconeVectorStore':
        """Create a PineconeVectorStore using an API key.
        
        Args:
            api_key: Pinecone API key
            dimension: The dimension of the vectors
            environment: Pinecone environment (e.g., 'us-west1-gcp')
            **kwargs: Additional arguments for PineconeVectorStore
            
        Returns:
            Configured PineconeVectorStore instance
            
        Example:
            ```python
            store = PineconeVectorStore.from_api_key(
                api_key="your-api-key",
                dimension=768,
                environment="us-west1-gcp"
            )
            ```
        """
        return cls(
            dimension=dimension,
            environment=environment,
            api_key=api_key,
            **kwargs
        )
    
    async def add(
        self,
        vectors: Union[Vector, List[Vector]],
        documents: Optional[Union[str, List[str]]] = None,
        metadatas: Optional[Union[Metadata, List[Metadata]]] = None,
        ids: Optional[Union[str, List[str]]] = None,
        **kwargs: Any
    ) -> List[str]:
        """Add vectors to the store."""
        if self._index is None:
            raise RuntimeError("Pinecone index not initialized")
            
        # Convert inputs to lists for batch processing
        is_single = not isinstance(vectors, list)
        if is_single:
            vectors = [cast(Vector, vectors)]
            documents = [documents] if documents is not None else None
            metadatas = [metadatas] if metadatas is not None else None
            ids = [ids] if ids is not None else None
        
        # Generate IDs if not provided
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        
        # Prepare metadata
        if metadatas is None:
            metadatas = [{} for _ in range(len(vectors))]
        
        # Add namespace to metadata if specified
        if self.namespace:
            for meta in metadatas:
                if meta is not None:
                    meta["namespace"] = self.namespace
        
        # Add document text to metadata if provided
        if documents is not None:
            for i, doc in enumerate(documents):
                if doc is not None and metadatas[i] is not None:
                    metadatas[i]["document"] = doc
        
        # Prepare records for upsert
        records = []
        for i, (vec, vec_id, meta) in enumerate(zip(vectors, ids, metadatas)):
            records.append((
                str(vec_id),  # Pinecone requires string IDs
                vec.tolist() if hasattr(vec, 'tolist') else list(vec),
                meta or {}
            ))
        
        # Upsert to Pinecone
        try:
            self._index.upsert(vectors=records, namespace=self.namespace)
            return ids
        except Exception as e:
            self._logger.error(f"Failed to add vectors to Pinecone: {str(e)}")
            raise
    
    async def get(
        self,
        ids: Optional[Union[str, List[str]]] = None,
        filter: Optional[MetadataFilter] = None,
        include_vectors: bool = False,
        **kwargs: Any
    ) -> List[VectorRecord]:
        """Get vectors by IDs or filter."""
        if self._index is None:
            return []
            
        results: List[VectorRecord] = []
        
        if ids is not None:
            if isinstance(ids, str):
                ids = [ids]
            
            # Fetch by IDs
            try:
                response = self._index.fetch(
                    ids=ids,
                    namespace=self.namespace
                )
                
                for vec_id, vector_data in response.vectors.items():
                    metadata = vector_data.metadata or {}
                    results.append(VectorRecord(
                        id=vec_id,
                        embedding=vector_data.values if include_vectors else None,
                        document=metadata.pop("document", None),
                        metadata=metadata,
                        namespace=self.namespace
                    ))
                    
            except Exception as e:
                self._logger.error(f"Failed to fetch vectors from Pinecone: {str(e)}")
                
        elif filter is not None:
            # Fetch by metadata filter
            try:
                response = self._index.query(
                    vector=[0.0] * self.dimension,  # Dummy query vector
                    filter=filter,
                    top_k=10000,  # Maximum number of results
                    include_metadata=True,
                    include_values=include_vectors,
                    namespace=self.namespace
                )
                
                for match in response.matches:
                    metadata = match.metadata or {}
                    results.append(VectorRecord(
                        id=match.id,
                        embedding=match.values if include_vectors else None,
                        document=metadata.pop("document", None),
                        metadata=metadata,
                        namespace=self.namespace
                    ))
                    
            except Exception as e:
                self._logger.error(f"Failed to query Pinecone: {str(e)}")
        
        return results
    
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
        if self._index is None:
            return []
            
        try:
            query_vector_list = query_vector.tolist() if hasattr(query_vector, 'tolist') else list(query_vector)
            
            response = self._index.query(
                vector=query_vector_list,
                top_k=k,
                filter=filter,
                include_metadata=include_metadata,
                include_values=include_vectors,
                namespace=self.namespace
            )
            
            results: List[QueryResult] = []
            for match in response.matches:
                metadata = match.metadata if include_metadata and match.metadata else {}
                result: QueryResult = {
                    "id": match.id,
                    "document": metadata.pop("document", None) if include_metadata else None,
                    "metadata": metadata if include_metadata else None,
                    "score": float(match.score),
                    "vector": list(match.values) if include_vectors and match.values else None
                }
                results.append(result)
                
            return results
            
        except Exception as e:
            self._logger.error(f"Search failed: {str(e)}")
            raise
    
    async def update(
        self,
        ids: Union[str, List[str]],
        vectors: Optional[Union[Vector, List[Vector]]] = None,
        documents: Optional[Union[str, List[str], None]] = None,
        metadatas: Optional[Union[Metadata, List[Metadata], None]] = None,
        **kwargs: Any
    ) -> None:
        """Update vectors in the store."""
        if self._index is None:
            raise RuntimeError("Pinecone index not initialized")
            
        # Convert inputs to lists
        if isinstance(ids, str):
            ids = [ids]
            
        if vectors is not None and not isinstance(vectors, list):
            vectors = [vectors]
            
        if documents is not None and not isinstance(documents, list):
            documents = [documents]
            
        if metadatas is not None and not isinstance(metadatas, list):
            metadatas = [metadatas]
        
        # Get existing records
        existing = await self.get(ids=ids, include_vectors=True)
        existing_map = {r.id: r for r in existing}
        
        # Prepare updates
        records_to_update = []
        
        for i, vec_id in enumerate(ids):
            if vec_id not in existing_map:
                self._logger.warning(f"ID not found, skipping update: {vec_id}")
                continue
                
            record = existing_map[vec_id]
            
            # Update vector if provided
            new_vector = vectors[i] if vectors and i < len(vectors) else None
            
            # Update metadata
            metadata = dict(record.metadata or {})
            
            # Update document if provided
            if documents is not None and i < len(documents) and documents[i] is not None:
                metadata["document"] = documents[i]
                
            # Update metadata if provided
            if metadatas is not None and i < len(metadatas) and metadatas[i] is not None:
                metadata.update(metadatas[i] or {})
            
            # Add namespace to metadata
            if self.namespace:
                metadata["namespace"] = self.namespace
            
            # Prepare record for update
            records_to_update.append((
                str(vec_id),
                new_vector.tolist() if hasattr(new_vector, 'tolist') else new_vector or record.embedding,
                metadata
            ))
        
        # Perform updates
        if records_to_update:
            try:
                self._index.upsert(vectors=records_to_update, namespace=self.namespace)
            except Exception as e:
                self._logger.error(f"Failed to update vectors in Pinecone: {str(e)}")
                raise
    
    async def delete(
        self,
        ids: Optional[Union[str, List[str]]] = None,
        filter: Optional[MetadataFilter] = None,
        **kwargs: Any
    ) -> None:
        """Delete vectors by IDs or filter."""
        if self._index is None:
            return
            
        try:
            if ids is not None:
                if isinstance(ids, str):
                    ids = [ids]
                self._index.delete(ids=ids, namespace=self.namespace)
            elif filter is not None:
                # For filter-based deletes, we need to find matching IDs first
                results = await self.get(filter=filter, include_vectors=False)
                if results:
                    self._index.delete(ids=[r.id for r in results], namespace=self.namespace)
        except Exception as e:
            self._logger.error(f"Failed to delete vectors from Pinecone: {str(e)}")
            raise
    
    async def count(self, **kwargs: Any) -> int:
        """Get the number of vectors in the store."""
        if self._index is None:
            return 0
            
        try:
            stats = self._index.describe_index_stats()
            if self.namespace in stats.namespaces:
                return stats.namespaces[self.namespace].vector_count
            return 0
        except Exception as e:
            self._logger.error(f"Failed to get vector count from Pinecone: {str(e)}")
            raise
    
    async def reset(self, **kwargs: Any) -> None:
        """Reset the vector store by deleting all vectors."""
        if self._index is None:
            return
            
        try:
            # Delete all vectors in the namespace
            self._index.delete(delete_all=True, namespace=self.namespace)
        except Exception as e:
            self._logger.error(f"Failed to reset Pinecone index: {str(e)}")
            raise
    
    async def close(self, **kwargs: Any) -> None:
        """Close the Pinecone client and release resources."""
        # Pinecone client is stateless, nothing to close
        pass
