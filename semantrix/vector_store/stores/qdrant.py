"""Qdrant vector store implementation.

Note: This module requires the 'qdrant-client' package to be installed.
Install it with: pip install qdrant-client
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, Sequence, Union, cast

# Optional import - will raise ImportError if not available
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qdrant_models
    from qdrant_client.http.models import (
        Distance as QdrantDistance,
        VectorParams,
        PointStruct,
        Filter,
        FieldCondition,
        MatchValue,
        Range,
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

from ..base import (
    BaseVectorStore,
    DistanceMetric,
    Metadata,
    MetadataFilter,
    QueryResult,
    Vector,
    VectorRecord,
)

# Type alias for Qdrant client
QdrantClientType = Any

class QdrantVectorStore(BaseVectorStore):
    """Qdrant-based vector store implementation."""
    
    def __init__(
        self,
        dimension: int,
        metric: DistanceMetric = DistanceMetric.COSINE,
        namespace: Optional[str] = None,
        collection_name: Optional[str] = None,
        location: Optional[str] = None,
        url: Optional[str] = None,
        port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        https: bool = False,
        api_key: Optional[str] = None,
        prefix: Optional[str] = None,
        timeout: Optional[float] = None,
        host: Optional[str] = None,
        path: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """Initialize the Qdrant vector store.
        
        Args:
            dimension: The dimension of the vectors to be stored
            metric: Distance metric to use for similarity search
            namespace: Optional namespace for multi-tenancy
            collection_name: Name of the Qdrant collection (default: auto-generated)
            location: If 'path' is not specified, this indicates the URL of Qdrant service.
            url: Either host or str of 'Optional[scheme], host, Optional[port], Optional[prefix]'.
            port: Port of the REST API interface. Default: 6333
            grpc_port: Port of the gRPC interface. Default: 6334
            prefer_grpc: If true - use gRPC interface whenever possible in custom methods.
            https: If true - use HTTPS(SSL) protocol. Default: False
            api_key: API key for authentication in Qdrant Cloud.
            prefix: Prefix to the REST URL path. Example: 'service/v1' will result in
                   'http://localhost:6333/service/v1/{qdrant_endpoint}' for the REST API.
            timeout: Timeout for REST and gRPC API requests.
            host: Host name of Qdrant service. If url and host are None, set to 'localhost'.
            path: Path in which the vectors will be stored when running locally.
            **kwargs: Additional arguments for QdrantClient
        """
        super().__init__(dimension=dimension, metric=metric, namespace=namespace)
        self._logger = logging.getLogger(__name__)
        
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "The 'qdrant-client' package is required for QdrantVectorStore. "
                "Please install it with: pip install qdrant-client"
            )
        
        self.collection_name = collection_name or f"semantrix-{namespace or 'default'}-{dimension}d"
        self._client: Optional[QdrantClientType] = None
        
        # Store connection parameters
        self._connection_params = {
            'location': location,
            'url': url,
            'port': port,
            'grpc_port': grpc_port,
            'prefer_grpc': prefer_grpc,
            'https': https,
            'api_key': api_key,
            'prefix': prefix,
            'timeout': timeout,
            'host': host,
            'path': path,
            **kwargs
        }
        
        # Initialize the client
        self._init_qdrant_client()
    
    def _init_qdrant_client(self) -> None:
        """Initialize the Qdrant client and ensure the collection exists."""
        try:
            # Initialize Qdrant client
            self._client = QdrantClient(**self._connection_params)
            
            # Check if collection exists, create if not
            collections = self._client.get_collections()
            collection_names = [collection.name for collection in collections.collections]
            
            if self.collection_name not in collection_names:
                self._logger.info(f"Creating Qdrant collection: {self.collection_name}")
                self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dimension,
                        distance=self._get_qdrant_metric()
                    )
                )
                
        except Exception as e:
            self._logger.error(f"Failed to initialize Qdrant: {str(e)}")
            raise
    
    def _get_qdrant_metric(self) -> QdrantDistance:
        """Convert our DistanceMetric to Qdrant's Distance enum."""
        metric_map = {
            DistanceMetric.COSINE: QdrantDistance.COSINE,
            DistanceMetric.EUCLIDEAN: QdrantDistance.EUCLID,
            DistanceMetric.DOT_PRODUCT: QdrantDistance.DOT,
        }
        return metric_map.get(self.metric, QdrantDistance.COSINE)
    
    @classmethod
    def from_url(
        cls,
        url: str,
        dimension: int,
        api_key: Optional[str] = None,
        **kwargs: Any
    ) -> 'QdrantVectorStore':
        """Create a QdrantVectorStore from a URL.
        
        Args:
            url: Qdrant server URL (e.g., 'http://localhost:6333')
            dimension: The dimension of the vectors
            api_key: Optional API key for authentication
            **kwargs: Additional arguments for QdrantVectorStore
            
        Returns:
            Configured QdrantVectorStore instance
            
        Example:
            ```python
            store = QdrantVectorStore.from_url(
                url="http://localhost:6333",
                dimension=768,
                api_key="your-api-key"
            )
            ```
        """
        return cls(
            dimension=dimension,
            url=url,
            api_key=api_key,
            **kwargs
        )
    
    @classmethod
    def from_local(
        cls,
        path: str,
        dimension: int,
        **kwargs: Any
    ) -> 'QdrantVectorStore':
        """Create a QdrantVectorStore with local storage.
        
        Args:
            path: Path to store the Qdrant data
            dimension: The dimension of the vectors
            **kwargs: Additional arguments for QdrantVectorStore
            
        Returns:
            Configured QdrantVectorStore instance
            
        Example:
            ```python
            store = QdrantVectorStore.from_local(
                path="./qdrant_data",
                dimension=768
            )
            ```
        """
        return cls(
            dimension=dimension,
            location=path,
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
        if self._client is None:
            raise RuntimeError("Qdrant client not initialized")
            
        # Convert inputs to lists for batch processing
        is_single = not isinstance(vectors, list)
        if is_single:
            vectors = [cast(Vector, vectors)]
            documents = [documents] if documents is not None else None
            metadatas = [metadatas] if metadatas is not None else None
            ids = [ids] if ids is not None else None
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        
        # Prepare metadata
        if metadatas is None:
            metadatas = [{} for _ in range(len(vectors))]
        
        # Add document text to metadata if provided
        if documents is not None:
            for i, doc in enumerate(documents):
                if doc is not None and metadatas[i] is not None:
                    metadatas[i]["document"] = doc
        
        # Add namespace to metadata if specified
        if self.namespace:
            for meta in metadatas:
                if meta is not None:
                    meta["namespace"] = self.namespace
        
        # Prepare points for upload
        points = []
        for i, (vec, vec_id, meta) in enumerate(zip(vectors, ids, metadatas)):
            point = PointStruct(
                id=vec_id,
                vector=vec.tolist() if hasattr(vec, 'tolist') else list(vec),
                payload=meta or {}
            )
            points.append(point)
        
        # Upload to Qdrant
        try:
            self._client.upsert(
                collection_name=self.collection_name,
                points=points,
                **kwargs
            )
            return ids
        except Exception as e:
            self._logger.error(f"Failed to add vectors to Qdrant: {str(e)}")
            raise
    
    async def get(
        self,
        ids: Optional[Union[str, List[str]]] = None,
        filter: Optional[MetadataFilter] = None,
        include_vectors: bool = False,
        **kwargs: Any
    ) -> List[VectorRecord]:
        """Get vectors by IDs or filter."""
        if self._client is None:
            return []
            
        results: List[VectorRecord] = []
        
        try:
            if ids is not None:
                if isinstance(ids, str):
                    ids = [ids]
                
                # Fetch by IDs
                records = self._client.retrieve(
                    collection_name=self.collection_name,
                    ids=ids,
                    with_vectors=include_vectors,
                    **kwargs
                )
                
                for record in records:
                    if record.payload is None:
                        continue
                        
                    results.append(VectorRecord(
                        id=str(record.id),
                        embedding=record.vector if include_vectors else None,
                        document=record.payload.get("document"),
                        metadata={
                            k: v for k, v in record.payload.items() 
                            if k != "document"
                        },
                        namespace=self.namespace
                    ))
                    
            elif filter is not None:
                # Build Qdrant filter from metadata filter
                qdrant_filter = self._build_qdrant_filter(filter)
                
                # Fetch by filter
                records, _ = self._client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=qdrant_filter,
                    with_vectors=include_vectors,
                    limit=10000,  # Maximum number of results
                    **kwargs
                )
                
                for record in records:
                    if record.payload is None:
                        continue
                        
                    results.append(VectorRecord(
                        id=str(record.id),
                        embedding=record.vector if include_vectors else None,
                        document=record.payload.get("document"),
                        metadata={
                            k: v for k, v in record.payload.items() 
                            if k != "document"
                        },
                        namespace=self.namespace
                    ))
                    
        except Exception as e:
            self._logger.error(f"Failed to get vectors from Qdrant: {str(e)}")
            raise
            
        return results
    
    def _build_qdrant_filter(self, filter: MetadataFilter) -> Optional[Filter]:
        """Convert our MetadataFilter to Qdrant's Filter."""
        if not filter:
            return None
            
        conditions = []
        
        for key, value in filter.items():
            if isinstance(value, (str, int, float, bool)):
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        match=MatchValue(value=value)
                    )
                )
            elif isinstance(value, dict) and "$eq" in value:
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        match=MatchValue(value=value["$eq"])
                    )
                )
            elif isinstance(value, dict) and "$in" in value:
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        match=MatchValue(any_of=value["$in"])
                    )
                )
            elif isinstance(value, dict) and "$gt" in value:
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        range=Range(gt=value["$gt"])
                    )
                )
            elif isinstance(value, dict) and "$gte" in value:
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        range=Range(gte=value["$gte"])
                    )
                )
            elif isinstance(value, dict) and "$lt" in value:
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        range=Range(lt=value["$lt"])
                    )
                )
            elif isinstance(value, dict) and "$lte" in value:
                conditions.append(
                    FieldCondition(
                        key=f"metadata.{key}",
                        range=Range(lte=value["$lte"])
                    )
                )
        
        if not conditions:
            return None
            
        return Filter(must=conditions)
    
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
        if self._client is None:
            return []
            
        try:
            query_vector_list = query_vector.tolist() if hasattr(query_vector, 'tolist') else list(query_vector)
            
            # Build Qdrant filter if provided
            qdrant_filter = self._build_qdrant_filter(filter) if filter else None
            
            # Perform search
            search_results = self._client.search(
                collection_name=self.collection_name,
                query_vector=query_vector_list,
                query_filter=qdrant_filter,
                limit=k,
                with_vectors=include_vectors,
                with_payload=include_metadata,
                **kwargs
            )
            
            # Prepare results
            results: List[QueryResult] = []
            
            for hit in search_results:
                payload = hit.payload or {}
                metadata = {
                    k: v for k, v in payload.items()
                    if k != "document"
                } if include_metadata else None
                
                result: QueryResult = {
                    "id": str(hit.id),
                    "document": payload.get("document") if include_metadata else None,
                    "metadata": metadata,
                    "score": float(hit.score),
                    "vector": list(hit.vector) if include_vectors and hit.vector else None
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
        if self._client is None:
            raise RuntimeError("Qdrant client not initialized")
            
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
        points = []
        
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
            
            # Prepare point for update
            point = PointStruct(
                id=vec_id,
                vector=new_vector.tolist() if hasattr(new_vector, 'tolist') else new_vector or record.embedding,
                payload=metadata
            )
            points.append(point)
        
        # Perform updates
        if points:
            try:
                self._client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    **kwargs
                )
            except Exception as e:
                self._logger.error(f"Failed to update vectors in Qdrant: {str(e)}")
                raise
    
    async def delete(
        self,
        ids: Optional[Union[str, List[str]]] = None,
        filter: Optional[MetadataFilter] = None,
        **kwargs: Any
    ) -> None:
        """Delete vectors by IDs or filter."""
        if self._client is None:
            return
            
        try:
            if ids is not None:
                if isinstance(ids, str):
                    ids = [ids]
                self._client.delete(
                    collection_name=self.collection_name,
                    points_selector=ids,
                    **kwargs
                )
            elif filter is not None:
                # For filter-based deletes, we need to find matching IDs first
                results = await self.get(filter=filter, include_vectors=False)
                if results:
                    self._client.delete(
                        collection_name=self.collection_name,
                        points_selector=[r.id for r in results],
                        **kwargs
                    )
        except Exception as e:
            self._logger.error(f"Failed to delete vectors from Qdrant: {str(e)}")
            raise
    
    async def count(self, **kwargs: Any) -> int:
        """Get the number of vectors in the store."""
        if self._client is None:
            return 0
            
        try:
            stats = self._client.get_collection(collection_name=self.collection_name)
            return stats.vectors_count
        except Exception as e:
            self._logger.error(f"Failed to get vector count from Qdrant: {str(e)}")
            raise
    
    async def reset(self, **kwargs: Any) -> None:
        """Reset the vector store by deleting all vectors."""
        if self._client is None:
            return
            
        try:
            # Delete the collection and recreate it
            self._client.delete_collection(collection_name=self.collection_name)
            self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=self._get_qdrant_metric()
                )
            )
        except Exception as e:
            self._logger.error(f"Failed to reset Qdrant collection: {str(e)}")
            raise
    
    async def close(self, **kwargs: Any) -> None:
        """Close the Qdrant client and release resources."""
        # Qdrant client is stateless, nothing to close
        pass
