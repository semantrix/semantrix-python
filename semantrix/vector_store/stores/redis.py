"""Redis vector store implementation.

Note: This module requires the 'redis' and 'numpy' packages to be installed.
Install them with: pip install redis numpy
"""

import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence, Union, cast

# Optional import - will raise ImportError if not available
try:
    import redis
    from redis.commands.search.field import (
        TextField,
        VectorField,
        NumericField,
        TagField
    )
    from redis.commands.search.index_definition import IndexDefinition, IndexType
    from redis.commands.search.query import Query
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

import numpy as np

from ..base import (
    BaseVectorStore,
    DistanceMetric,
    Metadata,
    MetadataFilter,
    QueryResult,
    Vector,
    VectorRecord,
)

# Type alias for Redis client
RedisClientType = Any

class RedisVectorStore(BaseVectorStore):
    """Redis-based vector store implementation using Redis Search."""
    
    # Redis key prefixes
    VECTOR_PREFIX = "vec:"
    METADATA_PREFIX = "meta:"
    
    def __init__(
        self,
        dimension: int,
        metric: DistanceMetric = DistanceMetric.COSINE,
        namespace: Optional[str] = None,
        redis_url: Optional[str] = None,
        index_name: Optional[str] = None,
        redis_client: Optional[RedisClientType] = None,
        vector_field_name: str = "vector",
        **kwargs: Any
    ) -> None:
        """Initialize the Redis vector store.
        
        Args:
            dimension: The dimension of the vectors to be stored
            metric: Distance metric to use for similarity search
            namespace: Optional namespace for multi-tenancy
            redis_url: URL for connecting to Redis (e.g., 'redis://localhost:6379')
            index_name: Name of the Redis search index (default: auto-generated)
            redis_client: Optional pre-configured Redis client
            vector_field_name: Name of the vector field in Redis (default: 'vector')
            **kwargs: Additional arguments for Redis client
        """
        super().__init__(dimension=dimension, metric=metric, namespace=namespace)
        self._logger = logging.getLogger(__name__)
        
        if not REDIS_AVAILABLE:
            raise ImportError(
                "The 'redis' package is required for RedisVectorStore. "
                "Please install it with: pip install redis"
            )
        
        self.redis_url = redis_url or "redis://localhost:6379"
        self.index_name = index_name or f"semantrix:{namespace or 'default'}"
        self.vector_field_name = vector_field_name
        self._client = redis_client
        
        # Initialize Redis client if not provided
        if self._client is None:
            self._client = self._init_redis_client(**kwargs)
        
        # Create index if it doesn't exist
        self._ensure_index()
    
    def _init_redis_client(self, **kwargs: Any) -> RedisClientType:
        """Initialize and return a Redis client."""
        try:
            return redis.from_url(self.redis_url, **kwargs)
        except Exception as e:
            self._logger.error(f"Failed to connect to Redis: {str(e)}")
            raise
    
    def _ensure_index(self) -> None:
        """Create the Redis search index if it doesn't exist."""
        try:
            # Check if index exists
            try:
                self._client.ft(self.index_name).info()
                return  # Index exists, nothing to do
            except redis.exceptions.ResponseError as e:
                if "no such index" not in str(e).lower():
                    raise
            
            # Define schema for the index
            schema = (
                TextField("$", no_stem=True, sortable=True),  # JSON data
                VectorField(
                    self.vector_field_name,
                    self._get_redis_vector_config(),
                    self._get_redis_vector_type()
                ),
                TagField("namespace"),
                NumericField("timestamp")
            )
            
            # Create index definition
            definition = IndexDefinition(
                prefix=[f"{self.VECTOR_PREFIX}"],
                index_type=IndexType.JSON
            )
            
            # Create the index
            self._client.ft(self.index_name).create_index(
                fields=schema,
                definition=definition
            )
            
            self._logger.info(f"Created Redis search index: {self.index_name}")
            
        except Exception as e:
            self._logger.error(f"Failed to create Redis index: {str(e)}")
            raise
    
    def _get_redis_vector_config(self) -> Dict[str, Any]:
        """Get Redis vector configuration."""
        return {
            "TYPE": "FLOAT32",
            "DIM": self.dimension,
            "DISTANCE_METRIC": self._get_redis_metric(),
            "INITIAL_CAP": 1000,
            "BLOCK_SIZE": 1000
        }
    
    def _get_redis_metric(self) -> str:
        """Convert our DistanceMetric to Redis metric string."""
        metric_map = {
            DistanceMetric.COSINE: "COSINE",
            DistanceMetric.EUCLIDEAN: "L2",
            DistanceMetric.DOT_PRODUCT: "IP",
        }
        return metric_map.get(self.metric, "COSINE")
    
    def _get_redis_vector_type(self) -> str:
        """Get Redis vector type string."""
        return "HNSW"  # or "FLAT" for exact search
    
    def _get_key(self, id: str) -> str:
        """Get Redis key for a vector ID."""
        return f"{self.VECTOR_PREFIX}{id}"
    
    def _get_metadata_key(self, id: str) -> str:
        """Get Redis key for metadata."""
        return f"{self.METADATA_PREFIX}{id}"
    
    @classmethod
    def from_url(
        cls,
        url: str,
        dimension: int,
        **kwargs: Any
    ) -> 'RedisVectorStore':
        """Create a RedisVectorStore from a URL.
        
        Args:
            url: Redis server URL (e.g., 'redis://localhost:6379')
            dimension: The dimension of the vectors
            **kwargs: Additional arguments for RedisVectorStore
            
        Returns:
            Configured RedisVectorStore instance
            
        Example:
            ```python
            store = RedisVectorStore.from_url(
                url="redis://localhost:6379",
                dimension=768
            )
            ```
        """
        return cls(
            dimension=dimension,
            redis_url=url,
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
        
        # Prepare pipeline for batch operations
        pipe = self._client.pipeline()
        
        try:
            for i, (vec, vec_id, meta) in enumerate(zip(vectors, ids, metadatas)):
                # Convert vector to list if it's a numpy array
                vector_list = vec.tolist() if hasattr(vec, 'tolist') else list(vec)
                
                # Create document with vector and metadata
                doc = {
                    self.vector_field_name: np.array(vector_list, dtype=np.float32).tobytes(),
                    "namespace": self.namespace or "",
                    "timestamp": int(time.time() * 1000),  # Current timestamp in ms
                    **meta
                }
                
                # Store document in Redis
                pipe.json().set(self._get_key(vec_id), "$", doc)
            
            # Execute pipeline
            pipe.execute()
            return ids
            
        except Exception as e:
            self._logger.error(f"Failed to add vectors to Redis: {str(e)}")
            raise
    
    async def get(
        self,
        ids: Optional[Union[str, List[str]]] = None,
        filter: Optional[MetadataFilter] = None,
        include_vectors: bool = False,
        **kwargs: Any
    ) -> List[VectorRecord]:
        """Get vectors by IDs or filter."""
        results: List[VectorRecord] = []
        
        try:
            if ids is not None:
                if isinstance(ids, str):
                    ids = [ids]
                
                # Fetch by IDs
                for vec_id in ids:
                    doc = self._client.json().get(self._get_key(vec_id))
                    if doc:
                        results.append(self._doc_to_record(doc, include_vectors))
            
            elif filter is not None:
                # Build and execute query based on filter
                query_str, query_params = self._build_redis_query(filter)
                query = Query(query_str).with_payloads(True).dialect(2)
                
                # Execute search
                search_results = self._client.ft(self.index_name).search(query, query_params=query_params)
                
                # Convert results to VectorRecord objects
                for doc in search_results.docs:
                    results.append(self._doc_to_record(doc, include_vectors))
            
            return results
            
        except Exception as e:
            self._logger.error(f"Failed to get vectors from Redis: {str(e)}")
            raise
    
    def _doc_to_record(self, doc: Any, include_vectors: bool) -> VectorRecord:
        """Convert Redis document to VectorRecord."""
        if hasattr(doc, 'payload'):  # Search result
            doc_id = doc.id[len(self.VECTOR_PREFIX):]  # Remove prefix
            payload = json.loads(doc.payload)
            vector = np.frombuffer(doc.vector, dtype=np.float32).tolist() if include_vectors and hasattr(doc, 'vector') else None
            return VectorRecord(
                id=doc_id,
                embedding=vector,
                document=payload.get("document"),
                metadata={
                    k: v for k, v in payload.items()
                    if k not in ["document", "vector", "namespace", "timestamp"]
                },
                namespace=payload.get("namespace")
            )
        else:  # Direct JSON get
            doc_id = doc.get("id")
            vector = np.frombuffer(doc.get(self.vector_field_name, b''), dtype=np.float32).tolist() if include_vectors else None
            return VectorRecord(
                id=doc_id,
                embedding=vector,
                document=doc.get("document"),
                metadata={
                    k: v for k, v in doc.items()
                    if k not in ["document", self.vector_field_name, "namespace", "timestamp"]
                },
                namespace=doc.get("namespace")
            )
    
    def _build_redis_query(self, filter: MetadataFilter) -> tuple[str, dict]:
        """Build Redis query string and parameters from MetadataFilter."""
        conditions = []
        params = {}
        
        # Add namespace filter if specified
        if self.namespace:
            conditions.append(f"@namespace:{{{self.namespace}}}")
        
        # Add filter conditions
        for i, (key, value) in enumerate(filter.items()):
            param_name = f"val_{i}"
            if isinstance(value, (str, bool)):
                conditions.append(f"@{key}:{'%' + '{%s}' % param_name + '%'}")
                params[param_name] = str(value).lower() if isinstance(value, bool) else value
            elif isinstance(value, (int, float)):
                conditions.append(f"@{key}:[{value} {value}]")
            elif isinstance(value, dict):
                if "$eq" in value:
                    conditions.append(f"@{key}:{'%' + '{%s}' % param_name + '%'}")
                    params[param_name] = str(value["$eq"])
                elif "$in" in value:
                    in_conditions = []
                    for j, item in enumerate(value["$in"]):
                        item_param = f"{param_name}_{j}"
                        in_conditions.append(f"@{key}:{'%' + '{%s}' % item_param + '%'}")
                        params[item_param] = str(item)
                    conditions.append(f"({' | '.join(in_conditions)})")
                elif "$gt" in value:
                    conditions.append(f"@{key}:[({value['$gt']} +inf]")
                elif "$gte" in value:
                    conditions.append(f"@{key}:[{value['$gte']} +inf]")
                elif "$lt" in value:
                    conditions.append(f"@{key}:[-inf ({value['$lt']}]")
                elif "$lte" in value:
                    conditions.append(f"@{key}:[-inf {value['$lte']}]")
        
        query_str = " ".join(conditions) if conditions else "*"
        return query_str, params
    
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
        try:
            # Convert query vector to bytes
            query_vector_list = query_vector.tolist() if hasattr(query_vector, 'tolist') else list(query_vector)
            query_vector_bytes = np.array(query_vector_list, dtype=np.float32).tobytes()
            
            # Build base query
            query_parts = [f"*=>[KNN {k} @{self.vector_field_name} $vec AS score]"]
            query_params = {"vec": query_vector_bytes}
            
            # Add namespace filter if specified
            if self.namespace:
                query_parts.append(f"@namespace:{{{self.namespace}}}")
            
            # Add custom filter conditions
            if filter:
                filter_query, filter_params = self._build_redis_query(filter)
                if filter_query != "*":
                    query_parts.append(filter_query)
                query_params.update(filter_params)
            
            # Build final query
            query_str = " ".join(query_parts)
            query = (
                Query(query_str)
                .sort_by("score")
                .return_fields("__{self.vector_field_name}_score", "$")
                .dialect(2)
            )
            
            # Execute search
            search_results = self._client.ft(self.index_name).search(
                query,
                query_params=query_params
            )
            
            # Process results
            results: List[QueryResult] = []
            
            for doc in search_results.docs:
                payload = json.loads(doc.payload) if hasattr(doc, 'payload') else {}
                
                # Extract metadata if needed
                metadata = {
                    k: v for k, v in payload.items()
                    if k not in ["document", self.vector_field_name, "namespace", "timestamp"]
                } if include_metadata else None
                
                # Prepare result
                result: QueryResult = {
                    "id": doc.id[len(self.VECTOR_PREFIX):],  # Remove prefix
                    "document": payload.get("document") if include_metadata else None,
                    "metadata": metadata,
                    "score": 1.0 - float(doc.score),  # Convert distance to similarity
                    "vector": np.frombuffer(doc.vector, dtype=np.float32).tolist() if include_vectors and hasattr(doc, 'vector') else None
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
        # Convert inputs to lists
        if isinstance(ids, str):
            ids = [ids]
            
        if vectors is not None and not isinstance(vectors, list):
            vectors = [vectors]
            
        if documents is not None and not isinstance(documents, list):
            documents = [documents]
            
        if metadatas is not None and not isinstance(metadatas, list):
            metadatas = [metadatas]
        
        try:
            # Get existing documents
            pipe = self._client.pipeline()
            
            for i, vec_id in enumerate(ids):
                key = self._get_key(vec_id)
                
                # Get existing document
                doc = self._client.json().get(key) or {}
                
                # Update vector if provided
                if vectors is not None and i < len(vectors):
                    vector = vectors[i]
                    vector_list = vector.tolist() if hasattr(vector, 'tolist') else list(vector)
                    doc[self.vector_field_name] = np.array(vector_list, dtype=np.float32).tobytes()
                
                # Update document if provided
                if documents is not None and i < len(documents) and documents[i] is not None:
                    doc["document"] = documents[i]
                
                # Update metadata if provided
                if metadatas is not None and i < len(metadatas) and metadatas[i] is not None:
                    # Preserve existing metadata not being updated
                    for k, v in metadatas[i].items():
                        doc[k] = v
                
                # Update timestamp
                doc["timestamp"] = int(time.time() * 1000)
                
                # Save updated document
                pipe.json().set(key, "$", doc)
            
            # Execute updates
            pipe.execute()
            
        except Exception as e:
            self._logger.error(f"Failed to update vectors in Redis: {str(e)}")
            raise
    
    async def delete(
        self,
        ids: Optional[Union[str, List[str]]] = None,
        filter: Optional[MetadataFilter] = None,
        **kwargs: Any
    ) -> None:
        """Delete vectors by IDs or filter."""
        try:
            if ids is not None:
                if isinstance(ids, str):
                    ids = [ids]
                
                # Delete by IDs
                keys = [self._get_key(vec_id) for vec_id in ids]
                if keys:
                    self._client.delete(*keys)
            
            elif filter is not None:
                # Build and execute query to find matching documents
                query_str, query_params = self._build_redis_query(filter)
                query = Query(query_str).no_content().dialect(2)
                
                # Use scan to get all matching document IDs
                results = self._client.ft(self.index_name).search(query, query_params=query_params)
                
                # Delete matching documents
                if results.docs:
                    keys = [doc.id for doc in results.docs]
                    self._client.delete(*keys)
        
        except Exception as e:
            self._logger.error(f"Failed to delete vectors from Redis: {str(e)}")
            raise
    
    async def count(self, **kwargs: Any) -> int:
        """Get the number of vectors in the store."""
        try:
            # Use FT.INFO to get document count
            info = self._client.ft(self.index_name).info()
            return int(info.get("num_docs", 0))
        except Exception as e:
            self._logger.error(f"Failed to get vector count from Redis: {str(e)}")
            raise
    
    async def reset(self, **kwargs: Any) -> None:
        """Reset the vector store by deleting all vectors."""
        try:
            # Delete the index
            self._client.ft(self.index_name).dropindex(delete_documents=True)
            
            # Recreate the index
            self._ensure_index()
            
        except Exception as e:
            self._logger.error(f"Failed to reset Redis index: {str(e)}")
            raise
    
    async def close(self, **kwargs: Any) -> None:
        """Close the Redis client connection."""
        try:
            self._client.close()
        except Exception as e:
            self._logger.error(f"Error closing Redis client: {str(e)}")
            raise
