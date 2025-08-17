"""Milvus vector store implementation.

Note: This module requires the 'pymilvus' package to be installed.
Install it with: pip install pymilvus
"""

import json
import uuid
from typing import Any, Dict, List, Optional, Sequence, Union, cast

# Optional import - will raise ImportError if not available
try:
    from pymilvus import (
        connections,
        utility,
        Collection,
        CollectionSchema,
        FieldSchema,
        DataType,
        MilvusException,
    )
    from pymilvus.client.types import IndexType, MetricType
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False

import numpy as np

from semantrix.exceptions import VectorOperationError
from semantrix.utils.logging import get_logger

from ..base import (
    BaseVectorStore,
    DistanceMetric,
    Metadata,
    MetadataFilter,
    QueryResult,
    Vector,
    VectorRecord,
)

# Type alias for Milvus collection
MilvusCollection = Any

class MilvusVectorStore(BaseVectorStore):
    """Milvus vector store implementation."""
    
    # Default collection name if not provided
    DEFAULT_COLLECTION_NAME = "semantrix_vectors"
    
    # Field names in the collection
    ID_FIELD = "id"
    VECTOR_FIELD = "vector"
    DOCUMENT_FIELD = "document"
    METADATA_FIELD = "metadata"
    NAMESPACE_FIELD = "namespace"
    
    def __init__(
        self,
        dimension: int,
        metric: DistanceMetric = DistanceMetric.COSINE,
        namespace: Optional[str] = None,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        uri: str = "http://localhost:19530",  # Default Milvus standalone
        token: Optional[str] = None,  # For Milvus Cloud
        **kwargs: Any
    ) -> None:
        """Initialize the Milvus vector store.
        
        Args:
            dimension: The dimension of the vectors to be stored
            metric: Distance metric to use for similarity search
            namespace: Optional namespace for multi-tenancy
            collection_name: Name of the collection to store vectors
            uri: Milvus server URI (default: "http://localhost:19530")
            token: Authentication token for Milvus Cloud
            **kwargs: Additional arguments for pymilvus.connections.connect()
        """
        super().__init__(dimension=dimension, metric=metric, namespace=namespace)
        self._logger = get_logger(__name__)
        
        if not MILVUS_AVAILABLE:
            raise ImportError(
                "The 'pymilvus' package is required for MilvusVectorStore. "
                "Please install it with: pip install pymilvus"
            )
        
        self.collection_name = collection_name
        self.uri = uri
        self.token = token
        self._connection_args = kwargs
        self._collection: Optional[MilvusCollection] = None
        
        # Map our distance metric to Milvus metric type
        self._metric_type = self._get_metric_type(metric)
        
        # Initialize connection and ensure collection exists
        self._init_connection()
        self._ensure_collection()
    
    def _get_metric_type(self, metric: DistanceMetric) -> str:
        """Convert our DistanceMetric to Milvus MetricType."""
        if metric == DistanceMetric.COSINE:
            return MetricType.COSINE
        elif metric == DistanceMetric.EUCLIDEAN:
            return MetricType.L2
        elif metric == DistanceMetric.DOT_PRODUCT:
            return MetricType.IP
        else:
            raise VectorOperationError(f"Unsupported distance metric: {metric}")
    
    def _init_connection(self) -> None:
        """Initialize the Milvus connection."""
        try:
            connections.connect(
                alias="default",
                uri=self.uri,
                token=self.token,
                **self._connection_args
            )
            self._logger.info(f"Connected to Milvus server at {self.uri}")
        except MilvusException as e:
            self._logger.error(f"Failed to connect to Milvus: {str(e)}")
            raise VectorOperationError("Failed to connect to Milvus") from e
    
    def _ensure_collection(self) -> None:
        """Ensure the collection exists and has the correct schema."""
        try:
            # Check if collection exists
            if utility.has_collection(self.collection_name):
                self._collection = Collection(self.collection_name)
                self._logger.info(f"Using existing collection: {self.collection_name}")
                return
            
            # Define the schema for our collection
            fields = [
                FieldSchema(
                    name=self.ID_FIELD,
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    max_length=65535,
                ),
                FieldSchema(
                    name=self.VECTOR_FIELD,
                    dtype=DataType.FLOAT_VECTOR,
                    dim=self.dimension,
                ),
                FieldSchema(
                    name=self.DOCUMENT_FIELD,
                    dtype=DataType.VARCHAR,
                    max_length=65535,
                ),
                FieldSchema(
                    name=self.METADATA_FIELD,
                    dtype=DataType.JSON,
                ),
                FieldSchema(
                    name=self.NAMESPACE_FIELD,
                    dtype=DataType.VARCHAR,
                    max_length=255,
                ),
            ]
            
            schema = CollectionSchema(fields=fields, description="Semantrix vector store")
            
            # Create the collection
            self._collection = Collection(
                name=self.collection_name,
                schema=schema,
                using="default",
                shards_num=2,  # Default number of shards
            )
            
            # Create an index for the vector field
            index_params = {
                "index_type": "IVF_FLAT",
                "metric_type": self._metric_type,
                "params": {"nlist": 1024},
            }
            
            self._collection.create_index(
                field_name=self.VECTOR_FIELD,
                index_params=index_params,
            )
            
            # Load the collection to memory
            self._collection.load()
            
            self._logger.info(f"Created new collection: {self.collection_name}")
            
        except MilvusException as e:
            self._logger.error(f"Failed to ensure collection: {str(e)}")
            raise VectorOperationError("Failed to ensure collection") from e
    
    @classmethod
    def from_connection_params(
        cls,
        dimension: int,
        host: str = "localhost",
        port: int = 19530,
        user: str = "",
        password: str = "",
        collection_name: str = DEFAULT_COLLECTION_NAME,
        **kwargs: Any
    ) -> 'MilvusVectorStore':
        """Create a MilvusVectorStore from connection parameters.
        
        Args:
            dimension: The dimension of the vectors
            host: Milvus server host (default: "localhost")
            port: Milvus server port (default: 19530)
            user: Username for authentication
            password: Password for authentication
            collection_name: Name of the collection to use
            **kwargs: Additional arguments for MilvusVectorStore
            
        Returns:
            Configured MilvusVectorStore instance
            
        Example:
            ```python
            store = MilvusVectorStore.from_connection_params(
                dimension=768,
                host="localhost",
                port=19530,
                collection_name="my_vectors"
            )
            ```
        """
        # Build URI from host and port
        uri = f"http://{host}:{port}"
        
        # Create auth token if username/password provided
        token = None
        if user and password:
            token = f"{user}:{password}"
        
        return cls(
            dimension=dimension,
            collection_name=collection_name,
            uri=uri,
            token=token,
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
        if not self._collection:
            raise VectorOperationError("Collection not initialized")
            
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
        
        # Prepare data for batch insert
        data = []
        for i, (vec, vec_id) in enumerate(zip(vectors, ids)):
            # Convert vector to list if it's a numpy array
            vector = vec.tolist() if hasattr(vec, 'tolist') else list(vec)
            
            # Prepare document and metadata
            doc = documents[i] if documents and i < len(documents) else ""
            meta = metadatas[i] if metadatas and i < len(metadatas) else {}
            
            # Add document to metadata if provided
            if doc:
                meta["document"] = doc
            
            # Add to batch
            data.append({
                self.ID_FIELD: vec_id,
                self.VECTOR_FIELD: vector,
                self.DOCUMENT_FIELD: doc,
                self.METADATA_FIELD: json.dumps(meta),
                self.NAMESPACE_FIELD: self.namespace or ""
            })
        
        # Perform batch insert
        try:
            # Convert data to the format expected by Milvus
            insert_data = {field: [] for field in data[0].keys()}
            for item in data:
                for field, value in item.items():
                    insert_data[field].append(value)
            
            # Insert data
            result = self._collection.insert(insert_data)
            self._collection.flush()
            
            # Return the generated IDs
            return [str(id_) for id_ in result.primary_keys]
            
        except MilvusException as e:
            self._logger.error(f"Failed to add vectors to Milvus: {str(e)}")
            raise VectorOperationError("Failed to add vectors to Milvus") from e
    
    async def get(
        self,
        ids: Optional[Union[str, List[str]]] = None,
        filter: Optional[MetadataFilter] = None,
        include_vectors: bool = True,
        **kwargs: Any
    ) -> List[VectorRecord]:
        """Get vectors from the store."""
        if not self._collection:
            raise VectorOperationError("Collection not initialized")
            
        # Convert single ID to list
        if ids is not None and not isinstance(ids, list):
            ids = [ids]
        
        # Build query expression
        expr_parts = []
        params = {}
        
        # Add ID filter if provided
        if ids:
            id_list = ", ".join([f"'{id_}'" for id_ in ids])
            expr_parts.append(f"{self.ID_FIELD} in [{id_list}]")
        
        # Add namespace filter
        if self.namespace:
            expr_parts.append(f"{self.NAMESPACE_FIELD} == '{self.namespace}'")
        
        # Add metadata filters
        if filter:
            for key, value in filter.items():
                if isinstance(value, dict):
                    # Handle operators like $eq, $gt, etc.
                    for op, op_value in value.items():
                        if op == "$eq":
                            expr_parts.append(f"json_contains_any({self.METADATA_FIELD}['{key}'], ['{op_value}'])")
                        elif op == "$ne":
                            expr_parts.append(f"not json_contains_any({self.METADATA_FIELD}['{key}'], ['{op_value}'])")
                        elif op in ["$gt", "$gte", "$lt", "$lte"]:
                            expr_parts.append(f"{self.METADATA_FIELD}['{key}'] {op} {op_value}")
                        elif op == "$in":
                            if not isinstance(op_value, (list, tuple)):
                                raise ValueError("$in operator requires a list value")
                            values = ", ".join([f"'{v}'" for v in op_value])
                            expr_parts.append(f"{self.METADATA_FIELD}['{key}'] in [{values}]")
                else:
                    # Default to equality check
                    expr_parts.append(f"json_contains_any({self.METADATA_FIELD}['{key}'], ['{value}'])")
        
        # Combine all conditions
        expr = " and ".join(expr_parts) if expr_parts else ""
        
        # Select fields to return
        output_fields = [self.ID_FIELD, self.DOCUMENT_FIELD, self.METADATA_FIELD]
        if include_vectors:
            output_fields.append(self.VECTOR_FIELD)
        
        try:
            # Execute query
            results = self._collection.query(
                expr=expr,
                output_fields=output_fields,
                **kwargs
            )
            
            # Convert results to VectorRecord objects
            records = []
            for result in results:
                vec_id = result[self.ID_FIELD]
                doc = result.get(self.DOCUMENT_FIELD, "")
                meta = json.loads(result.get(self.METADATA_FIELD, "{}"))
                vector = result.get(self.VECTOR_FIELD) if include_vectors else None
                
                records.append(VectorRecord(
                    id=vec_id,
                    document=doc,
                    metadata=meta,
                    embedding=vector
                ))
            
            return records
            
        except MilvusException as e:
            self._logger.error(f"Failed to get vectors from Milvus: {str(e)}")
            raise VectorOperationError("Failed to get vectors from Milvus") from e
    
    async def search(
        self,
        query_vector: Vector,
        k: int = 10,
        filter: Optional[MetadataFilter] = None,
        **kwargs: Any
    ) -> List[QueryResult]:
        """Search for similar vectors."""
        if not self._collection:
            raise VectorOperationError("Collection not initialized")
            
        # Convert query vector to list if it's a numpy array
        query_vec = query_vector.tolist() if hasattr(query_vector, 'tolist') else list(query_vector)
        
        # Build search parameters
        search_params = {
            "metric_type": self._metric_type,
            "params": {"nprobe": 10},  # Number of clusters to search
        }
        
        # Build filter expression
        expr_parts = []
        
        # Add namespace filter
        if self.namespace:
            expr_parts.append(f"{self.NAMESPACE_FIELD} == '{self.namespace}'")
        
        # Add metadata filters
        if filter:
            for key, value in filter.items():
                if isinstance(value, dict):
                    # Handle operators
                    for op, op_value in value.items():
                        if op == "$eq":
                            expr_parts.append(f"json_contains_any({self.METADATA_FIELD}['{key}'], ['{op_value}'])")
                        elif op == "$ne":
                            expr_parts.append(f"not json_contains_any({self.METADATA_FIELD}['{key}'], ['{op_value}'])")
                        elif op in ["$gt", "$gte", "$lt", "$lte"]:
                            expr_parts.append(f"{self.METADATA_FIELD}['{key}'] {op} {op_value}")
                        elif op == "$in":
                            if not isinstance(op_value, (list, tuple)):
                                raise ValueError("$in operator requires a list value")
                            values = ", ".join([f"'{v}'" for v in op_value])
                            expr_parts.append(f"{self.METADATA_FIELD}['{key}'] in [{values}]")
                else:
                    # Default to equality check
                    expr_parts.append(f"json_contains_any({self.METADATA_FIELD}['{key}'], ['{value}'])")
        
        # Add tombstone filter to exclude tombstoned vectors
        expr_parts.append(f"not json_contains_any({self.METADATA_FIELD}['tombstoned'], ['true'])")
        
        # Combine all conditions
        expr = " and ".join(expr_parts) if expr_parts else ""
        
        try:
            # Execute search
            results = self._collection.search(
                data=[query_vec],
                anns_field=self.VECTOR_FIELD,
                param=search_params,
                limit=k,
                expr=expr,
                output_fields=[self.ID_FIELD, self.DOCUMENT_FIELD, self.METADATA_FIELD],
                **kwargs
            )
            
            # Convert results to QueryResult format
            query_results = []
            for hits in results:
                for hit in hits:
                    vec_id = hit.id
                    score = hit.score
                    doc = hit.entity.get(self.DOCUMENT_FIELD, "")
                    meta = json.loads(hit.entity.get(self.METADATA_FIELD, "{}"))
                    
                    # For cosine similarity, convert from distance to similarity
                    if self._metric_type == MetricType.COSINE:
                        score = 1.0 - score
                    
                    query_results.append({
                        "id": vec_id,
                        "document": doc,
                        "metadata": meta,
                        "score": float(score),
                        "vector": hit.entity.get(self.VECTOR_FIELD)
                    })
            
            return query_results
            
        except MilvusException as e:
            self._logger.error(f"Failed to search vectors in Milvus: {str(e)}")
            raise VectorOperationError("Failed to search vectors in Milvus") from e
    
    async def update(
        self,
        ids: Union[str, List[str]],
        vectors: Optional[Union[Vector, List[Vector]]] = None,
        documents: Optional[Union[str, List[str]]] = None,
        metadatas: Optional[Union[Metadata, List[Metadata]]] = None,
        **kwargs: Any
    ) -> None:
        """Update vectors in the store."""
        if not self._collection:
            raise VectorOperationError("Collection not initialized")
            
        # Convert inputs to lists
        if not isinstance(ids, list):
            ids = [ids]
        
        # Get existing records
        existing = await self.get(ids=ids, include_vectors=vectors is not None)
        existing_map = {r.id: r for r in existing}
        
        # Prepare updates
        updates = []
        for i, vec_id in enumerate(ids):
            if vec_id not in existing_map:
                self._logger.warning(f"Vector with id {vec_id} not found, skipping update")
                continue
                
            # Get existing record
            record = existing_map[vec_id]
            
            # Update fields if provided
            if vectors is not None:
                if isinstance(vectors, list) and i < len(vectors):
                    vector = vectors[i]
                    record.embedding = vector.tolist() if hasattr(vector, 'tolist') else list(vector)
            
            if documents is not None:
                if isinstance(documents, list) and i < len(documents):
                    record.document = documents[i]
            
            if metadatas is not None:
                if isinstance(metadatas, list) and i < len(metadatas):
                    record.metadata.update(metadatas[i] or {})
            
            # Add to updates
            updates.append({
                self.ID_FIELD: vec_id,
                self.VECTOR_FIELD: record.embedding,
                self.DOCUMENT_FIELD: record.document,
                self.METADATA_FIELD: json.dumps(record.metadata),
                self.NAMESPACE_FIELD: self.namespace or ""
            })
        
        # Perform batch update
        if updates:
            try:
                # Convert updates to the format expected by Milvus
                update_data = {field: [] for field in updates[0].keys()}
                for item in updates:
                    for field, value in item.items():
                        update_data[field].append(value)
                
                # Upsert the data (update if exists, insert if not)
                self._collection.upsert(update_data)
                self._collection.flush()
                
            except MilvusException as e:
                self._logger.error(f"Failed to update vectors in Milvus: {str(e)}")
                raise VectorOperationError("Failed to update vectors in Milvus") from e
    
    async def delete(
        self,
        ids: Optional[Union[str, List[str]]] = None,
        filter: Optional[MetadataFilter] = None,
        documents: Optional[Union[str, List[str]]] = None,
        **kwargs: Any
    ) -> None:
        """Delete vectors by IDs, filter, or documents."""
        if not self._collection:
            raise VectorOperationError("Collection not initialized")
        
        # Check if any tombstoning-related parameters are passed
        if any(param is not None for param in [ids, filter, documents]):
            logger.info(f"Tombstoning requested for Milvus store, using direct deletion")
            # Fall back to direct deletion for external stores
        
        await self._direct_delete(ids=ids, filter=filter, documents=documents, **kwargs)
    
    async def _direct_delete(
        self,
        ids: Optional[Union[str, List[str]]] = None,
        filter: Optional[MetadataFilter] = None,
        documents: Optional[Union[str, List[str]]] = None,
        **kwargs: Any
    ) -> None:
        """Delete vectors from the store."""
        if not self._collection:
            raise VectorOperationError("Collection not initialized")
            
        # If neither IDs nor filter provided, delete all vectors
        if ids is None and filter is None:
            try:
                if self.namespace:
                    # Delete only vectors in this namespace
                    self._collection.delete(f"{self.NAMESPACE_FIELD} == '{self.namespace}'")
                else:
                    # Delete all vectors
                    self._collection.drop()
                    # Recreate the collection
                    self._ensure_collection()
                return
                    
            except MilvusException as e:
                self._logger.error(f"Failed to delete all vectors from Milvus: {str(e)}")
                raise VectorOperationError("Failed to delete all vectors from Milvus") from e
        
        # Build filter expression for targeted delete
        expr_parts = []
        
        # Add ID filter if provided
        if ids is not None:
            if not isinstance(ids, list):
                ids = [ids]
            id_list = ", ".join([f"'{id_}'" for id_ in ids])
            expr_parts.append(f"{self.ID_FIELD} in [{id_list}]")
        
        # Add namespace filter
        if self.namespace:
            expr_parts.append(f"{self.NAMESPACE_FIELD} == '{self.namespace}'")
        
        # Add metadata filters
        if filter:
            for key, value in filter.items():
                if isinstance(value, dict):
                    # Handle operators
                    for op, op_value in value.items():
                        if op == "$eq":
                            expr_parts.append(f"json_contains_any({self.METADATA_FIELD}['{key}'], ['{op_value}'])")
                        elif op == "$ne":
                            expr_parts.append(f"not json_contains_any({self.METADATA_FIELD}['{key}'], ['{op_value}'])")
                        elif op in ["$gt", "$gte", "$lt", "$lte"]:
                            expr_parts.append(f"{self.METADATA_FIELD}['{key}'] {op} {op_value}")
                        elif op == "$in":
                            if not isinstance(op_value, (list, tuple)):
                                raise ValueError("$in operator requires a list value")
                            values = ", ".join([f"'{v}'" for v in op_value])
                            expr_parts.append(f"{self.METADATA_FIELD}['{key}'] in [{values}]")
                else:
                    # Default to equality check
                    expr_parts.append(f"json_contains_any({self.METADATA_FIELD}['{key}'], ['{value}'])")
        
        # Execute delete with the combined expression
        if expr_parts:
            try:
                expr = " and ".join(expr_parts)
                self._collection.delete(expr)
                self._collection.flush()
                    
            except MilvusException as e:
                self._logger.error(f"Failed to delete vectors from Milvus: {str(e)}")
                raise VectorOperationError("Failed to delete vectors from Milvus") from e
    
    async def tombstone(
        self,
        ids: Optional[Union[str, List[str]]] = None,
        filter: Optional[MetadataFilter] = None,
        documents: Optional[Union[str, List[str]]] = None,
        **kwargs: Any
    ) -> bool:
        """Mark vectors as deleted (tombstoning) without removing them from storage."""
        if not self._collection:
            raise VectorOperationError("Collection not initialized")
            
        try:
            ids_to_tombstone = []
            
            # Collect IDs to tombstone
            if ids is not None:
                if isinstance(ids, str):
                    ids_to_tombstone = [ids]
                else:
                    ids_to_tombstone = ids
            elif documents is not None:
                # Find vectors by document content
                if isinstance(documents, str):
                    doc_list = [documents]
                else:
                    doc_list = documents
                
                for doc in doc_list:
                    filter_doc = MetadataFilter(field="document", value=doc, operator="eq")
                    results = await self.get(filter=filter_doc, include_vectors=False)
                    if results:
                        ids_to_tombstone.extend([r.id for r in results])
            elif filter is not None:
                results = await self.get(filter=filter, include_vectors=False)
                if results:
                    ids_to_tombstone.extend([r.id for r in results])
            
            if not ids_to_tombstone:
                return False
            
            # Mark as tombstoned by updating metadata
            for vec_id in ids_to_tombstone:
                try:
                    # Get existing record
                    existing = await self.get(ids=[vec_id], include_vectors=True)
                    if not existing:
                        continue
                    
                    record = existing[0]
                    metadata = dict(record.metadata or {})
                    metadata["tombstoned"] = True
                    metadata["tombstoned_at"] = int(time.time() * 1000)
                    
                    # Update the record
                    self._collection.upsert(
                        data=[
                            [vec_id],  # ID
                            [record.embedding],  # Vector
                            [record.document or ""],  # Document
                            [json.dumps(metadata)]  # Metadata
                        ]
                    )
                except MilvusException as e:
                    self._logger.warning(f"Failed to tombstone vector {vec_id}: {e}")
                    continue
            
            return True
            
        except MilvusException as e:
            self._logger.error(f"Failed to tombstone vectors in Milvus: {str(e)}")
            raise VectorOperationError("Failed to tombstone vectors in Milvus") from e
    
    async def purge_tombstones(self) -> int:
        """Permanently remove all tombstoned vectors from storage."""
        if not self._collection:
            raise VectorOperationError("Collection not initialized")
            
        try:
            # Find all tombstoned vectors
            filter_tombstone = MetadataFilter(field="tombstoned", value=True, operator="eq")
            results = await self.get(filter=filter_tombstone, include_vectors=False)
            
            if not results:
                return 0
            
            # Delete tombstoned vectors
            ids_to_delete = [r.id for r in results]
            id_list = ", ".join([f"'{id_}'" for id_ in ids_to_delete])
            expr = f"{self.ID_FIELD} in [{id_list}]"
            
            self._collection.delete(expr)
            self._collection.flush()
            
            return len(ids_to_delete)
            
        except MilvusException as e:
            self._logger.error(f"Failed to purge tombstones from Milvus: {str(e)}")
            raise VectorOperationError("Failed to purge tombstones from Milvus") from e
    
    async def count(self, **kwargs: Any) -> int:
        """Count the number of vectors in the store."""
        if not self._collection:
            raise VectorOperationError("Collection not initialized")
            
        try:
            # Build filter expression
            expr = f"{self.NAMESPACE_FIELD} == '{self.namespace}'" if self.namespace else ""
            
            # Get collection stats
            stats = self._collection.get_replica_info()
            
            # If we have a namespace filter, we need to do a count query
            if expr:
                result = self._collection.query(
                    expr=expr,
                    output_fields=[self.ID_FIELD],
                    limit=1,  # We just need the count
                    **kwargs
                )
                return len(result)
            else:
                # Otherwise, use the collection stats
                return stats.row_count
                
        except MilvusException as e:
            self._logger.error(f"Failed to count vectors in Milvus: {str(e)}")
            raise VectorOperationError("Failed to count vectors in Milvus") from e
    
    async def reset(self, **kwargs: Any) -> None:
        """Reset the vector store by dropping and recreating the collection."""
        if not self._collection:
            raise VectorOperationError("Collection not initialized")
            
        try:
            # Drop the collection
            utility.drop_collection(self.collection_name)
            
            # Recreate the collection
            self._ensure_collection()
                
        except MilvusException as e:
            self._logger.error(f"Failed to reset Milvus store: {str(e)}")
            raise VectorOperationError("Failed to reset Milvus store") from e
    
    async def close(self, **kwargs: Any) -> None:
        """Close the Milvus connection."""
        try:
            if self._collection:
                self._collection.release()
            connections.disconnect("default")
        except MilvusException as e:
            self._logger.error(f"Error closing Milvus connection: {str(e)}")
            raise VectorOperationError("Error closing Milvus connection") from e
