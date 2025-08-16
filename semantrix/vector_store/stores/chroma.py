"""Chroma vector store implementation.

This module provides a vector store implementation using ChromaDB, supporting both
local in-memory/persistent storage and remote server connections.
"""

import os
import logging
import uuid
from typing import Any, ClassVar, Dict, List, Optional, Union, cast, overload

from chromadb import Client, Collection
from chromadb.config import Settings as ChromaSettings

from ..base import (
    BaseVectorStore,
    DistanceMetric,
    IndexType,
    Metadata,
    MetadataFilter,
    QueryResult,
    Vector,
    VectorRecord,
)

class ChromaVectorStore(BaseVectorStore):
    """ChromaDB-based vector store implementation.
    
    Supports both local storage (in-memory or persistent) and remote Chroma server connections.
    For server connections, use the `from_server` classmethod for better ergonomics.
    """
    
    def __init__(
        self,
        dimension: int,
        metric: DistanceMetric = DistanceMetric.COSINE,
        namespace: Optional[str] = None,
        persist_directory: Optional[str] = None,
        in_memory: bool = True,
        **kwargs: Any
    ) -> None:
        super().__init__(dimension=dimension, metric=metric, namespace=namespace)
        self._logger = logging.getLogger(__name__)
        self.in_memory = in_memory
        self.persist_directory = persist_directory
        self._chroma_metric = self._get_chroma_metric(metric)
        self._client = self._init_chroma_client(**kwargs)
        self._collection: Optional[Collection] = None
        self._init_collection()
    
    @classmethod
    def from_server(
        cls,
        dimension: int,
        host: str,
        port: int = 8000,
        ssl: bool = False,
        api_key: Optional[str] = None,
        namespace: Optional[str] = None,
        metric: DistanceMetric = DistanceMetric.COSINE,
        **kwargs: Any
    ) -> 'ChromaVectorStore':
        """Create a ChromaVectorStore connected to a remote Chroma server.
        
        Args:
            dimension: The dimension of the vectors to be stored
            host: Chroma server hostname or IP address
            port: Chroma server port (default: 8000)
            ssl: Whether to use HTTPS (default: False)
            api_key: Optional API key for authentication
            namespace: Optional namespace for multi-tenancy
            metric: Distance metric to use for similarity search
            **kwargs: Additional Chroma client settings
            
        Returns:
            ChromaVectorStore instance connected to the specified server
            
        Example:
            ```python
            # Connect to a Chroma server
            store = ChromaVectorStore.from_server(
                dimension=768,
                host="chroma-server.example.com",
                port=8000,
                ssl=True,
                api_key="your-api-key"
            )
            ```
        """
        settings: Dict[str, Any] = {
            'chroma_api_impl': 'rest',
            'chroma_server_host': host,
            'chroma_server_http_port': port,
            'chroma_server_ssl': ssl
        }
        
        if api_key:
            settings['chroma_server_auth_credentials'] = api_key
            
        # Pass any additional settings, allowing overrides of the defaults
        settings.update(kwargs)
        
        return cls(
            dimension=dimension,
            metric=metric,
            namespace=namespace,
            in_memory=False,  # Ignored when using server mode
            persist_directory=None,  # Ignored when using server mode
            chroma_settings=settings
        )
    
    def _init_chroma_client(self, **kwargs: Any) -> Client:
        """Initialize the Chroma client with appropriate settings.
        
        This handles both local (in-memory/persistent) and server connection modes.
        """
        # If chroma_settings is provided, use it directly (for server connections)
        if 'chroma_settings' in kwargs:
            return Client(settings=ChromaSettings(**kwargs['chroma_settings']))
            
        # Otherwise, configure for local storage
        settings: Dict[str, Any] = {}
        if self.in_memory:
            settings['is_persistent'] = False
        elif self.persist_directory:
            settings['is_persistent'] = True
            settings['persist_directory'] = self.persist_directory
            os.makedirs(self.persist_directory, exist_ok=True)
            
        # Apply any additional settings
        settings.update(kwargs.get('chroma_settings', {}))
        
        return Client(settings=ChromaSettings(**settings))
    
    def _get_chroma_metric(self, metric: DistanceMetric) -> str:
        return {
            DistanceMetric.COSINE: 'cosine',
            DistanceMetric.EUCLIDEAN: 'l2',
            DistanceMetric.MANHATTAN: 'l1',
        }.get(metric, 'cosine')
    
    def _get_collection_name(self) -> str:
        name_parts = ["semantrix"]
        if self.namespace:
            name_parts.append(self.namespace)
        name_parts.extend([f"d{self.dimension}", self.metric.name.lower()])
        return "_".join(name_parts)
    
    def _init_collection(self) -> None:
        if self._collection is not None:
            return
        collection_name = self._get_collection_name()
        try:
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": self._chroma_metric},
                embedding_function=None,
            )
            self._logger.info(f"Initialized Chroma collection: {collection_name}")
        except Exception as e:
            self._logger.error(f"Failed to initialize Chroma collection: {str(e)}")
            raise
    
    async def add(
        self,
        vectors: Union[Vector, List[Vector]],
        documents: Optional[Union[str, List[str]]] = None,
        metadatas: Optional[Union[Metadata, List[Metadata]]] = None,
        ids: Optional[Union[str, List[str]]] = None,
        **kwargs: Any
    ) -> List[str]:
        if self._collection is None:
            raise RuntimeError("Chroma collection not initialized")
            
        is_single = not isinstance(vectors, list)
        if is_single:
            vectors = [cast(Vector, vectors)]
            documents = [documents] if documents is not None else None
            metadatas = [metadatas] if metadatas is not None else None
            ids = [ids] if ids is not None else None
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        
        if documents is None:
            documents = ["" for _ in range(len(vectors))]
        
        if metadatas is None:
            metadatas = [{} for _ in range(len(vectors))]
        
        if self.namespace:
            for meta in metadatas:
                if meta is not None:
                    meta["namespace"] = self.namespace
        
        self._collection.add(
            embeddings=vectors,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        return ids

    async def get(
        self,
        ids: Optional[Union[str, List[str]]] = None,
        filter: Optional[MetadataFilter] = None,
        include_vectors: bool = False,
        **kwargs: Any
    ) -> List[VectorRecord]:
        if self._collection is None:
            return []
            
        results: List[VectorRecord] = []
        
        if ids is not None:
            if isinstance(ids, str):
                ids = [ids]
                
            records = self._collection.get(
                ids=ids,
                include=["embeddings", "metadatas", "documents"] if include_vectors 
                       else ["metadatas", "documents"]
            )
            
            for i, id in enumerate(ids):
                if id in records["ids"]:
                    idx = records["ids"].index(id)
                    results.append(VectorRecord(
                        id=id,
                        embedding=records["embeddings"][idx] if include_vectors and records["embeddings"] else None,
                        document=records["documents"][idx] if records["documents"] else None,
                        metadata=records["metadatas"][idx] if records["metadatas"] else {},
                        namespace=self.namespace
                    ))
        elif filter is not None:
            chroma_filter = self._convert_filter(filter)
            records = self._collection.get(
                where=chroma_filter,
                include=["embeddings", "metadatas", "documents"] if include_vectors 
                       else ["metadatas", "documents"]
            )
            
            for i, id in enumerate(records["ids"]):
                results.append(VectorRecord(
                    id=id,
                    embedding=records["embeddings"][i] if include_vectors and records["embeddings"] else None,
                    document=records["documents"][i] if records["documents"] else None,
                    metadata=records["metadatas"][i] if records["metadatas"] else {},
                    namespace=self.namespace
                ))
        
        return results
    
    def _convert_filter(self, filter: MetadataFilter) -> Dict[str, Any]:
        if not filter:
            return {}
        return {key: {"$eq": value} for key, value in filter.items()}
    
    async def search(
        self,
        query_vector: Vector,
        k: int = 4,
        filter: Optional[MetadataFilter] = None,
        include_vectors: bool = False,
        include_metadata: bool = True,
        **kwargs: Any
    ) -> List[QueryResult]:
        if self._collection is None:
            return []
            
        chroma_filter = self._convert_filter(filter) if filter else None
        
        results = self._collection.query(
            query_embeddings=[query_vector],
            n_results=k,
            where=chroma_filter,
            include=["embeddings", "metadatas", "documents", "distances"]
        )
        
        query_results: List[QueryResult] = []
        
        if results and "ids" in results and results["ids"]:
            for i, id in enumerate(results["ids"][0]):
                if not id:
                    continue
                    
                distance = results["distances"][0][i] if results.get("distances") else None
                score = 1.0 / (1.0 + distance) if distance is not None else 1.0
                
                metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                document = results["documents"][0][i] if results.get("documents") else None
                
                result: QueryResult = {
                    "id": id,
                    "document": document,
                    "metadata": metadata if include_metadata else None,
                    "score": float(score),
                    "vector": results["embeddings"][0][i] if include_vectors and results.get("embeddings") else None
                }
                
                query_results.append(result)
        
        return query_results
    
    async def update(
        self,
        ids: Union[str, List[str]],
        vectors: Optional[Union[Vector, List[Vector]]] = None,
        documents: Optional[Union[str, List[str], None]] = None,
        metadatas: Optional[Union[Metadata, List[Metadata], None]] = None,
        **kwargs: Any
    ) -> None:
        if self._collection is None:
            raise RuntimeError("Chroma collection not initialized")
            
        if isinstance(ids, str):
            ids = [ids]
            
        if vectors is not None and not isinstance(vectors, list):
            vectors = [vectors]
            
        if documents is not None and not isinstance(documents, list):
            documents = [documents]
            
        if metadatas is not None and not isinstance(metadatas, list):
            metadatas = [metadatas]
        
        existing = await self.get(ids=ids, include_vectors=True)
        existing_map = {r.id: r for r in existing}
        
        update_vectors = []
        update_documents = []
        update_metadatas = []
        update_ids = []
        
        for i, id in enumerate(ids):
            if id not in existing_map:
                self._logger.warning(f"ID not found, skipping update: {id}")
                continue
                
            record = existing_map[id]
            
            if vectors is not None and i < len(vectors):
                record.embedding = vectors[i]
            
            if documents is not None and i < len(documents):
                record.document = documents[i]
            
            if metadatas is not None and i < len(metadatas):
                if metadatas[i] is None:
                    record.metadata = {}
                else:
                    if record.metadata is None:
                        record.metadata = {}
                    record.metadata.update(metadatas[i])
            
            update_vectors.append(record.embedding)
            update_documents.append(record.document or "")
            update_metadatas.append(record.metadata or {})
            update_ids.append(id)
        
        if update_ids:
            self._collection.delete(ids=update_ids)
            self._collection.add(
                embeddings=update_vectors,
                documents=update_documents,
                metadatas=update_metadatas,
                ids=update_ids
            )
    
    async def delete(
        self,
        ids: Optional[Union[str, List[str]]] = None,
        filter: Optional[MetadataFilter] = None,
        **kwargs: Any
    ) -> None:
        if self._collection is None:
            return
            
        if ids is not None:
            if isinstance(ids, str):
                ids = [ids]
            self._collection.delete(ids=ids)
        elif filter is not None:
            chroma_filter = self._convert_filter(filter)
            results = self._collection.get(
                where=chroma_filter,
                include=[],
                limit=10000
            )
            if results and results["ids"]:
                self._collection.delete(ids=results["ids"])
    
    async def count(self, filter: Optional[MetadataFilter] = None) -> int:
        if self._collection is None:
            return 0
            
        if filter is None:
            return self._collection.count()
        else:
            chroma_filter = self._convert_filter(filter)
            results = self._collection.get(
                where=chroma_filter,
                include=[],
                limit=100000
            )
            return len(results["ids"]) if results and "ids" in results else 0
    
    async def reset(self) -> None:
        if self._collection is not None:
            self._collection.delete(where={})  # Delete all records
    
    async def close(self) -> None:
        """Close the Chroma client and release resources."""
        if hasattr(self, '_client') and self._client is not None:
            self._client.heartbeat()  # Ensure any pending writes are flushed
            self._client = None
        self._collection = None

    async def create_index(
        self,
        index_type: IndexType = IndexType.FLAT,
        metric: Optional[DistanceMetric] = None,
        **kwargs: Any
    ) -> bool:
        """
        Create or update the vector index.
        
        Note: ChromaDB handles index creation automatically, so this is a no-op.
        """
        try:
            # ChromaDB creates indexes automatically, so we just ensure the collection exists
            if self._collection is None:
                self._init_collection()
            return True
        except Exception as e:
            self._logger.error(f"Error creating index: {e}")
            return False
    
    async def get_index_info(self) -> Dict[str, Any]:
        """Get information about the current index."""
        try:
            if self._collection is None:
                return {"error": "No collection initialized"}
            
            # Get collection info
            collection_info = self._collection.get()
            return {
                "collection_name": self._collection.name,
                "count": len(collection_info.get("ids", [])),
                "metadata": collection_info.get("metadatas", []),
                "dimension": self.dimension,
                "metric": self._chroma_metric
            }
        except Exception as e:
            self._logger.error(f"Error getting index info: {e}")
            return {"error": str(e)}
    
    async def list_collections(self) -> List[str]:
        """List all collections in the store."""
        try:
            collections = self._client.list_collections()
            return [col.name for col in collections]
        except Exception as e:
            self._logger.error(f"Error listing collections: {e}")
            return []
    
    async def delete_collection(self, name: str, **kwargs: Any) -> bool:
        """Delete a collection."""
        try:
            self._client.delete_collection(name=name)
            return True
        except Exception as e:
            self._logger.error(f"Error deleting collection {name}: {e}")
            return False
    
    async def count(self, filter: Optional[MetadataFilter] = None) -> int:
        """Count the number of vectors matching the filter."""
        try:
            if self._collection is None:
                return 0
            
            # Get all documents and count them
            result = self._collection.get()
            return len(result.get("ids", []))
        except Exception as e:
            self._logger.error(f"Error counting vectors: {e}")
            return 0
