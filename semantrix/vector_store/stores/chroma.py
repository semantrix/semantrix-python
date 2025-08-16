"""Chroma vector store implementation.

This module provides a vector store implementation using ChromaDB, supporting both
local in-memory/persistent storage and remote server connections.
"""

import os
import logging
import uuid
from typing import Any, ClassVar, Dict, List, Optional, Union, cast, overload
import asyncio
import numpy as np

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
        """Create a ChromaVectorStore connected to a remote Chroma server."""
        settings: Dict[str, Any] = {
            'chroma_api_impl': 'rest',
            'chroma_server_host': host,
            'chroma_server_http_port': port,
        }
        
        # Only add SSL setting if it's True
        if ssl:
            settings['chroma_server_ssl_enabled'] = True
        
        if api_key:
            settings['chroma_server_auth_credentials'] = api_key
            
        # Pass any additional settings, allowing overrides of the defaults
        settings.update(kwargs)
        
        return cls(
            dimension=dimension,
            metric=metric,
            namespace=namespace,
            **settings
        )
    
    def _init_chroma_client(self, **kwargs: Any) -> Client:
        """Initialize the Chroma client with appropriate settings."""
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
        """Add vectors to the store."""
        # Debug logging
        self._logger.debug(f"Input vectors type: {type(vectors)}, length: {len(vectors) if hasattr(vectors, '__len__') else 'N/A'}")
        
        # Ensure vectors is a list
        if not isinstance(vectors, list):
            vectors = [vectors]
        
        # Ensure documents is a list
        if documents is not None and not isinstance(documents, list):
            documents = [documents]
        
        # Ensure metadatas is a list
        if metadatas is not None and not isinstance(metadatas, list):
            metadatas = [metadatas]
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        elif not isinstance(ids, list):
            ids = [ids]
        
        # Ensure all lists have the same length
        max_len = len(vectors)
        if documents is not None:
            documents = documents + [None] * (max_len - len(documents))
        if metadatas is not None:
            metadatas = metadatas + [{}] * (max_len - len(metadatas))
        
        # Convert vectors to numpy arrays
        vectors_array = [np.array(v, dtype=np.float32) for v in vectors]
        
        # Prepare metadata with placeholder if needed
        if metadatas is None:
            metadatas = [{"_placeholder": "true"} for _ in range(max_len)]
        else:
            for i, meta in enumerate(metadatas):
                if meta is None:
                    metadatas[i] = {"_placeholder": "true"}
                elif not meta:  # Empty dict
                    metadatas[i] = {"_placeholder": "true"}
        
        # Prepare documents - use empty string if None
        if documents is None:
            documents = ["" for _ in range(max_len)]
        else:
            documents = [doc if doc is not None else "" for doc in documents]
        
        def _add_chroma():
            self._collection.add(
                embeddings=vectors_array,
                documents=documents,  # Pass documents directly
                metadatas=metadatas,
                ids=ids
            )
        
        await asyncio.get_event_loop().run_in_executor(None, _add_chroma)
        
        return ids

    def _convert_filter(self, filter: MetadataFilter) -> Dict[str, Any]:
        """Convert metadata filter to ChromaDB format."""
        if not filter:
            return {}
        
        chroma_filter = {}
        for key, value in filter.items():
            # Handle numpy arrays and other array-like objects
            if hasattr(value, '__len__') and not isinstance(value, (str, bytes)):
                # Convert arrays to lists to avoid numpy boolean evaluation issues
                if hasattr(value, 'tolist'):
                    value = value.tolist()
                elif hasattr(value, '__iter__'):
                    value = list(value)
            
            chroma_filter[key] = {"$eq": value}
        
        return chroma_filter
    
    def _safe_boolean_check(self, value) -> bool:
        """Safely check if a value is truthy, handling numpy arrays."""
        if hasattr(value, '__len__') and not isinstance(value, (str, bytes)):
            # For arrays, check if they have any elements
            return len(value) > 0
        return bool(value)

    async def get(
        self,
        ids: Optional[Union[str, List[str]]] = None,
        filter: Optional[MetadataFilter] = None,
        include_vectors: bool = False,
        **kwargs: Any
    ) -> List[VectorRecord]:
        if self._collection is None:
            return []
            
        def _get_from_chroma():
            results: List[VectorRecord] = []
            
            if ids is not None:
                if isinstance(ids, str):
                    ids_list = [ids]
                else:
                    ids_list = ids
                    
                # Always include documents in the query
                records = self._collection.get(
                    ids=ids_list,
                    include=["embeddings", "metadatas", "documents"] if include_vectors 
                       else ["metadatas", "documents"]
                )
                
                self._logger.debug(f"Retrieved records: {records}")
                
                for i, id in enumerate(ids_list):
                    if id in records["ids"] and self._safe_boolean_check(records["ids"]):
                        idx = records["ids"].index(id)
                        embedding = None
                        if include_vectors and self._safe_boolean_check(records["embeddings"]):
                            embedding = records["embeddings"][idx]
                        
                        # Get document from the documents field
                        document = None
                        if "documents" in records and self._safe_boolean_check(records["documents"]):
                            document = records["documents"][idx]
                            # Convert empty string back to None for consistency
                            if document == "":
                                document = None
                        
                        results.append(VectorRecord(
                            id=id,
                            embedding=embedding,
                            document=document,
                            metadata=records["metadatas"][idx] if self._safe_boolean_check(records["metadatas"]) else {},
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
                    embedding = None
                    if include_vectors and self._safe_boolean_check(records["embeddings"]):
                        embedding = records["embeddings"][i]
                    
                    # Get document from the documents field
                    document = None
                    if "documents" in records and self._safe_boolean_check(records["documents"]):
                        document = records["documents"][i]
                        # Convert empty string back to None for consistency
                        if document == "":
                            document = None
                    
                    results.append(VectorRecord(
                        id=id,
                        embedding=embedding,
                        document=document,
                        metadata=records["metadatas"][i] if self._safe_boolean_check(records["metadatas"]) else {},
                        namespace=self.namespace
                    ))
            
            return results
        
        return await asyncio.get_event_loop().run_in_executor(None, _get_from_chroma)
    
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
        def _search_chroma():
            try:
                # Convert filter to ChromaDB format if provided
                where = None
                if filter:
                    where = self._convert_filter(filter)
                
                # Perform search
                results = self._collection.query(
                    query_embeddings=[query_vector],
                    n_results=k,
                    where=where,
                    include=["metadatas", "documents", "distances"],
                    **kwargs
                )
                
                # Convert results to QueryResult format
                query_results = []
                if self._safe_boolean_check(results["ids"]) and len(results["ids"]) > 0:
                    for i, (id_val, distance, metadata, document) in enumerate(zip(
                        results["ids"][0],
                        results["distances"][0],
                        results["metadatas"][0],
                        results["documents"][0]
                    )):
                        query_results.append(QueryResult(
                            id=id_val,
                            score=1.0 - distance,  # Convert distance to similarity score
                            metadata=metadata,
                            document=document
                        ))
                
                return query_results
            except Exception as e:
                self._logger.error(f"Error searching ChromaDB: {e}")
                raise
        
        return await asyncio.get_event_loop().run_in_executor(None, _search_chroma)
    
    async def update(
        self,
        ids: Union[str, List[str]],
        vectors: Optional[Union[Vector, List[Vector]]] = None,
        documents: Optional[Union[str, List[str]]] = None,
        metadatas: Optional[Union[Metadata, List[Metadata]]] = None,
        **kwargs: Any
    ) -> bool:
        """Update vectors in the store."""
        if not self._collection:
            raise RuntimeError("Collection not initialized")
        
        # Ensure ids is a list
        if not isinstance(ids, list):
            ids = [ids]
        
        # Ensure vectors is a list
        if vectors is not None and not isinstance(vectors, list):
            vectors = [vectors]
        
        # Ensure documents is a list
        if documents is not None and not isinstance(documents, list):
            documents = [documents]
        
        # Ensure metadatas is a list
        if metadatas is not None and not isinstance(metadatas, list):
            metadatas = [metadatas]
        
        # Ensure all lists have the same length
        max_len = len(ids)
        if vectors is not None:
            vectors = vectors + [None] * (max_len - len(vectors))
        if documents is not None:
            documents = documents + [None] * (max_len - len(documents))
        if metadatas is not None:
            metadatas = metadatas + [{}] * (max_len - len(metadatas))
        
        # Convert vectors to numpy arrays
        if vectors is not None:
            vectors_array = [np.array(v, dtype=np.float32) if v is not None else None for v in vectors]
        else:
            vectors_array = None
        
        # Prepare metadata with placeholder if needed
        if metadatas is not None:
            for i, meta in enumerate(metadatas):
                if meta is None:
                    metadatas[i] = {"_placeholder": "true"}
                elif not meta:  # Empty dict
                    metadatas[i] = {"_placeholder": "true"}
        
        # Prepare documents - use empty string if None
        if documents is not None:
            documents = [doc if doc is not None else "" for doc in documents]
        
        def _update_chroma():
            self._collection.update(
                ids=ids,
                embeddings=vectors_array,
                documents=documents,
                metadatas=metadatas
            )
        
        await asyncio.get_event_loop().run_in_executor(None, _update_chroma)
        return True
    
    async def delete(
        self,
        ids: Optional[Union[str, List[str]]] = None,
        filter: Optional[MetadataFilter] = None,
        **kwargs: Any
    ) -> None:
        def _delete_from_chroma():
            if self._collection is None:
                return
                
            if ids is not None:
                if isinstance(ids, str):
                    ids_list = [ids]
                else:
                    ids_list = ids
                self._collection.delete(ids=ids_list)
            elif filter is not None:
                chroma_filter = self._convert_filter(filter)
                results = self._collection.get(
                    where=chroma_filter,
                    include=[],
                    limit=10000
                )
                    # Fix the array truth value issue with explicit checks
                if (results is not None and 
                    isinstance(results, dict) and
                    "ids" in results and 
                    results["ids"] is not None and 
                    isinstance(results["ids"], list) and
                    len(results["ids"]) > 0):
                    self._collection.delete(ids=results["ids"])
        
        await asyncio.get_event_loop().run_in_executor(None, _delete_from_chroma)
    
    async def create_index(
        self,
        index_type: IndexType = IndexType.FLAT,
        metric: Optional[DistanceMetric] = None,
        **kwargs: Any
    ) -> bool:
        """Create or update the vector index."""
        def _create_index():
            try:
                # ChromaDB creates indexes automatically, so we just ensure the collection exists
                if self._collection is None:
                    self._init_collection()
                return True
            except Exception as e:
                self._logger.error(f"Error creating index: {e}")
                return False
        
        return await asyncio.get_event_loop().run_in_executor(None, _create_index)
    
    async def get_index_info(self) -> Dict[str, Any]:
        """Get information about the current index."""
        def _get_index_info():
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
        
        return await asyncio.get_event_loop().run_in_executor(None, _get_index_info)
    
    async def list_collections(self) -> List[str]:
        """List all collections in the store."""
        def _list_collections():
            try:
                collections = self._client.list_collections()
                return [col.name for col in collections]
            except Exception as e:
                self._logger.error(f"Error listing collections: {e}")
                return []
        
        return await asyncio.get_event_loop().run_in_executor(None, _list_collections)
    
    async def delete_collection(self, name: str, **kwargs: Any) -> bool:
        """Delete a collection."""
        def _delete_collection():
            try:
                self._client.delete_collection(name=name)
                return True
            except Exception as e:
                self._logger.error(f"Error deleting collection {name}: {e}")
                return False
        
        return await asyncio.get_event_loop().run_in_executor(None, _delete_collection)
    
    async def count(self, filter: Optional[MetadataFilter] = None) -> int:
        """Count the number of vectors matching the filter."""
        def _count():
            try:
                if self._collection is None:
                    return 0
            
                # Get all documents and count them
                result = self._collection.get()
                return len(result.get("ids", []))
            except Exception as e:
                self._logger.error(f"Error counting vectors: {e}")
            return 0
        
        return await asyncio.get_event_loop().run_in_executor(None, _count)
    
    async def reset(self) -> None:
        if self._collection is not None:
            self._collection.delete(where={})  # Delete all records
    
    async def close(self) -> None:
        """Close the Chroma client and release resources."""
        if hasattr(self, '_client') and self._client is not None:
            self._client.heartbeat()  # Ensure any pending writes are flushed
            self._client = None
        self._collection = None
