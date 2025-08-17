import asyncio
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, cast
from uuid import uuid4

import numpy as np
import numpy.typing as npt

try:
    import faiss
    from faiss import Index, IndexFlatIP, IndexIVFFlat, IndexHNSWFlat
    from faiss.swigfaiss import IndexIDMap
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    faiss = None
    Index = object
    IndexFlatIP = object
    IndexIVFFlat = object
    IndexHNSWFlat = object
    IndexIDMap = object

from semantrix.exceptions import VectorOperationError
from semantrix.utils.logging import get_logger

from ..base import (
    BaseVectorStore, DistanceMetric, IndexType, Metadata, MetadataFilter, 
    QueryResult, Vector, VectorRecord
)

class FAISSVectorStore(BaseVectorStore):
    """
    FAISS-based vector store implementation with async support.
    
    This implementation uses FAISS (Facebook AI Similarity Search) for efficient
    similarity search of dense vectors. It's optimized for production use with
    async/await support, thread safety, and advanced features like metadata filtering,
    batch operations, and index management.
    
    Features:
    - Support for multiple index types (Flat, IVF, HNSW)
    - Metadata storage and filtering
    - Batch operations
    - Persistence to disk
    - Namespace support for multi-tenancy
    - Async/await interface
    - Thread-safe operations
    """
    
    def __init__(
        self,
        dimension: int,
        metric: DistanceMetric = DistanceMetric.COSINE,
        namespace: Optional[str] = None,
        index_type: IndexType = IndexType.FLAT,
        persist_path: Optional[str] = None,
        persist_on_disk: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Initialize the FAISS vector store.
        
        Args:
            dimension: The dimension of the embedding vectors
            metric: Distance metric to use for similarity search
            namespace: Optional namespace for multi-tenancy
            index_type: Type of FAISS index to use
            persist_path: Path to persist the index to disk (required if persist_on_disk is True)
            persist_on_disk: Whether to persist the index to disk. If True, persist_path must be provided.
            **kwargs: Additional FAISS index parameters
            
        Raises:
            ValueError: If persist_on_disk is True but no persist_path is provided
        """
        if not FAISS_AVAILABLE:
            raise VectorOperationError(
                "faiss-cpu is required for FAISSVectorStore. "
                "Install with: pip install faiss-cpu"
            )
            
        if persist_on_disk and not persist_path:
            raise VectorOperationError("persist_path must be provided when persist_on_disk is True")
            
        super().__init__(dimension=dimension, metric=metric, namespace=namespace)
        
        self.index_type = index_type
        self.persist_path = persist_path if persist_on_disk else None
        self._index: Optional[Index] = None
        self._id_to_record: Dict[str, VectorRecord] = {}
        self._string_id_to_int_id: Dict[str, int] = {}  # NEW: Mapping from string IDs to int IDs
        self._int_id_to_string_id: Dict[int, str] = {}  # NEW: Mapping from int IDs to string IDs
        self._next_int_id = 0  # NEW: Next available integer ID for FAISS
        self._lock = asyncio.Lock()
        self._index_params = kwargs
        self._is_trained = False
        
        # Initialize index
        self._init_index()
        
        # Load from disk if persistence is enabled and index exists
        if self.persist_path and os.path.exists(f"{self.persist_path}.index"):
            try:
                self._load_from_disk()
            except VectorOperationError as e:
                self._logger.warning(f"Failed to load index from {self.persist_path}: {e}")
                self._logger.warning("Starting with an empty index")
                self._init_index()  # Re-initialize if loading fails
    
    def _init_index(self) -> None:
        """Initialize the FAISS index based on configuration."""
        if not FAISS_AVAILABLE or faiss is None:
            raise VectorOperationError("FAISS is not available. Please install faiss-cpu or faiss-gpu.")
            
        # Map distance metric to FAISS metric
        if self.metric == DistanceMetric.COSINE:
            # Use inner product for cosine similarity (vectors must be normalized)
            metric = faiss.METRIC_INNER_PRODUCT
        elif self.metric == DistanceMetric.EUCLIDEAN:
            metric = faiss.METRIC_L2
        elif self.metric == DistanceMetric.MANHATTAN:
            metric = faiss.METRIC_L1
        else:
            raise VectorOperationError(f"Unsupported distance metric: {self.metric}")
        
        # Create appropriate index type - use simpler index without ID mapping
        if self.index_type == IndexType.FLAT:
            self._index = faiss.IndexFlatIP(self.dimension) if metric == faiss.METRIC_INNER_PRODUCT else \
                         faiss.IndexFlatL2(self.dimension)
        elif self.index_type == IndexType.IVF:
            nlist = self._index_params.get('nlist', 100)
            quantizer = faiss.IndexFlatL2(self.dimension) if metric == faiss.METRIC_L2 else \
                       faiss.IndexFlatIP(self.dimension)
            self._index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, metric)
        elif self.index_type == IndexType.HNSW:
            M = self._index_params.get('M', 16)  # Number of connections per layer
            self._index = faiss.IndexHNSWFlat(self.dimension, M, metric)
            self._index.hnsw.ef_construction = self._index_params.get('ef_construction', 200)
            self._index.hnsw.ef_search = self._index_params.get('ef_search', 50)
        else:
            raise VectorOperationError(f"Unsupported index type: {self.index_type}")
        
        # Don't wrap with IndexIDMap - use simple index
        self._is_trained = self.index_type == IndexType.FLAT  # Only Flat index doesn't need training
    
    def _normalize_vector(self, vector: Vector) -> npt.NDArray[np.float32]:
        """Convert and normalize a vector."""
        try:
            if vector is None:
                raise VectorOperationError("Cannot normalize None vector")

            if isinstance(vector, (list, tuple)) and not vector:  # Empty list/tuple
                raise VectorOperationError("Cannot normalize empty vector")

            if isinstance(vector, np.ndarray) and (vector.size == 0 or np.all(np.isnan(vector))):
                raise VectorOperationError("Cannot normalize empty or NaN vector")
                
            # Convert to numpy array if needed
            if not isinstance(vector, np.ndarray):
                vector = np.array(vector, dtype=np.float32)
            else:
                vector = np.asarray(vector, dtype=np.float32)
            
            # Handle scalar input
            if vector.ndim == 0:
                vector = np.array([vector], dtype=np.float32)
            
            # Ensure 2D array
            if vector.ndim == 1:
                vector = vector.reshape(1, -1)
            
            # Check for invalid shapes
            if vector.size == 0 or vector.shape[-1] == 0:
                raise VectorOperationError(f"Cannot normalize vector with invalid shape: {vector.shape}")

            # Verify dimensions match
            if vector.shape[-1] != self.dimension:
                raise VectorOperationError(
                    f"Vector dimension {vector.shape[-1]} does not match "
                    f"expected dimension {self.dimension}"
                )
            
            # Normalize for cosine similarity if needed
            if self.metric == DistanceMetric.COSINE:
                faiss.normalize_L2(vector)
            
            return vector
            
        except Exception as e:
            error_msg = f"Error normalizing vector: {str(e)}"
            self._logger.error(f"{error_msg}. Vector type: {type(vector)}, shape: {getattr(vector, 'shape', 'N/A')}")
            raise VectorOperationError(error_msg) from e
    
    def _denormalize_score(self, score: float) -> float:
        """Convert FAISS score to similarity score."""
        if self.metric == DistanceMetric.COSINE:
            # FAISS returns dot product for normalized vectors, which is cosine similarity
            return float((score + 1) / 2)  # Convert from [-1, 1] to [0, 1]
        elif self.metric == DistanceMetric.EUCLIDEAN:
            # Convert L2 distance to similarity (higher is more similar)
            return float(1 / (1 + score)) if score > 0 else 1.0
        return float(score)
    
    def _filter_records(
        self, 
        filter: Optional[MetadataFilter] = None
    ) -> List[VectorRecord]:
        """Filter records based on metadata."""
        if not filter:
            return list(self._id_to_record.values())
            
        results = []
        for record in self._id_to_record.values():
            if not record.metadata:
                continue
                
            match = True
            for key, value in filter.items():
                if key not in record.metadata or record.metadata[key] != value:
                    match = False
                    break
                    
            if match:
                results.append(record)
                
        return results
    
    async def _run_in_executor(self, func, *args):
        """Run a function in the thread pool executor."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, func, *args)
        
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
        try:
            async with self._lock:
                # Update index type and metric if provided
                if index_type:
                    self.index_type = index_type
                if metric:
                    self.metric = metric
                    
                # Update any additional parameters
                self._index_params.update(kwargs)
                
                # Reinitialize the index with new parameters
                self._init_index()
                
                # If we have existing vectors, we need to add them to the new index
                if self._id_to_record:
                    # Get all vectors and their IDs
                    ids = []
                    vectors = []
                    for id, record in self._id_to_record.items():
                        ids.append(int(id))
                        vectors.append(record.embedding)
                    
                    # Add vectors to the new index
                    if vectors:
                        vectors = np.array(vectors, dtype=np.float32)
                        if self.metric == DistanceMetric.COSINE:
                            faiss.normalize_L2(vectors)
                        self._index.add_with_ids(vectors, np.array(ids, dtype=np.int64))
                
                # Save to disk if persistence is enabled
                if self.persist_path:
                    await self._run_in_executor(self._save_to_disk)
                    
                return True
                
        except Exception as e:
            self._logger.error(f"Failed to create/update index: {str(e)}")
            raise VectorOperationError("Failed to create/update index") from e

    def _save_to_disk(self) -> None:
        """
        Save the index and metadata to disk atomically.
        
        This method will:
        1. Save the index and metadata to temporary files
        2. Atomically rename them to their final paths
        3. Handle any errors during the process and clean up temp files
        """
        if not self.persist_path or self._index is None:
            return
            
        temp_index_path = f"{self.persist_path}.index.tmp"
        final_index_path = f"{self.persist_path}.index"
        temp_meta_path = f"{self.persist_path}.meta.tmp"
        final_meta_path = f"{self.persist_path}.meta"
        
        try:
            # Save FAISS index
            faiss.write_index(self._index, temp_index_path)
            
            # Save metadata including ID mappings
            metadata = {
                'id_to_record': self._id_to_record,
                'string_id_to_int_id': self._string_id_to_int_id,
                'int_id_to_string_id': self._int_id_to_string_id,
                'next_int_id': self._next_int_id,
                'dimension': self.dimension,
                'metric': self.metric.value,
                'namespace': self.namespace,
                'index_type': self.index_type.value,
                'index_params': self._index_params,
                'is_trained': self._is_trained
            }
            
            # Write to temp file first
            with open(temp_meta_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            # Atomic rename to ensure consistency
            os.replace(temp_index_path, final_index_path)
            os.replace(temp_meta_path, final_meta_path)
                
        except Exception as e:
            self._logger.error(f"Failed to save index to {self.persist_path}: {str(e)}")
            # Clean up any temporary files
            for path in [temp_index_path, temp_meta_path]:
                try:
                    if path and os.path.exists(path):
                        os.remove(path)
                except Exception as cleanup_error:
                    self._logger.warning(f"Failed to clean up temp file {path}: {str(cleanup_error)}")
            raise VectorOperationError("Failed to save index to disk") from e
    
    def _load_from_disk(self) -> None:
        """
        Load the index and metadata from disk.
        
        This method will:
        1. Check if the index files exist
        2. Load the index and metadata atomically
        3. Handle any errors during loading
        4. Fall back to a new index if loading fails
        """
        if not self.persist_path:
            self._logger.debug("No persist_path set, nothing to load from disk")
            return
            
        index_path = f"{self.persist_path}.index"
        meta_path = f"{self.persist_path}.meta"
        
        # Check if both required files exist
        if not os.path.exists(index_path) or not os.path.exists(meta_path):
            self._logger.info(f"No existing index found at {self.persist_path}, starting fresh")
            return
            
        temp_index_path = f"{index_path}.tmp"
        temp_meta_path = f"{meta_path}.tmp"
        
        try:
            # Clean up any leftover temporary files
            for path in [temp_index_path, temp_meta_path]:
                if os.path.exists(path):
                    self._logger.debug(f"Removing leftover temp file: {path}")
                    try:
                        os.remove(path)
                    except Exception as e:
                        self._logger.warning(f"Failed to remove temp file {path}: {str(e)}")
            
            # Load FAISS index
            self._logger.debug(f"Loading FAISS index from {index_path}")
            self._index = faiss.read_index(index_path)
            
            # Load metadata
            self._logger.debug(f"Loading metadata from {meta_path}")
            with open(meta_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # Validate metadata
            required_keys = {'dimension', 'metric', 'index_type', 'next_id', 'records'}
            if not all(key in metadata for key in required_keys):
                raise VectorOperationError(f"Invalid metadata format in {meta_path}")

            # Check dimension compatibility
            if metadata['dimension'] != self.dimension:
                raise VectorOperationError(
                    f"Dimension mismatch: stored={metadata['dimension']}, "
                    f"current={self.dimension}"
                )
            
            # Restore state
            self._id_to_record = metadata.get('id_to_record', {})
            self._string_id_to_int_id = metadata.get('string_id_to_int_id', {})  # NEW
            self._int_id_to_string_id = metadata.get('int_id_to_string_id', {})  # NEW
            self._next_int_id = metadata.get('next_int_id', 0)  # NEW
            self._is_trained = True
            self._logger.info(
                f"Successfully loaded {len(self._id_to_record)} vectors from {self.persist_path}"
            )
            
        except Exception as e:
            self._logger.error(f"Failed to load index from {self.persist_path}: {str(e)}", exc_info=True)
            # Clean up any partial state
            self._index = None
            self._id_to_record = {}
            self._string_id_to_int_id = {}  # NEW
            self._int_id_to_string_id = {}  # NEW
            self._next_int_id = 0  # NEW
            self._is_trained = False
            
            # Reinitialize index
            self._init_index()
            raise VectorOperationError("Failed to load index from disk") from e
    
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
        # Convert inputs to lists for batch processing
        is_single = not isinstance(vectors, list)
        if is_single:
            vectors = [cast(Vector, vectors)]
            documents = [documents] if documents is not None else None
            metadatas = [metadatas] if metadatas is not None else None
            ids = [ids] if ids is not None else None
        
        # Validate inputs
        num_vectors = len(vectors)
        if documents is not None and len(documents) != num_vectors:
            raise VectorOperationError("Number of documents must match number of vectors")
        if metadatas is not None and len(metadatas) != num_vectors:
            raise VectorOperationError("Number of metadata dicts must match number of vectors")
        if ids is not None and len(ids) != num_vectors:
            raise VectorOperationError("Number of IDs must match number of vectors")
        
        # Generate string IDs if not provided
        if ids is None:
            ids = [str(uuid4()) for _ in range(num_vectors)]
        
        # Process vectors and create records
        records = []
        vectors_to_add = []
        
        for i, vector in enumerate(vectors):
            string_id = ids[i]
            document = documents[i] if documents else None
            metadata = metadatas[i] if metadatas else None
            
            # Create record
            record = VectorRecord(
                id=string_id,
                embedding=self._normalize_vector(vector)[0],
                document=document,
                metadata=metadata,
                namespace=self.namespace
            )
            records.append(record)
            vectors_to_add.append(record.embedding)
        
        # Add to index and store records
        async with self._lock:
            if self._index is None:
                raise VectorOperationError("FAISS index not initialized")
                
            # Convert to numpy array
            vectors_array = np.vstack(vectors_to_add).astype(np.float32)
            
            # Train index if needed
            if not self._is_trained and self.index_type != IndexType.FLAT:
                self._logger.info("Training FAISS index...")
                n_samples = min(10000, len(vectors_array) * 2)
                train_vectors = np.vstack([
                    np.random.randn(n_samples, self.dimension).astype(np.float32)
                    for _ in range(2)
                ])
                faiss.normalize_L2(train_vectors)
                self._index.train(train_vectors)
                self._is_trained = True
            
            # Add vectors to FAISS index (without IDs)
            try:
                # Ensure arrays are contiguous and have correct data types
                if not vectors_array.flags['C_CONTIGUOUS']:
                    vectors_array = np.ascontiguousarray(vectors_array, dtype=np.float32)
                
                # Ensure vectors are float32
                if vectors_array.dtype != np.float32:
                    vectors_array = vectors_array.astype(np.float32)
                
                # Use simple add method without IDs
                self._index.add(vectors_array)
                
            except Exception as e:
                self._logger.error(f"Error adding vectors to FAISS index: {str(e)}")
                self._logger.error(f"Vectors shape: {vectors_array.shape}")
                self._logger.error(f"Vectors dtype: {vectors_array.dtype}")
                self._logger.error(f"Index type: {type(self._index).__name__}")
                raise VectorOperationError("Failed to add vectors to FAISS index") from e
            
            # Update records with string IDs
            for record in records:
                self._id_to_record[record.id] = record
            
            # Save to disk if configured
            if self.persist_path:
                await self._run_in_executor(self._save_to_disk)
        
        return [record.id for record in records]
    
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
        if self._index is None or not self._id_to_record:
            return []
            
        # Process query vector
        query_vector_np = self._normalize_vector(query_vector)
        
        # Apply filter if provided
        if filter is not None:
            filtered_records = await self._run_in_executor(
                self._filter_records, filter
            )
            if not filtered_records:
                return []
                
            # NEW: Get filtered integer IDs for FAISS
            filtered_int_ids = []
            for record in filtered_records:
                if record.id in self._string_id_to_int_id:
                    filtered_int_ids.append(self._string_id_to_int_id[record.id])
            
            if not filtered_int_ids:
                return []
                
            # Create ID selector
            selector = faiss.IDSelectorBatch(
                len(filtered_int_ids),
                faiss.swig_ptr(np.array(filtered_int_ids, dtype=np.int64))
            )
            
            # Search with filter
            k = min(k, len(filtered_int_ids))
            distances, indices = self._index.search(
                query_vector_np, 
                k, 
                params=faiss.SearchParametersIVF(sel=selector) if hasattr(faiss, 'SearchParametersIVF') else None
            )
        else:
            # Regular search without filter
            k = min(k, len(self._id_to_record))
            distances, indices = self._index.search(query_vector_np, k)
        
        # Convert to results
        results: List[QueryResult] = []
        
        for i, (distance, int_idx) in enumerate(zip(distances[0], indices[0])):
            if int_idx < 0:  # Skip invalid indices
                continue
                
            # NEW: Convert integer ID back to string ID
            if int_idx in self._int_id_to_string_id:
                string_id = self._int_id_to_string_id[int_idx]
                record = self._id_to_record.get(string_id)
                
                if record is not None:
                    # Skip tombstoned vectors
                    if (record.metadata and 
                        record.metadata.get('tombstoned', False)):
                        continue
                    
                    result = QueryResult(
                        id=string_id,
                        score=self._denormalize_score(distance),
                        vector=record.embedding if include_vectors else None,
                        metadata=record.metadata if include_metadata else None,
                        document=record.document if include_metadata else None
                    )
                    results.append(result)
        
        return results
    
    async def get(
        self,
        ids: Optional[Union[str, List[str]]] = None,
        filter: Optional[MetadataFilter] = None,
        include_vectors: bool = False,
        **kwargs: Any
    ) -> List[VectorRecord]:
        """
        Get vectors by IDs or filter.
        
        Args:
            ids: Single ID or list of IDs to retrieve
            filter: Optional metadata filter to apply
            include_vectors: Whether to include vectors in the results
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            List of vector records matching the query
            
        Raises:
            VectorOperationError: If neither ids nor filter is provided
        """
        if ids is not None and filter is not None:
            raise VectorOperationError("Cannot specify both ids and filter")

        if ids is None and filter is None:
            raise VectorOperationError("Must provide either ids or filter")
            
        # Get by IDs
        if ids is not None:
            if isinstance(ids, str):
                ids = [ids]
                
            records = []
            for id in ids:
                if id in self._id_to_record:
                    record = self._id_to_record[id]
                    if not include_vectors:
                        record = record.copy(update={'embedding': None})
                    records.append(record)
            return records
            
        # Get by filter
        if filter is not None:
            records = await self._run_in_executor(self._filter_records, filter)
            if not include_vectors:
                records = [r.copy(update={'embedding': None}) for r in records]
            return records
            
        return []
    
    async def update(
        self,
        ids: Union[str, List[str]],
        vectors: Optional[Union[Vector, List[Vector]]] = None,
        documents: Optional[Union[str, List[str], None]] = None,
        metadatas: Optional[Union[Metadata, List[Metadata], None]] = None,
        **kwargs: Any
    ) -> None:
        """
        Update vectors in the store.
        
        Args:
            ids: Single ID or list of IDs to update
            vectors: Optional new vector(s) to update
            documents: Optional new document text(s) to update
            metadatas: Optional new metadata dict(s) to update
            **kwargs: Additional implementation-specific parameters
            
        Raises:
            ValueError: If inputs are invalid or IDs don't exist
        """
        # Convert inputs to lists
        if isinstance(ids, str):
            ids = [ids]
            
        if vectors is not None and not isinstance(vectors, list):
            vectors = [vectors]
            
        if documents is not None and not isinstance(documents, list):
            documents = [documents]
            
        if metadatas is not None and not isinstance(metadatas, list):
            metadatas = [metadatas]
            
        # Validate inputs
        num_updates = len(ids)
        if vectors is not None and len(vectors) != num_updates:
            raise VectorOperationError("Number of vectors must match number of IDs")
        if documents is not None and len(documents) != num_updates:
            raise VectorOperationError("Number of documents must match number of IDs")
        if metadatas is not None and len(metadatas) != num_updates:
            raise VectorOperationError("Number of metadata dicts must match number of IDs")
            
        async with self._lock:
            if self._index is None:
                raise VectorOperationError("FAISS index not initialized")
                
            # Process updates
            for i, id in enumerate(ids):
                if id not in self._id_to_record:
                    raise VectorOperationError(f"ID not found: {id}")
                    
                record = self._id_to_record[id]
                
                # Update vector if provided
                if vectors is not None:
                    new_vector = self._normalize_vector(vectors[i])[0]
                    record.embedding = new_vector
                    
                    # Update in FAISS index
                    int_id = self._string_id_to_int_id[id] # Get integer ID
                    self._index.remove_ids(np.array([int_id], dtype=np.int64))
                    self._index.add_with_ids(
                        new_vector.reshape(1, -1),
                        np.array([int_id], dtype=np.int64)
                    )
                
                # Update document if provided
                if documents is not None:
                    record.document = documents[i]
                    
                # Update metadata if provided
                if metadatas is not None:
                    if metadatas[i] is None:
                        record.metadata = {}
                    else:
                        if record.metadata is None:
                            record.metadata = {}
                        record.metadata.update(metadatas[i])
            
            # Save to disk if configured
            if self.persist_path:
                await self._run_in_executor(self._save_to_disk)
    
    async def delete(
        self,
        ids: Optional[Union[str, List[str]]] = None,
        filter: Optional[MetadataFilter] = None,
        documents: Optional[Union[str, List[str]]] = None,
        **kwargs: Any
    ) -> None:
        """
        Delete vectors from the store.
        
        Args:
            ids: Single ID or list of IDs to delete
            filter: Optional metadata filter to select vectors to delete
            documents: Single document text or list of document texts to delete
            **kwargs: Additional implementation-specific parameters
            
        Raises:
            ValueError: If neither ids, filter, nor documents is provided, or if IDs don't exist
        """
        # Validate that only one deletion method is specified
        specified_methods = sum([
            ids is not None,
            filter is not None,
            documents is not None
        ])
        
        if specified_methods == 0:
            raise VectorOperationError("Must provide either ids, filter, or documents")
        elif specified_methods > 1:
            raise VectorOperationError("Cannot specify multiple deletion methods (ids, filter, documents)")
            
        # Convert to list of IDs
        ids_to_delete: List[str] = []
        
        if ids is not None:
            if isinstance(ids, str):
                ids_to_delete = [ids]
            else:
                ids_to_delete = ids.copy()
                
            # Verify all IDs exist
            for id in ids_to_delete:
                if id not in self._id_to_record:
                    raise VectorOperationError(f"ID not found: {id}")
                    
        elif filter is not None:
            # Get IDs from filter
            filtered_records = await self._run_in_executor(self._filter_records, filter)
            ids_to_delete = [r.id for r in filtered_records]
            
        elif documents is not None:
            # Find IDs by document text
            if isinstance(documents, str):
                documents = [documents]
            
            for doc in documents:
                for record_id, record in self._id_to_record.items():
                    if record.document == doc:
                        ids_to_delete.append(record_id)
            
        if not ids_to_delete:
            return
            
        # Perform deletion
        async with self._lock:
            if self._index is None:
                raise VectorOperationError("FAISS index not initialized")
                
            # NEW: Convert string IDs to integer IDs for FAISS deletion
            int_ids_to_delete = []
            for string_id in ids_to_delete:
                if string_id in self._string_id_to_int_id:
                    int_ids_to_delete.append(self._string_id_to_int_id[string_id])
            
            # Remove from FAISS index
            self._index.remove_ids(np.array(int_ids_to_delete, dtype=np.int64))
            
            # Remove from records
            for id in ids_to_delete:
                self._id_to_record.pop(id, None)
                # Also remove from ID mappings if they were deleted
                if id in self._string_id_to_int_id:
                    int_id = self._string_id_to_int_id[id]
                    del self._string_id_to_int_id[id]
                    del self._int_id_to_string_id[int_id]
            
            # Save to disk if configured
            if self.persist_path:
                await self._run_in_executor(self._save_to_disk)
    
                # Save to disk if configured
            if self.persist_path:
                await self._run_in_executor(self._save_to_disk)

    async def tombstone(
        self,
        ids: Optional[Union[str, List[str]]] = None,
        filter: Optional[MetadataFilter] = None,
        documents: Optional[Union[str, List[str]]] = None,
        **kwargs: Any
    ) -> bool:
        """
        Mark vectors as tombstoned (deleted) without immediately removing them.
        
        Args:
            ids: Single ID or list of IDs to tombstone
            filter: Optional metadata filter for bulk tombstoning
            documents: Document text(s) to tombstone
            **kwargs: Additional implementation-specific parameters
            
        Returns:
            True if tombstoning was successful
        """
        # Validate that only one tombstoning method is specified
        specified_methods = sum([
            ids is not None,
            filter is not None,
            documents is not None
        ])
        
        if specified_methods == 0:
            return False
        elif specified_methods > 1:
            return False
            
        # Convert to list of IDs
        ids_to_tombstone: List[str] = []
        
        if ids is not None:
            if isinstance(ids, str):
                ids_to_tombstone = [ids]
            else:
                ids_to_tombstone = ids.copy()
                
        elif filter is not None:
            # Get IDs from filter
            filtered_records = await self._run_in_executor(self._filter_records, filter)
            ids_to_tombstone = [r.id for r in filtered_records]
            
        elif documents is not None:
            # Find IDs by document text
            if isinstance(documents, str):
                documents = [documents]
            
            for doc in documents:
                for record_id, record in self._id_to_record.items():
                    if record.document == doc:
                        ids_to_tombstone.append(record_id)
        
        if not ids_to_tombstone:
            return False
            
        # Mark as tombstoned
        async with self._lock:
            for id in ids_to_tombstone:
                if id in self._id_to_record:
                    record = self._id_to_record[id]
                    # Add tombstone flag to metadata
                    if record.metadata is None:
                        record.metadata = {}
                    record.metadata['tombstoned'] = True
                    record.metadata['tombstoned_at'] = datetime.now().isoformat()
            
            # Save to disk if configured
            if self.persist_path:
                await self._run_in_executor(self._save_to_disk)
        
        return True
    
    async def purge_tombstones(self) -> int:
        """
        Permanently remove all tombstoned vectors from storage.
        
        Returns:
            int: Number of tombstoned vectors that were purged
        """
        try:
            async with self._lock:
                # Find all tombstoned records
                tombstoned_ids = []
                for record_id, record in self._id_to_record.items():
                    if (record.metadata and 
                        record.metadata.get('tombstoned', False)):
                        tombstoned_ids.append(record_id)
                
                if not tombstoned_ids:
                    return 0
                
                # Delete them permanently using the existing delete method
                await self.delete(ids=tombstoned_ids)
                return len(tombstoned_ids)
                
        except Exception as e:
            self._logger.error(f"Error purging tombstones: {e}")
            return 0
    
    async def count(self, filter: Optional[MetadataFilter] = None) -> int:
        """
        Count the number of vectors in the store.
        
        Args:
            filter: Optional metadata filter to apply
            
        Returns:
            Number of vectors matching the filter (or total if no filter)
        """
        if filter is None:
            return len(self._id_to_record)
            
        filtered_records = await self._run_in_executor(self._filter_records, filter)
        return len(filtered_records)
    
    async def reset(self) -> None:
        """
        Reset the store, removing all vectors.
        """
        async with self._lock:
            self._init_index()
            self._id_to_record = {}
            self._string_id_to_int_id = {}  # NEW
            self._int_id_to_string_id = {}  # NEW
            self._next_int_id = 0  # NEW
            
            if self.persist_path and os.path.exists(f"{self.persist_path}.index"):
                try:
                    os.remove(f"{self.persist_path}.index")
                    if os.path.exists(f"{self.persist_path}.meta"):
                        os.remove(f"{self.persist_path}.meta")
                except Exception as e:
                    self._logger.error(f"Failed to delete index files: {str(e)}")
    
    async def get_index_info(self) -> Dict[str, Any]:
        """
        Get information about the index.
        
        Returns:
            Dictionary containing index information
        """
        if self._index is None:
            return {}
            
        return {
            'dimension': self.dimension,
            'metric': self.metric.value,
            'index_type': self.index_type.value,
            'is_trained': self._is_trained,
            'size': len(self._id_to_record),
            'index_params': self._index_params
        }
    
    async def persist(self, path: Optional[str] = None) -> None:
        """
        Persist the index to disk.
        
        Args:
            path: Optional path to save the index. If not provided, uses the path
                  from initialization or does nothing if no path was provided.
        """
        if path is not None:
            self.persist_path = path
            
        if self.persist_path:
            await self._run_in_executor(self._save_to_disk)
    
    async def close(self) -> None:
        """
        Close the store and release resources.
        """
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
            
        if self.persist_path:
            await self._run_in_executor(self._save_to_disk)
    
    async def clear(self) -> None:
        """
        Clear all stored embeddings.
        
        This resets the FAISS index and clears all stored responses.
        """
        async with self._lock:
            self._init_index()
            self._id_to_record = {}
            self._string_id_to_int_id = {}  # NEW
            self._int_id_to_string_id = {}  # NEW
            self._next_int_id = 0  # NEW
            
            if self.persist_path and os.path.exists(f"{self.persist_path}.index"):
                # Delete persisted files
                try:
                    os.remove(f"{self.persist_path}.index")
                    os.remove(f"{self.persist_path}.meta")
                except OSError:
                    pass  # Files might not exist
    
    async def size(self) -> int:
        """Get the number of stored embeddings."""
        return len(self._id_to_record)
    
    async def list_collections(self) -> List[str]:
        """
        List all collections.
        
        Note: FAISS doesn't have a native concept of collections, so this returns
        a list with the current namespace if one is set, or an empty list.
        
        Returns:
            List of collection names
        """
        if self.namespace:
            return [self.namespace]
        return []
    
    async def delete_collection(self, name: str, **kwargs: Any) -> bool:
        """
        Delete a collection.
        
        Note: FAISS doesn't have a native concept of collections. This method
        will clear the entire store if the name matches the current namespace.
        
        Args:
            name: Name of the collection to delete
            **kwargs: Additional arguments (ignored)
            
        Returns:
            True if the collection was deleted, False otherwise
        """
        if name == self.namespace or (not self.namespace and not name):
            await self.clear()
            return True
        return False
        
    async def __aenter__(self):
        """Async context manager entry."""
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup resources."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
