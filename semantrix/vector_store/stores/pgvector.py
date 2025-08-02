"""pgvector vector store implementation.

Note: This module requires the 'psycopg2-binary' and 'numpy' packages to be installed.
Install them with: pip install psycopg2-binary numpy
"""

import json
import logging
import uuid
from typing import Any, Dict, List, Optional, Sequence, Union, cast

# Optional import - will raise ImportError if not available
try:
    import psycopg2
    from psycopg2 import sql
    from psycopg2.extras import execute_values
    import psycopg2.extensions
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False

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

# Type alias for PostgreSQL connection
PGConnection = Any

class PgVectorStore(BaseVectorStore):
    """PostgreSQL with pgvector vector store implementation."""
    
    def __init__(
        self,
        dimension: int,
        metric: DistanceMetric = DistanceMetric.COSINE,
        namespace: Optional[str] = None,
        connection_string: Optional[str] = None,
        table_name: str = "semantrix_vectors",
        **kwargs: Any
    ) -> None:
        """Initialize the pgvector vector store.
        
        Args:
            dimension: The dimension of the vectors to be stored
            metric: Distance metric to use for similarity search
            namespace: Optional namespace for multi-tenancy
            connection_string: PostgreSQL connection string
            table_name: Name of the table to store vectors
            **kwargs: Additional arguments for psycopg2.connect()
        """
        super().__init__(dimension=dimension, metric=metric, namespace=namespace)
        self._logger = logging.getLogger(__name__)
        
        if not PGVECTOR_AVAILABLE:
            raise ImportError(
                "The 'psycopg2-binary' package is required for PgVectorStore. "
                "Please install it with: pip install psycopg2-binary numpy"
            )
        
        self.connection_string = connection_string or ""
        self.table_name = table_name
        self._connection: Optional[PGConnection] = None
        self._connection_kwargs = kwargs
        
        # Initialize connection and ensure table exists
        self._init_connection()
        self._ensure_table()
    
    def _init_connection(self) -> None:
        """Initialize the PostgreSQL connection."""
        try:
            self._connection = psycopg2.connect(
                self.connection_string,
                **self._connection_kwargs
            )
            # Enable pgvector extension
            with self._connection.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                self._connection.commit()
        except Exception as e:
            self._logger.error(f"Failed to connect to PostgreSQL: {str(e)}")
            if self._connection:
                self._connection.close()
                self._connection = None
            raise
    
    def _ensure_table(self) -> None:
        """Ensure the vectors table exists."""
        if not self._connection:
            raise RuntimeError("Database connection not initialized")
        
        try:
            with self._connection.cursor() as cur:
                # Create the table if it doesn't exist
                cur.execute(
                    sql.SQL("""
                    CREATE TABLE IF NOT EXISTS {} (
                        id TEXT PRIMARY KEY,
                        vector VECTOR({}),
                        document TEXT,
                        metadata JSONB,
                        namespace TEXT,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    )
                    """).format(
                        sql.Identifier(self.table_name),
                        sql.SQL(str(self.dimension))
                    )
                )
                
                # Create index on the vector column for faster search
                index_name = f"{self.table_name}_vector_idx"
                opclass = "vector_cosine_ops" if self.metric == DistanceMetric.COSINE else "vector_l2_ops"
                
                cur.execute(
                    sql.SQL("""
                    CREATE INDEX IF NOT EXISTS {} ON {} 
                    USING hnsw (vector {})
                    """).format(
                        sql.Identifier(index_name),
                        sql.Identifier(self.table_name),
                        sql.SQL(opclass)
                    )
                )
                
                self._connection.commit()
                self._logger.info(f"Ensured table {self.table_name} exists with vector index")
                
        except Exception as e:
            self._connection.rollback()
            self._logger.error(f"Failed to ensure table exists: {str(e)}")
            raise
    
    @classmethod
    def from_connection_params(
        cls,
        dimension: int,
        dbname: str,
        user: str,
        password: str,
        host: str = "localhost",
        port: int = 5432,
        **kwargs: Any
    ) -> 'PgVectorStore':
        """Create a PgVectorStore from connection parameters.
        
        Args:
            dimension: The dimension of the vectors
            dbname: Database name
            user: Database user
            password: Database password
            host: Database host (default: localhost)
            port: Database port (default: 5432)
            **kwargs: Additional arguments for PgVectorStore
            
        Returns:
            Configured PgVectorStore instance
            
        Example:
            ```python
            store = PgVectorStore.from_connection_params(
                dimension=768,
                dbname="vector_db",
                user="postgres",
                password="password",
                host="localhost"
            )
            ```
        """
        connection_string = f"dbname={dbname} user={user} password={password} host={host} port={port}"
        return cls(dimension=dimension, connection_string=connection_string, **kwargs)
    
    async def add(
        self,
        vectors: Union[Vector, List[Vector]],
        documents: Optional[Union[str, List[str]]] = None,
        metadatas: Optional[Union[Metadata, List[Metadata]]] = None,
        ids: Optional[Union[str, List[str]]] = None,
        **kwargs: Any
    ) -> List[str]:
        """Add vectors to the store."""
        if not self._connection:
            raise RuntimeError("Database connection not initialized")
            
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
        
        # Prepare data for batch insert
        data = []
        for i, (vec, vec_id, meta) in enumerate(zip(vectors, ids, metadatas)):
            # Convert vector to list if it's a numpy array
            vector_list = vec.tolist() if hasattr(vec, 'tolist') else list(vec)
            
            # Add document to metadata if provided
            metadata = dict(meta or {})
            if documents is not None and i < len(documents) and documents[i] is not None:
                metadata["document"] = documents[i]
            
            # Add namespace to metadata if specified
            if self.namespace:
                metadata["namespace"] = self.namespace
            
            data.append((
                vec_id,
                vector_list,
                documents[i] if documents and i < len(documents) else None,
                json.dumps(metadata),
                self.namespace or ""
            ))
        
        # Perform batch insert
        try:
            with self._connection.cursor() as cur:
                execute_values(
                    cur,
                    sql.SQL("""
                    INSERT INTO {} (id, vector, document, metadata, namespace)
                    VALUES %s
                    ON CONFLICT (id) DO UPDATE
                    SET vector = EXCLUDED.vector,
                        document = EXCLUDED.document,
                        metadata = EXCLUDED.metadata,
                        namespace = EXCLUDED.namespace
                    """).format(sql.Identifier(self.table_name)),
                    data,
                    template="(%s, %s::vector, %s, %s::jsonb, %s)"
                )
                self._connection.commit()
                return ids
                
        except Exception as e:
            self._connection.rollback()
            self._logger.error(f"Failed to add vectors to pgvector: {str(e)}")
            raise
    
    async def get(
        self,
        ids: Optional[Union[str, List[str]]] = None,
        filter: Optional[MetadataFilter] = None,
        include_vectors: bool = True,
        **kwargs: Any
    ) -> List[VectorRecord]:
        """Get vectors from the store."""
        if not self._connection:
            raise RuntimeError("Database connection not initialized")
            
        # Convert single ID to list
        if ids is not None and not isinstance(ids, list):
            ids = [ids]
        
        # Build WHERE conditions
        conditions = []
        params: List[Any] = []
        
        if ids:
            placeholders = ", ".join(["%s"] * len(ids))
            conditions.append(f"id IN ({placeholders})")
            params.extend(ids)
        
        if self.namespace:
            conditions.append("namespace = %s")
            params.append(self.namespace)
        
        # Add metadata filters
        if filter:
            for key, value in filter.items():
                if isinstance(value, dict):
                    # Handle operators like $eq, $gt, etc.
                    for op, op_value in value.items():
                        if op == "$eq":
                            conditions.append(f"metadata->>%s = %s")
                            params.extend([key, str(op_value)])
                        elif op == "$ne":
                            conditions.append(f"metadata->>%s != %s")
                            params.extend([key, str(op_value)])
                        elif op in ["$gt", "$gte", "$lt", "$lte"]:
                            conditions.append(f"(metadata->>%s)::numeric {op} %s")
                            params.extend([key, str(op_value)])
                        elif op == "$in":
                            if not isinstance(op_value, (list, tuple)):
                                raise ValueError("$in operator requires a list value")
                            placeholders = ", ".join(["%s"] * len(op_value))
                            conditions.append(f"metadata->>%s IN ({placeholders})")
                            params.extend([key] + [str(v) for v in op_value])
                else:
                    # Default to equality check
                    conditions.append("metadata->>%s = %s")
                    params.extend([key, str(value)])
        
        # Build the query
        select_fields = "id, document, metadata, vector::text" if include_vectors else "id, document, metadata, NULL as vector"
        query = f"SELECT {select_fields} FROM {self.table_name}"
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        try:
            with self._connection.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
                
                results = []
                for row in rows:
                    vec_id, doc, meta, vec_str = row
                    
                    # Parse vector if included
                    vector = None
                    if include_vectors and vec_str:
                        # Convert from PostgreSQL vector format: (0.1, 0.2, 0.3)
                        vector = [float(x) for x in vec_str.strip("()").split(",")]
                    
                    # Parse metadata
                    metadata = json.loads(meta) if meta else {}
                    
                    # Create vector record
                    record = VectorRecord(
                        id=vec_id,
                        document=doc or "",
                        metadata=metadata,
                        embedding=vector
                    )
                    results.append(record)
                
                return results
                
        except Exception as e:
            self._logger.error(f"Failed to get vectors from pgvector: {str(e)}")
            raise
    
    async def search(
        self,
        query_vector: Vector,
        k: int = 10,
        filter: Optional[MetadataFilter] = None,
        **kwargs: Any
    ) -> List[QueryResult]:
        """Search for similar vectors."""
        if not self._connection:
            raise RuntimeError("Database connection not initialized")
            
        # Convert query vector to list if it's a numpy array
        query_vec = query_vector.tolist() if hasattr(query_vector, 'tolist') else list(query_vector)
        
        # Build WHERE conditions
        conditions = []
        params: List[Any] = [query_vec, k]
        
        if self.namespace:
            conditions.append("namespace = %s")
            params.append(self.namespace)
        
        # Add metadata filters
        if filter:
            for key, value in filter.items():
                if isinstance(value, dict):
                    # Handle operators
                    for op, op_value in value.items():
                        if op == "$eq":
                            conditions.append(f"metadata->>%s = %s")
                            params.extend([key, str(op_value)])
                        elif op == "$ne":
                            conditions.append(f"metadata->>%s != %s")
                            params.extend([key, str(op_value)])
                        elif op in ["$gt", "$gte", "$lt", "$lte"]:
                            conditions.append(f"(metadata->>%s)::numeric {op} %s")
                            params.extend([key, str(op_value)])
                        elif op == "$in":
                            if not isinstance(op_value, (list, tuple)):
                                raise ValueError("$in operator requires a list value")
                            placeholders = ", ".join(["%s"] * len(op_value))
                            conditions.append(f"metadata->>%s IN ({placeholders})")
                            params.extend([key] + [str(v) for v in op_value])
                else:
                    # Default to equality check
                    conditions.append("metadata->>%s = %s")
                    params.extend([key, str(value)])
        
        # Build the query
        metric_op = "<=>" if self.metric == DistanceMetric.COSINE else "<->"
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        
        query = f"""
        SELECT id, document, metadata, vector::text, 
               vector {metric_op} %s::vector AS distance
        FROM {self.table_name}
        WHERE {where_clause}
        ORDER BY vector {metric_op} %s::vector
        LIMIT %s
        """
        
        try:
            with self._connection.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
                
                results = []
                for row in rows:
                    vec_id, doc, meta, vec_str, distance = row
                    
                    # Parse metadata
                    metadata = json.loads(meta) if meta else {}
                    
                    # For cosine similarity, convert from distance to similarity
                    if self.metric == DistanceMetric.COSINE:
                        score = 1.0 - distance
                    else:
                        score = float(distance)
                    
                    results.append({
                        "id": vec_id,
                        "document": doc or "",
                        "metadata": metadata,
                        "score": score,
                        "vector": [float(x) for x in vec_str.strip("()").split(",")] if vec_str else None
                    })
                
                return results
                
        except Exception as e:
            self._logger.error(f"Failed to search vectors in pgvector: {str(e)}")
            raise
    
    async def update(
        self,
        ids: Union[str, List[str]],
        vectors: Optional[Union[Vector, List[Vector]]] = None,
        documents: Optional[Union[str, List[str]]] = None,
        metadatas: Optional[Union[Metadata, List[Metadata]]] = None,
        **kwargs: Any
    ) -> None:
        """Update vectors in the store."""
        if not self._connection:
            raise RuntimeError("Database connection not initialized")
            
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
            updates.append((
                vec_id,
                record.embedding,
                record.document,
                json.dumps(record.metadata)
            ))
        
        # Perform batch update
        if updates:
            try:
                with self._connection.cursor() as cur:
                    execute_values(
                        cur,
                        sql.SQL("""
                        UPDATE {} SET
                            vector = data.vector::vector,
                            document = data.document,
                            metadata = data.metadata::jsonb
                        FROM (VALUES %s) AS data (id, vector, document, metadata)
                        WHERE {}.id = data.id
                        """).format(
                            sql.Identifier(self.table_name),
                            sql.Identifier(self.table_name)
                        ),
                        updates,
                        template="(%s, %s::vector, %s, %s::jsonb)"
                    )
                    self._connection.commit()
                    
            except Exception as e:
                self._connection.rollback()
                self._logger.error(f"Failed to update vectors in pgvector: {str(e)}")
                raise
    
    async def delete(
        self,
        ids: Optional[Union[str, List[str]]] = None,
        filter: Optional[MetadataFilter] = None,
        **kwargs: Any
    ) -> None:
        """Delete vectors from the store."""
        if not self._connection:
            raise RuntimeError("Database connection not initialized")
            
        # If neither IDs nor filter provided, delete all vectors
        if ids is None and filter is None:
            try:
                with self._connection.cursor() as cur:
                    if self.namespace:
                        cur.execute(
                            sql.SQL("DELETE FROM {} WHERE namespace = %s").format(
                                sql.Identifier(self.table_name)
                            ),
                            (self.namespace,)
                        )
                    else:
                        cur.execute(
                            sql.SQL("TRUNCATE TABLE {}").format(
                                sql.Identifier(self.table_name)
                            )
                        )
                    self._connection.commit()
                    return
                    
            except Exception as e:
                self._connection.rollback()
                self._logger.error(f"Failed to delete all vectors from pgvector: {str(e)}")
                raise
        
        # Build WHERE conditions for targeted delete
        conditions = []
        params: List[Any] = []
        
        if ids is not None:
            if not isinstance(ids, list):
                ids = [ids]
            placeholders = ", ".join(["%s"] * len(ids))
            conditions.append(f"id IN ({placeholders})")
            params.extend(ids)
        
        if self.namespace:
            conditions.append("namespace = %s")
            params.append(self.namespace)
        
        # Add metadata filters
        if filter:
            for key, value in filter.items():
                if isinstance(value, dict):
                    # Handle operators
                    for op, op_value in value.items():
                        if op == "$eq":
                            conditions.append(f"metadata->>%s = %s")
                            params.extend([key, str(op_value)])
                        elif op == "$ne":
                            conditions.append(f"metadata->>%s != %s")
                            params.extend([key, str(op_value)])
                        elif op in ["$gt", "$gte", "$lt", "$lte"]:
                            conditions.append(f"(metadata->>%s)::numeric {op} %s")
                            params.extend([key, str(op_value)])
                        elif op == "$in":
                            if not isinstance(op_value, (list, tuple)):
                                raise ValueError("$in operator requires a list value")
                            placeholders = ", ".join(["%s"] * len(op_value))
                            conditions.append(f"metadata->>%s IN ({placeholders})")
                            params.extend([key] + [str(v) for v in op_value])
                else:
                    # Default to equality check
                    conditions.append("metadata->>%s = %s")
                    params.extend([key, str(value)])
        
        # Build and execute the delete query
        if conditions:
            try:
                with self._connection.cursor() as cur:
                    query = sql.SQL("DELETE FROM {} WHERE {}").format(
                        sql.Identifier(self.table_name),
                        sql.SQL(" AND ").join([sql.SQL(cond) for cond in conditions])
                    )
                    cur.execute(query, params)
                    self._connection.commit()
                    
            except Exception as e:
                self._connection.rollback()
                self._logger.error(f"Failed to delete vectors from pgvector: {str(e)}")
                raise
    
    async def count(self, **kwargs: Any) -> int:
        """Count the number of vectors in the store."""
        if not self._connection:
            raise RuntimeError("Database connection not initialized")
            
        try:
            with self._connection.cursor() as cur:
                query = f"SELECT COUNT(*) FROM {self.table_name}"
                params = []
                
                if self.namespace:
                    query += " WHERE namespace = %s"
                    params.append(self.namespace)
                
                cur.execute(query, params)
                return cur.fetchone()[0]
                
        except Exception as e:
            self._logger.error(f"Failed to count vectors in pgvector: {str(e)}")
            raise
    
    async def reset(self, **kwargs: Any) -> None:
        """Reset the vector store by dropping and recreating the table."""
        if not self._connection:
            raise RuntimeError("Database connection not initialized")
            
        try:
            with self._connection.cursor() as cur:
                # Drop the table
                cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {}").format(
                        sql.Identifier(self.table_name)
                    )
                )
                
                # Recreate the table
                self._ensure_table()
                self._connection.commit()
                
        except Exception as e:
            self._connection.rollback()
            self._logger.error(f"Failed to reset pgvector store: {str(e)}")
            raise
    
    async def close(self, **kwargs: Any) -> None:
        """Close the database connection."""
        if self._connection:
            self._connection.close()
            self._connection = None
