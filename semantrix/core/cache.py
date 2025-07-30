import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from semantrix.embedding import BaseEmbedder, get_embedder
from semantrix.vector_store.vector_store import BaseVectorStore, FAISSVectorStore
from semantrix.cache_store import BaseCacheStore, InMemoryStore
from semantrix.utils.resource_limits import ResourceLimits
from semantrix.utils.profiling import Profiler
from semantrix.models.explain import ExplainResult, CacheMatch, create_explain_result

# Set up logging
logger = logging.getLogger(__name__)

class Semantrix:
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        resource_limits: ResourceLimits = ResourceLimits(),
        enable_profiling: bool = False,
        embedder: Optional[BaseEmbedder] = None,
        vector_store: Optional[BaseVectorStore] = None,
        cache_store: Optional[BaseCacheStore] = None
    ):
        """
        Initialize Semantrix semantic cache.
        
        Args:
            similarity_threshold: Cosine similarity threshold for semantic match
            resource_limits: Resource limits configuration
            enable_profiling: Enable lightweight profiling
            embedder: Custom embedder implementation (defaults to sentence-transformers)
            vector_store: Custom vector store implementation (defaults to FAISS)
            cache_store: Custom cache store implementation (defaults to InMemoryStore)
        """
        # Initialize components with defaults if not provided
        self.embedder = embedder or get_embedder("sentence-transformers")
        self.vector_store = vector_store or FAISSVectorStore(dimension=self.embedder.get_dimension())
        self.cache_store = cache_store or InMemoryStore()
        
        self.resource_limits = resource_limits
        self.profiler = Profiler(enabled=enable_profiling)
        self.similarity_threshold = similarity_threshold

    async def get(self, prompt: str) -> Optional[str]:
        """
        Get a cached response for the given prompt asynchronously.
        
        Args:
            prompt: The prompt to look up in the cache.
            
        Returns:
            The cached response if found, None otherwise.
            
        Raises:
            RuntimeError: If there's an error during the embedding process
        """
        with self.profiler.record("get"):
            # 1. Check resource constraints
            if not self.resource_limits.allow_operation():
                return None
                
            # 2. Exact match
            exact_match = await self.cache_store.get_exact(prompt)
            if exact_match is not None:
                return exact_match
                
            # 3. Semantic search (if vector store is available)
            if hasattr(self.embedder, 'aencode') and hasattr(self.vector_store, 'search'):
                try:
                    # Use async embedding
                    embedding = await self.embedder.aencode(prompt)
                    
                    # Search in vector store
                    match = await self.vector_store.search(embedding, self.similarity_threshold)
                    if isinstance(match, str):
                        return match
                except Exception as e:
                    logger.error(f"Error during semantic search: {e}")
                    # Continue to return None if there's an error
                    
            return None

    async def set(self, prompt: str, response: str) -> None:
        """
        Add a prompt-response pair to the cache asynchronously.
        
        Args:
            prompt: The prompt to cache.
            response: The response to cache.
            
        Raises:
            RuntimeError: If there's an error during the embedding process
        """
        with self.profiler.record("set"):
            if not self.resource_limits.allow_operation():
                return
                
            # 1. Add to cache
            if hasattr(self.embedder, 'aencode') and hasattr(self.vector_store, 'add'):
                try:
                    # Use async embedding
                    embedding = await self.embedder.aencode(prompt)
                    
                    # Add to vector store
                    await self.vector_store.add(embedding, response)
                except Exception as e:
                    logger.error(f"Error during embedding or vector store add: {e}")
                    # Continue to cache the exact match even if embedding fails
            
            # Always cache the exact match
            await self.cache_store.add(prompt, response)
            
            # 2. Enforce limits
            try:
                await self.cache_store.enforce_limits(self.resource_limits)
            except Exception as e:
                logger.error(f"Error enforcing cache limits: {e}")
            
    async def explain(self, prompt: str) -> ExplainResult:
        """
        Explain why a prompt missed or hit the cache asynchronously.
        
        Args:
            prompt: The prompt to explain
            
        Returns:
            ExplainResult with detailed information about the cache lookup
            
        Raises:
            RuntimeError: If there's an error during the explanation process
        """
        start_time = time.time()
        
        # Check resource constraints
        resource_limited = not self.resource_limits.allow_operation()
        resource_warnings = []
        
        if resource_limited:
            resource_warnings.append("Resource limits exceeded")
        
        # Check exact match
        exact_match = await self.cache_store.get_exact(prompt)
        exact_match_found = exact_match is not None
        
        # Initialize variables for semantic search
        embedding = None
        matches = []
        embedding_time = 0
        search_time = 0
        
        # Only perform semantic search if needed and possible
        if not exact_match_found and hasattr(self.embedder, 'aencode') and hasattr(self.vector_store, 'search'):
            try:
                # Get embedding asynchronously
                embedding_start = time.time()
                embedding = await self.embedder.aencode(prompt)
                embedding_time = (time.time() - embedding_start) * 1000  # Convert to ms
                
                # Search for similar vectors
                search_start = time.time()
                matches = await self.vector_store.search(embedding, top_k=5)
                search_time = (time.time() - search_start) * 1000  # Convert to ms
                
                # Ensure matches is a list of (text, similarity) tuples
                if not isinstance(matches, list):
                    matches = []
            except Exception as e:
                logger.error(f"Error during explain embedding/search: {e}")
                matches = []
        
        total_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Convert matches to CacheMatch objects
        top_matches = []
        if matches and isinstance(matches, list):
            for text, similarity in matches:
                if isinstance(text, str) and isinstance(similarity, (int, float)):
                    top_matches.append(CacheMatch(text=text, similarity=float(similarity)))
        
        # Determine cache status
        best_similarity = max([match.similarity for match in top_matches]) if top_matches else 0.0
        cache_hit = exact_match_found or best_similarity >= self.similarity_threshold
        semantic_match = not exact_match_found and cache_hit
        
        return create_explain_result(
            query=prompt,
            similarity_threshold=self.similarity_threshold,
            top_matches=top_matches,
            cache_hit=cache_hit,
            exact_match=exact_match_found,
            semantic_match=semantic_match,
            resource_limited=resource_limited,
            resource_warnings=resource_warnings,
            embedding_time_ms=embedding_time,
            search_time_ms=search_time,
            total_time_ms=total_time
        ) 