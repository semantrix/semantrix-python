"""
Semantrix Client
----------------
User-facing entry point for initializing and using Semantrix as a library.

Example:
    from semantrix.client import SemantrixClient
    cache = SemantrixClient(similarity_threshold=0.9, max_memory_gb=0.5)
    cache.set("What is AI?", "Artificial Intelligence is ...")
    print(cache.get("Explain artificial intelligence"))
"""
from .core.cache import Semantrix
from .utils.resource_limits import ResourceLimits
from .utils.resource_limits import DEFAULT_MAX_MEMORY_GB, DEFAULT_MAX_CPU_PERCENT, DEFAULT_MAX_MEMORY_PERCENT
from .embedding.base import BaseEmbedder
from .vector_store.base import BaseVectorStore
from .cache_store.base import BaseCacheStore
from .models.explain import ExplainResult
from typing import Optional

class SemantrixClient:
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_memory_gb: Optional[float] = None,
        max_cpu_percent: float = DEFAULT_MAX_CPU_PERCENT,
        max_memory_percent: Optional[float] = None,
        enable_profiling: bool = False,
        embedder: Optional[BaseEmbedder] = None,
        vector_store: Optional[BaseVectorStore] = None,
        cache_store: Optional[BaseCacheStore] = None
    ):
        """
        Initialize a Semantrix semantic cache instance.

        Args:
            similarity_threshold (float): Cosine similarity threshold for semantic match.
            max_memory_gb (float, optional): Maximum memory usage in GB. 
                Specify either this OR max_memory_percent, not both.
            max_cpu_percent (float): Maximum CPU usage percent.
            max_memory_percent (float, optional): Maximum memory usage as percentage of total system memory.
                Specify either this OR max_memory_gb, not both.
            enable_profiling (bool): Enable lightweight profiling.
            embedder (BaseEmbedder, optional): Custom embedder implementation.
            vector_store (BaseVectorStore, optional): Custom vector store implementation.
            cache_store (BaseCacheStore, optional): Custom cache store implementation.
        """
        resource_limits = ResourceLimits(
            max_memory_gb=max_memory_gb,
            max_cpu_percent=max_cpu_percent,
            max_memory_percent=max_memory_percent
        )
        self.cache = Semantrix(
            similarity_threshold=similarity_threshold,
            resource_limits=resource_limits,
            enable_profiling=enable_profiling,
            embedder=embedder,
            vector_store=vector_store,
            cache_store=cache_store
        )

    def get(self, prompt: str) -> str | None:
        """Retrieve a cached response for the given prompt, if available."""
        return self.cache.get(prompt)

    def set(self, prompt: str, response: str):
        """Store a prompt-response pair in the cache."""
        self.cache.set(prompt, response)

    def explain(self, prompt: str) -> ExplainResult:
        """Debug why a prompt missed the cache (returns detailed explanation)."""
        return self.cache.explain(prompt)

    @property
    def profiler_stats(self) -> dict:
        """Return profiling statistics if enabled."""
        return self.cache.profiler.get_stats()

    @property
    def resource_limits(self) -> ResourceLimits:
        """Return the resource limits object."""
        return self.cache.resource_limits 