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
from . import Semantrix, ResourceLimits

class SemantrixClient:
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        max_memory_gb: float = 1.0,
        max_cpu_percent: float = 20.0,
        enable_profiling: bool = False
    ):
        """
        Initialize a Semantrix semantic cache instance.

        Args:
            similarity_threshold (float): Cosine similarity threshold for semantic match.
            max_memory_gb (float): Maximum memory usage in GB.
            max_cpu_percent (float): Maximum CPU usage percent.
            enable_profiling (bool): Enable lightweight profiling.
        """
        resource_limits = ResourceLimits(
            max_memory_gb=max_memory_gb,
            max_cpu_percent=max_cpu_percent
        )
        self.cache = Semantrix(
            similarity_threshold=similarity_threshold,
            resource_limits=resource_limits,
            enable_profiling=enable_profiling
        )

    def get(self, prompt: str) -> str | None:
        """Retrieve a cached response for the given prompt, if available."""
        return self.cache.get(prompt)

    def set(self, prompt: str, response: str):
        """Store a prompt-response pair in the cache."""
        self.cache.set(prompt, response)

    def explain(self, prompt: str) -> dict:
        """Debug why a prompt missed the cache (returns top semantic matches)."""
        return self.cache.explain(prompt)

    @property
    def profiler_stats(self) -> dict:
        """Return profiling statistics if enabled."""
        return self.cache.profiler.get_stats()

    @property
    def resource_limits(self) -> ResourceLimits:
        """Return the resource limits object."""
        return self.cache.resource_limits 