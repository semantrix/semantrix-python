from semantrix.embedding.embedding import Embedder
from semantrix.vector_store.vector_store import FAISSVectorStore
from semantrix.cache_store.eviction import LRUEviction
from semantrix.utils.resource_limits import ResourceLimits
from semantrix.utils.profiling import Profiler

class Semantrix:
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        resource_limits: ResourceLimits = ResourceLimits(),
        enable_profiling: bool = False
    ):
        self.embedder = Embedder()
        self.vector_store = FAISSVectorStore()
        self.eviction = LRUEviction()
        self.resource_limits = resource_limits
        self.profiler = Profiler(enabled=enable_profiling)
        self.similarity_threshold = similarity_threshold

    def get(self, prompt: str) -> str | None:
        with self.profiler.record("get"):
            # 1. Check resource constraints
            if not self.resource_limits.allow_operation():
                return None
            # 2. Exact match
            exact_match = self.eviction.get_exact(prompt)
            if exact_match is not None:
                return exact_match
            # 3. Semantic search
            embedding = self.embedder.encode(prompt)
            match = self.vector_store.search(embedding, self.similarity_threshold)
            if match is not None:
                return match
            return None

    def set(self, prompt: str, response: str):
        with self.profiler.record("set"):
            if not self.resource_limits.allow_operation():
                return
            # 1. Add to cache
            embedding = self.embedder.encode(prompt)
            self.vector_store.add(embedding, response)
            self.eviction.add(prompt, response)
            # 2. Enforce limits
            self.eviction.enforce_limits(self.resource_limits)
            
    def explain(self, prompt: str) -> dict:
        """Debug why a prompt missed cache"""
        embedding = self.embedder.encode(prompt)
        matches = self.vector_store.search(embedding, top_k=3)
        if matches is None:
            matches = []
        return {
            "threshold": self.similarity_threshold,
            "matches": [
                {"text": text, "similarity": float(sim)} 
                for text, sim in matches
            ]
        } 