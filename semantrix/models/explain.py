"""
Explain Model
============

Data models for explaining cache behavior and debugging cache misses.
"""

from typing import List, Optional
import attrs
from dataclasses import dataclass


@attrs.define
class CacheMatch:
    """Represents a single cache match with similarity score."""
    
    text: str = attrs.field(validator=attrs.validators.instance_of(str))
    similarity: float = attrs.field(
        validator=attrs.validators.and_(
            attrs.validators.instance_of(float),
            attrs.validators.ge(0.0),
            attrs.validators.le(1.0)
        )
    )
    
    def __str__(self) -> str:
        return f"CacheMatch(text='{self.text[:50]}...', similarity={self.similarity:.3f})"


@attrs.define
class ExplainResult:
    """Result of explaining why a prompt missed or hit the cache."""
    
    # Query information
    query: str = attrs.field(validator=attrs.validators.instance_of(str))
    
    # Cache configuration
    similarity_threshold: float = attrs.field(
        validator=attrs.validators.and_(
            attrs.validators.instance_of(float),
            attrs.validators.ge(0.0),
            attrs.validators.le(1.0)
        )
    )
    
    # Cache status
    cache_hit: bool = attrs.field(default=False)
    exact_match: bool = attrs.field(default=False)
    semantic_match: bool = attrs.field(default=False)
    
    # Match details
    top_matches: List[CacheMatch] = attrs.field(
        default=attrs.Factory(list),
        validator=attrs.validators.instance_of(list)
    )
    
    # Resource constraints
    resource_limited: bool = attrs.field(default=False)
    resource_warnings: List[str] = attrs.field(
        default=attrs.Factory(list),
        validator=attrs.validators.instance_of(list)
    )
    
    # Performance metrics
    embedding_time_ms: Optional[float] = attrs.field(default=None)
    search_time_ms: Optional[float] = attrs.field(default=None)
    total_time_ms: Optional[float] = attrs.field(default=None)
    
    def __post_init__(self):
        """Validate the explain result after initialization."""
        # Ensure cache_hit is consistent with match types
        if self.exact_match or self.semantic_match:
            if not self.cache_hit:
                self.cache_hit = True
        
        # Validate top_matches
        for match in self.top_matches:
            if not isinstance(match, CacheMatch):
                raise ValueError("All top_matches must be CacheMatch instances")
    
    @property
    def best_match(self) -> Optional[CacheMatch]:
        """Get the best matching cache entry."""
        if not self.top_matches:
            return None
        return max(self.top_matches, key=lambda m: m.similarity)
    
    @property
    def best_similarity(self) -> float:
        """Get the similarity score of the best match."""
        if not self.top_matches:
            return 0.0
        return max(match.similarity for match in self.top_matches)
    
    @property
    def missed_by(self) -> float:
        """Get how much the best match missed the threshold by."""
        return max(0.0, self.similarity_threshold - self.best_similarity)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "similarity_threshold": self.similarity_threshold,
            "cache_hit": self.cache_hit,
            "exact_match": self.exact_match,
            "semantic_match": self.semantic_match,
            "top_matches": [
                {"text": match.text, "similarity": match.similarity}
                for match in self.top_matches
            ],
            "resource_limited": self.resource_limited,
            "resource_warnings": self.resource_warnings,
            "embedding_time_ms": self.embedding_time_ms,
            "search_time_ms": self.search_time_ms,
            "total_time_ms": self.total_time_ms,
            "best_similarity": self.best_similarity,
            "missed_by": self.missed_by
        }
    
    def __str__(self) -> str:
        """String representation of the explain result."""
        status = "HIT" if self.cache_hit else "MISS"
        match_type = "exact" if self.exact_match else "semantic" if self.semantic_match else "none"
        
        result = f"Cache {status} ({match_type} match)\n"
        result += f"Query: '{self.query}'\n"
        result += f"Threshold: {self.similarity_threshold:.3f}\n"
        
        if self.top_matches:
            result += f"Best similarity: {self.best_similarity:.3f}\n"
            if not self.cache_hit:
                result += f"Missed by: {self.missed_by:.3f}\n"
            result += "Top matches:\n"
            for i, match in enumerate(self.top_matches[:3], 1):
                result += f"  {i}. {match}\n"
        
        if self.resource_limited:
            result += f"Resource limited: {', '.join(self.resource_warnings)}\n"
        
        if self.total_time_ms:
            result += f"Total time: {self.total_time_ms:.2f}ms\n"
        
        return result


# Convenience function to create explain results
def create_explain_result(
    query: str,
    similarity_threshold: float,
    top_matches: List[CacheMatch],
    cache_hit: bool = False,
    exact_match: bool = False,
    semantic_match: bool = False,
    resource_limited: bool = False,
    resource_warnings: Optional[List[str]] = None,
    embedding_time_ms: Optional[float] = None,
    search_time_ms: Optional[float] = None,
    total_time_ms: Optional[float] = None
) -> ExplainResult:
    """
    Create an ExplainResult with the given parameters.
    
    Args:
        query: The query that was explained
        similarity_threshold: The similarity threshold used
        top_matches: List of top cache matches
        cache_hit: Whether the query hit the cache
        exact_match: Whether it was an exact match
        semantic_match: Whether it was a semantic match
        resource_limited: Whether resource limits were hit
        resource_warnings: List of resource warnings
        embedding_time_ms: Time taken for embedding
        search_time_ms: Time taken for search
        total_time_ms: Total time taken
        
    Returns:
        ExplainResult instance
    """
    return ExplainResult(
        query=query,
        similarity_threshold=similarity_threshold,
        top_matches=top_matches,
        cache_hit=cache_hit,
        exact_match=exact_match,
        semantic_match=semantic_match,
        resource_limited=resource_limited,
        resource_warnings=resource_warnings or [],
        embedding_time_ms=embedding_time_ms,
        search_time_ms=search_time_ms,
        total_time_ms=total_time_ms
    ) 