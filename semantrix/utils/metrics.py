"""
Metrics collection system for Semantrix.

This module provides a comprehensive metrics collection system to track
operation counters, error rates, performance metrics, and system health.
"""
import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from threading import Lock

from .logging import get_logger, get_metrics_logger


@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class MetricSeries:
    """A series of metric data points."""
    name: str
    points: deque = field(default_factory=lambda: deque(maxlen=1000))
    labels: Dict[str, str] = field(default_factory=dict)


class Counter:
    """
    A simple counter metric.
    
    Counters only increase and are used to track cumulative values
    like total requests, errors, etc.
    """
    
    def __init__(self, name: str, description: str = "", labels: Optional[Dict[str, str]] = None):
        """
        Initialize a counter.
        
        Args:
            name: Counter name
            description: Counter description
            labels: Optional labels for the counter
        """
        self.name = name
        self.description = description
        self.labels = labels or {}
        self._value = 0
        self._lock = Lock()
    
    def increment(self, value: int = 1):
        """
        Increment the counter.
        
        Args:
            value: Value to increment by (default: 1)
        """
        with self._lock:
            self._value += value
    
    def get_value(self) -> int:
        """
        Get the current counter value.
        
        Returns:
            Current counter value
        """
        with self._lock:
            return self._value
    
    def reset(self):
        """Reset the counter to zero."""
        with self._lock:
            self._value = 0


class Gauge:
    """
    A gauge metric that can go up and down.
    
    Gauges are used to track current values like memory usage,
    active connections, etc.
    """
    
    def __init__(self, name: str, description: str = "", labels: Optional[Dict[str, str]] = None):
        """
        Initialize a gauge.
        
        Args:
            name: Gauge name
            description: Gauge description
            labels: Optional labels for the gauge
        """
        self.name = name
        self.description = description
        self.labels = labels or {}
        self._value = 0.0
        self._lock = Lock()
    
    def set(self, value: float):
        """
        Set the gauge value.
        
        Args:
            value: New gauge value
        """
        with self._lock:
            self._value = value
    
    def increment(self, value: float = 1.0):
        """
        Increment the gauge value.
        
        Args:
            value: Value to increment by (default: 1.0)
        """
        with self._lock:
            self._value += value
    
    def decrement(self, value: float = 1.0):
        """
        Decrement the gauge value.
        
        Args:
            value: Value to decrement by (default: 1.0)
        """
        with self._lock:
            self._value -= value
    
    def get_value(self) -> float:
        """
        Get the current gauge value.
        
        Returns:
            Current gauge value
        """
        with self._lock:
            return self._value


class Histogram:
    """
    A histogram metric for tracking value distributions.
    
    Histograms are used to track the distribution of values
    like request durations, response sizes, etc.
    """
    
    def __init__(self, name: str, description: str = "", labels: Optional[Dict[str, str]] = None, buckets: Optional[List[float]] = None):
        """
        Initialize a histogram.
        
        Args:
            name: Histogram name
            description: Histogram description
            labels: Optional labels for the histogram
            buckets: Bucket boundaries for the histogram
        """
        self.name = name
        self.description = description
        self.labels = labels or {}
        self.buckets = buckets or [0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0]
        self._lock = Lock()
        self._reset()
    
    def _reset(self):
        """Reset histogram state."""
        self._sum = 0.0
        self._count = 0
        self._bucket_counts = [0] * (len(self.buckets) + 1)  # +1 for +Inf bucket
        self._min = float('inf')
        self._max = float('-inf')
    
    def observe(self, value: float):
        """
        Observe a value in the histogram.
        
        Args:
            value: Value to observe
        """
        with self._lock:
            self._sum += value
            self._count += 1
            self._min = min(self._min, value)
            self._max = max(self._max, value)
            
            # Find the appropriate bucket
            bucket_index = len(self.buckets)
            for i, bucket in enumerate(self.buckets):
                if value <= bucket:
                    bucket_index = i
                    break
            
            self._bucket_counts[bucket_index] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get histogram summary statistics.
        
        Returns:
            Dictionary with summary statistics
        """
        with self._lock:
            if self._count == 0:
                return {
                    'count': 0,
                    'sum': 0.0,
                    'min': float('inf'),
                    'max': float('-inf'),
                    'mean': 0.0,
                    'buckets': dict(zip(self.buckets + ['+Inf'], self._bucket_counts))
                }
            
            return {
                'count': self._count,
                'sum': self._sum,
                'min': self._min,
                'max': self._max,
                'mean': self._sum / self._count,
                'buckets': dict(zip(self.buckets + ['+Inf'], self._bucket_counts))
            }
    
    def reset(self):
        """Reset the histogram."""
        with self._lock:
            self._reset()


class Timer:
    """
    A timer for measuring operation durations.
    
    Timers automatically record measurements in an associated histogram.
    """
    
    def __init__(self, histogram: Histogram):
        """
        Initialize a timer.
        
        Args:
            histogram: Histogram to record measurements in
        """
        self.histogram = histogram
        self._start_time: Optional[float] = None
    
    def start(self):
        """Start the timer."""
        self._start_time = time.time()
    
    def stop(self) -> float:
        """
        Stop the timer and record the duration.
        
        Returns:
            Duration in seconds
        """
        if self._start_time is None:
            raise RuntimeError("Timer was not started")
        
        duration = time.time() - self._start_time
        self.histogram.observe(duration)
        self._start_time = None
        return duration
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class MetricsRegistry:
    """
    Central registry for all metrics.
    
    This class manages all metrics and provides methods to collect
    and export metric data.
    """
    
    def __init__(self):
        """Initialize the metrics registry."""
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._lock = Lock()
        self.logger = get_logger("semantrix.metrics")
        self.metrics_logger = get_metrics_logger()
    
    def counter(self, name: str, description: str = "", labels: Optional[Dict[str, str]] = None) -> Counter:
        """
        Get or create a counter.
        
        Args:
            name: Counter name
            description: Counter description
            labels: Optional labels
            
        Returns:
            Counter instance
        """
        with self._lock:
            if name not in self._counters:
                self._counters[name] = Counter(name, description, labels)
            return self._counters[name]
    
    def gauge(self, name: str, description: str = "", labels: Optional[Dict[str, str]] = None) -> Gauge:
        """
        Get or create a gauge.
        
        Args:
            name: Gauge name
            description: Gauge description
            labels: Optional labels
            
        Returns:
            Gauge instance
        """
        with self._lock:
            if name not in self._gauges:
                self._gauges[name] = Gauge(name, description, labels)
            return self._gauges[name]
    
    def histogram(self, name: str, description: str = "", labels: Optional[Dict[str, str]] = None, buckets: Optional[List[float]] = None) -> Histogram:
        """
        Get or create a histogram.
        
        Args:
            name: Histogram name
            description: Histogram description
            labels: Optional labels
            buckets: Bucket boundaries
            
        Returns:
            Histogram instance
        """
        with self._lock:
            if name not in self._histograms:
                self._histograms[name] = Histogram(name, description, labels, buckets)
            return self._histograms[name]
    
    def timer(self, name: str, description: str = "", labels: Optional[Dict[str, str]] = None) -> Timer:
        """
        Get or create a timer.
        
        Args:
            name: Timer name
            description: Timer description
            labels: Optional labels
            
        Returns:
            Timer instance
        """
        histogram = self.histogram(f"{name}_duration", description, labels)
        return Timer(histogram)
    
    def collect_metrics(self) -> Dict[str, Any]:
        """
        Collect all current metric values.
        
        Returns:
            Dictionary containing all metric data
        """
        with self._lock:
            metrics = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'counters': {},
                'gauges': {},
                'histograms': {}
            }
            
            # Collect counter values
            for name, counter in self._counters.items():
                metrics['counters'][name] = {
                    'value': counter.get_value(),
                    'description': counter.description,
                    'labels': counter.labels
                }
            
            # Collect gauge values
            for name, gauge in self._gauges.items():
                metrics['gauges'][name] = {
                    'value': gauge.get_value(),
                    'description': gauge.description,
                    'labels': gauge.labels
                }
            
            # Collect histogram summaries
            for name, histogram in self._histograms.items():
                metrics['histograms'][name] = {
                    'summary': histogram.get_summary(),
                    'description': histogram.description,
                    'labels': histogram.labels
                }
            
            return metrics
    
    def reset_all(self):
        """Reset all metrics."""
        with self._lock:
            for counter in self._counters.values():
                counter.reset()
            for histogram in self._histograms.values():
                histogram.reset()
            # Note: Gauges are not reset as they represent current state
    
    def log_metrics(self, level: str = "INFO"):
        """
        Log current metrics.
        
        Args:
            level: Log level to use
        """
        metrics = self.collect_metrics()
        log_method = getattr(self.logger, level.lower())
        log_method("Current metrics: %s", metrics)


# Global metrics registry
_metrics_registry: Optional[MetricsRegistry] = None


def get_metrics_registry() -> MetricsRegistry:
    """
    Get the global metrics registry.
    
    Returns:
        Metrics registry instance
    """
    global _metrics_registry
    if _metrics_registry is None:
        _metrics_registry = MetricsRegistry()
    return _metrics_registry


def counter(name: str, description: str = "", labels: Optional[Dict[str, str]] = None) -> Counter:
    """
    Get or create a counter.
    
    Args:
        name: Counter name
        description: Counter description
        labels: Optional labels
        
    Returns:
        Counter instance
    """
    return get_metrics_registry().counter(name, description, labels)


def gauge(name: str, description: str = "", labels: Optional[Dict[str, str]] = None) -> Gauge:
    """
    Get or create a gauge.
    
    Args:
        name: Gauge name
        description: Gauge description
        labels: Optional labels
        
    Returns:
        Gauge instance
    """
    return get_metrics_registry().gauge(name, description, labels)


def histogram(name: str, description: str = "", labels: Optional[Dict[str, str]] = None, buckets: Optional[List[float]] = None) -> Histogram:
    """
    Get or create a histogram.
    
    Args:
        name: Histogram name
        description: Histogram description
        labels: Optional labels
        buckets: Bucket boundaries
        
    Returns:
        Histogram instance
    """
    return get_metrics_registry().histogram(name, description, labels, buckets)


def timer(name: str, description: str = "", labels: Optional[Dict[str, str]] = None) -> Timer:
    """
    Get or create a timer.
    
    Args:
        name: Timer name
        description: Timer description
        labels: Optional labels
        
    Returns:
        Timer instance
    """
    return get_metrics_registry().timer(name, description, labels)


def collect_metrics() -> Dict[str, Any]:
    """
    Collect all current metric values.
    
    Returns:
        Dictionary containing all metric data
    """
    return get_metrics_registry().collect_metrics()


def reset_metrics():
    """Reset all metrics."""
    get_metrics_registry().reset_all()


def log_metrics(level: str = "INFO"):
    """
    Log current metrics.
    
    Args:
        level: Log level to use
    """
    get_metrics_registry().log_metrics(level)


# Pre-defined metrics for common operations (Prometheus-compatible naming)
REQUEST_COUNTER = counter("semantrix_requests_total", "Total number of requests")
ERROR_COUNTER = counter("semantrix_errors_total", "Total number of errors")
CACHE_HIT_COUNTER = counter("semantrix_cache_hits_total", "Total number of cache hits")
CACHE_MISS_COUNTER = counter("semantrix_cache_misses_total", "Total number of cache misses")
ACTIVE_CONNECTIONS_GAUGE = gauge("semantrix_active_connections", "Number of active connections")
REQUEST_DURATION_HISTOGRAM = histogram("semantrix_request_duration_seconds", "Request duration distribution")
CACHE_OPERATION_DURATION_HISTOGRAM = histogram("semantrix_cache_operation_duration_seconds", "Cache operation duration distribution")

# Additional metrics for better observability
SEMANTIC_SEARCH_COUNTER = counter("semantrix_semantic_searches_total", "Total number of semantic searches")
SEMANTIC_SEARCH_DURATION_HISTOGRAM = histogram("semantrix_semantic_search_duration_seconds", "Semantic search duration distribution")
EMBEDDING_GENERATION_COUNTER = counter("semantrix_embeddings_generated_total", "Total number of embeddings generated")
EMBEDDING_DURATION_HISTOGRAM = histogram("semantrix_embedding_duration_seconds", "Embedding generation duration distribution")
VECTOR_STORE_OPERATIONS_COUNTER = counter("semantrix_vector_store_operations_total", "Total number of vector store operations")
TOMBSTONE_OPERATIONS_COUNTER = counter("semantrix_tombstone_operations_total", "Total number of tombstone operations")

# Cache Store-Specific Metrics
REDIS_OPERATIONS_COUNTER = counter("semantrix_redis_operations_total", "Total number of Redis operations")
REDIS_OPERATION_DURATION_HISTOGRAM = histogram("semantrix_redis_operation_duration_seconds", "Redis operation duration distribution")
REDIS_CONNECTION_POOL_GAUGE = gauge("semantrix_redis_connection_pool_size", "Redis connection pool size")
REDIS_MEMORY_USAGE_GAUGE = gauge("semantrix_redis_memory_usage_bytes", "Redis memory usage in bytes")

MONGODB_QUERIES_COUNTER = counter("semantrix_mongodb_queries_total", "Total number of MongoDB queries")
MONGODB_QUERY_DURATION_HISTOGRAM = histogram("semantrix_mongodb_query_duration_seconds", "MongoDB query duration distribution")
MONGODB_CONNECTIONS_ACTIVE_GAUGE = gauge("semantrix_mongodb_connections_active", "Active MongoDB connections")
MONGODB_COLLECTION_SIZE_GAUGE = gauge("semantrix_mongodb_collection_size_bytes", "MongoDB collection size in bytes")

POSTGRESQL_OPERATIONS_COUNTER = counter("semantrix_postgresql_operations_total", "Total number of PostgreSQL operations")
POSTGRESQL_OPERATION_DURATION_HISTOGRAM = histogram("semantrix_postgresql_operation_duration_seconds", "PostgreSQL operation duration distribution")
POSTGRESQL_CONNECTIONS_ACTIVE_GAUGE = gauge("semantrix_postgresql_connections_active", "Active PostgreSQL connections")
POSTGRESQL_CONNECTION_POOL_UTILIZATION_GAUGE = gauge("semantrix_postgresql_connection_pool_utilization", "PostgreSQL connection pool utilization percentage")

DYNAMODB_OPERATIONS_COUNTER = counter("semantrix_dynamodb_operations_total", "Total number of DynamoDB operations")
DYNAMODB_OPERATION_DURATION_HISTOGRAM = histogram("semantrix_dynamodb_operation_duration_seconds", "DynamoDB operation duration distribution")
DYNAMODB_THROTTLED_REQUESTS_COUNTER = counter("semantrix_dynamodb_throttled_requests_total", "Total number of throttled DynamoDB requests")
DYNAMODB_CONSUMED_CAPACITY_GAUGE = gauge("semantrix_dynamodb_consumed_capacity", "DynamoDB consumed capacity units")

ELASTICACHE_OPERATIONS_COUNTER = counter("semantrix_elasticache_operations_total", "Total number of ElastiCache operations")
ELASTICACHE_OPERATION_DURATION_HISTOGRAM = histogram("semantrix_elasticache_operation_duration_seconds", "ElastiCache operation duration distribution")
ELASTICACHE_NODE_HEALTH_GAUGE = gauge("semantrix_elasticache_node_health", "ElastiCache node health status")
ELASTICACHE_REPLICATION_LAG_GAUGE = gauge("semantrix_elasticache_replication_lag_seconds", "ElastiCache replication lag in seconds")

MEMCACHED_OPERATIONS_COUNTER = counter("semantrix_memcached_operations_total", "Total number of Memcached operations")
MEMCACHED_OPERATION_DURATION_HISTOGRAM = histogram("semantrix_memcached_operation_duration_seconds", "Memcached operation duration distribution")
MEMCACHED_MEMORY_USAGE_GAUGE = gauge("semantrix_memcached_memory_usage_bytes", "Memcached memory usage in bytes")
MEMCACHED_EVICTIONS_COUNTER = counter("semantrix_memcached_evictions_total", "Total number of Memcached evictions")

SQLITE_OPERATIONS_COUNTER = counter("semantrix_sqlite_operations_total", "Total number of SQLite operations")
SQLITE_OPERATION_DURATION_HISTOGRAM = histogram("semantrix_sqlite_operation_duration_seconds", "SQLite operation duration distribution")
SQLITE_DATABASE_SIZE_GAUGE = gauge("semantrix_sqlite_database_size_bytes", "SQLite database size in bytes")
SQLITE_WAL_SIZE_GAUGE = gauge("semantrix_sqlite_wal_size_bytes", "SQLite WAL file size in bytes")

GOOGLE_MEMORYSTORE_OPERATIONS_COUNTER = counter("semantrix_google_memorystore_operations_total", "Total number of Google Memorystore operations")
GOOGLE_MEMORYSTORE_OPERATION_DURATION_HISTOGRAM = histogram("semantrix_google_memorystore_operation_duration_seconds", "Google Memorystore operation duration distribution")
GOOGLE_MEMORYSTORE_INSTANCE_HEALTH_GAUGE = gauge("semantrix_google_memorystore_instance_health", "Google Memorystore instance health status")
GOOGLE_MEMORYSTORE_MEMORY_USAGE_GAUGE = gauge("semantrix_google_memorystore_memory_usage_bytes", "Google Memorystore memory usage in bytes")

# Vector Store Performance Metrics
VECTOR_SEARCH_LATENCY_HISTOGRAM = histogram("semantrix_vector_search_latency_seconds", "Vector search latency distribution")
VECTOR_INDEX_SIZE_GAUGE = gauge("semantrix_vector_index_size", "Vector index size in number of vectors")
VECTOR_SEARCH_RESULTS_COUNT_HISTOGRAM = histogram("semantrix_vector_search_results_count", "Number of results returned by vector search")
VECTOR_SEARCH_ACCURACY_SCORE_HISTOGRAM = histogram("semantrix_vector_search_accuracy_score", "Vector search accuracy score distribution")

# Vector Store-Specific Metrics
FAISS_OPERATIONS_COUNTER = counter("semantrix_faiss_operations_total", "Total number of FAISS operations")
FAISS_OPERATION_DURATION_HISTOGRAM = histogram("semantrix_faiss_operation_duration_seconds", "FAISS operation duration distribution")
FAISS_INDEX_BUILD_TIME_HISTOGRAM = histogram("semantrix_faiss_index_build_time_seconds", "FAISS index build time distribution")
FAISS_MEMORY_USAGE_GAUGE = gauge("semantrix_faiss_memory_usage_bytes", "FAISS memory usage in bytes")
FAISS_SEARCH_COMPLEXITY_GAUGE = gauge("semantrix_faiss_search_complexity", "FAISS search complexity metric")

PINECONE_OPERATIONS_COUNTER = counter("semantrix_pinecone_operations_total", "Total number of Pinecone operations")
PINECONE_OPERATION_DURATION_HISTOGRAM = histogram("semantrix_pinecone_operation_duration_seconds", "Pinecone operation duration distribution")
PINECONE_API_CALL_LATENCY_HISTOGRAM = histogram("semantrix_pinecone_api_call_latency_seconds", "Pinecone API call latency distribution")
PINECONE_INDEX_OPERATIONS_COUNTER = counter("semantrix_pinecone_index_operations_total", "Total number of Pinecone index operations")
PINECONE_NAMESPACE_METRICS_GAUGE = gauge("semantrix_pinecone_namespace_metrics", "Pinecone namespace metrics")

QDRANT_OPERATIONS_COUNTER = counter("semantrix_qdrant_operations_total", "Total number of Qdrant operations")
QDRANT_OPERATION_DURATION_HISTOGRAM = histogram("semantrix_qdrant_operation_duration_seconds", "Qdrant operation duration distribution")
QDRANT_COLLECTION_SIZE_GAUGE = gauge("semantrix_qdrant_collection_size", "Qdrant collection size in number of vectors")
QDRANT_SHARD_DISTRIBUTION_GAUGE = gauge("semantrix_qdrant_shard_distribution", "Qdrant shard distribution metric")
QDRANT_REPLICATION_METRICS_GAUGE = gauge("semantrix_qdrant_replication_metrics", "Qdrant replication metrics")

MILVUS_OPERATIONS_COUNTER = counter("semantrix_milvus_operations_total", "Total number of Milvus operations")
MILVUS_OPERATION_DURATION_HISTOGRAM = histogram("semantrix_milvus_operation_duration_seconds", "Milvus operation duration distribution")
MILVUS_COLLECTION_STATISTICS_GAUGE = gauge("semantrix_milvus_collection_statistics", "Milvus collection statistics")
MILVUS_QUERY_PERFORMANCE_HISTOGRAM = histogram("semantrix_milvus_query_performance_seconds", "Milvus query performance distribution")
MILVUS_RESOURCE_USAGE_GAUGE = gauge("semantrix_milvus_resource_usage", "Milvus resource usage metrics")

CHROMA_OPERATIONS_COUNTER = counter("semantrix_chroma_operations_total", "Total number of Chroma operations")
CHROMA_OPERATION_DURATION_HISTOGRAM = histogram("semantrix_chroma_operation_duration_seconds", "Chroma operation duration distribution")
CHROMA_COLLECTION_SIZE_GAUGE = gauge("semantrix_chroma_collection_size", "Chroma collection size in number of vectors")
CHROMA_EMBEDDING_STORE_SIZE_GAUGE = gauge("semantrix_chroma_embedding_store_size_bytes", "Chroma embedding store size in bytes")

PGVECTOR_OPERATIONS_COUNTER = counter("semantrix_pgvector_operations_total", "Total number of pgvector operations")
PGVECTOR_OPERATION_DURATION_HISTOGRAM = histogram("semantrix_pgvector_operation_duration_seconds", "pgvector operation duration distribution")
PGVECTOR_TABLE_SIZE_GAUGE = gauge("semantrix_pgvector_table_size_bytes", "pgvector table size in bytes")
PGVECTOR_INDEX_USAGE_GAUGE = gauge("semantrix_pgvector_index_usage", "pgvector index usage statistics")

VECTOR_REDIS_OPERATIONS_COUNTER = counter("semantrix_vector_redis_operations_total", "Total number of Vector Redis operations")
VECTOR_REDIS_OPERATION_DURATION_HISTOGRAM = histogram("semantrix_vector_redis_operation_duration_seconds", "Vector Redis operation duration distribution")
VECTOR_REDIS_INDEX_SIZE_GAUGE = gauge("semantrix_vector_redis_index_size", "Vector Redis index size in number of vectors")
VECTOR_REDIS_MEMORY_USAGE_GAUGE = gauge("semantrix_vector_redis_memory_usage_bytes", "Vector Redis memory usage in bytes")

# Embedding Model Performance Metrics
EMBEDDING_MODEL_LATENCY_HISTOGRAM = histogram("semantrix_embedding_model_latency_seconds", "Embedding model latency distribution")
EMBEDDING_MODEL_MEMORY_USAGE_GAUGE = gauge("semantrix_embedding_model_memory_usage_bytes", "Embedding model memory usage in bytes")
EMBEDDING_MODEL_THROUGHPUT_HISTOGRAM = histogram("semantrix_embedding_model_throughput_embeddings_per_second", "Embedding model throughput in embeddings per second")
EMBEDDING_MODEL_ACCURACY_SCORE_HISTOGRAM = histogram("semantrix_embedding_model_accuracy_score", "Embedding model accuracy score distribution")

# Embedding Model-Specific Metrics
OPENAI_EMBEDDING_OPERATIONS_COUNTER = counter("semantrix_openai_embedding_operations_total", "Total number of OpenAI embedding operations")
OPENAI_EMBEDDING_LATENCY_HISTOGRAM = histogram("semantrix_openai_embedding_latency_seconds", "OpenAI embedding latency distribution")
OPENAI_API_RATE_LIMITS_GAUGE = gauge("semantrix_openai_api_rate_limits", "OpenAI API rate limit status")
OPENAI_TOKEN_USAGE_COUNTER = counter("semantrix_openai_token_usage_total", "Total OpenAI token usage")
OPENAI_MODEL_VERSION_GAUGE = gauge("semantrix_openai_model_version", "OpenAI model version tracking")

COHERE_EMBEDDING_OPERATIONS_COUNTER = counter("semantrix_cohere_embedding_operations_total", "Total number of Cohere embedding operations")
COHERE_EMBEDDING_LATENCY_HISTOGRAM = histogram("semantrix_cohere_embedding_latency_seconds", "Cohere embedding latency distribution")
COHERE_API_RESPONSE_TIMES_HISTOGRAM = histogram("semantrix_cohere_api_response_times_seconds", "Cohere API response times distribution")
COHERE_MODEL_PERFORMANCE_GAUGE = gauge("semantrix_cohere_model_performance", "Cohere model performance metrics")

SENTENCE_TRANSFORMER_OPERATIONS_COUNTER = counter("semantrix_sentence_transformer_operations_total", "Total number of Sentence Transformer operations")
SENTENCE_TRANSFORMER_LATENCY_HISTOGRAM = histogram("semantrix_sentence_transformer_latency_seconds", "Sentence Transformer latency distribution")
SENTENCE_TRANSFORMER_MODEL_LOADING_TIME_HISTOGRAM = histogram("semantrix_sentence_transformer_model_loading_time_seconds", "Sentence Transformer model loading time distribution")
SENTENCE_TRANSFORMER_INFERENCE_SPEED_HISTOGRAM = histogram("semantrix_sentence_transformer_inference_speed_embeddings_per_second", "Sentence Transformer inference speed distribution")
SENTENCE_TRANSFORMER_MEMORY_EFFICIENCY_GAUGE = gauge("semantrix_sentence_transformer_memory_efficiency", "Sentence Transformer memory efficiency metric")

ONNX_EMBEDDING_OPERATIONS_COUNTER = counter("semantrix_onnx_embedding_operations_total", "Total number of ONNX embedding operations")
ONNX_EMBEDDING_LATENCY_HISTOGRAM = histogram("semantrix_onnx_embedding_latency_seconds", "ONNX embedding latency distribution")
ONNX_MODEL_OPTIMIZATION_METRICS_GAUGE = gauge("semantrix_onnx_model_optimization_metrics", "ONNX model optimization metrics")
ONNX_INFERENCE_LATENCY_HISTOGRAM = histogram("semantrix_onnx_inference_latency_seconds", "ONNX inference latency distribution")
ONNX_HARDWARE_UTILIZATION_GAUGE = gauge("semantrix_onnx_hardware_utilization", "ONNX hardware utilization metrics")

MISTRAL_EMBEDDING_OPERATIONS_COUNTER = counter("semantrix_mistral_embedding_operations_total", "Total number of Mistral embedding operations")
MISTRAL_EMBEDDING_LATENCY_HISTOGRAM = histogram("semantrix_mistral_embedding_latency_seconds", "Mistral embedding latency distribution")
MISTRAL_MODEL_PERFORMANCE_GAUGE = gauge("semantrix_mistral_model_performance", "Mistral model performance metrics")

OLLAMA_EMBEDDING_OPERATIONS_COUNTER = counter("semantrix_ollama_embedding_operations_total", "Total number of Ollama embedding operations")
OLLAMA_EMBEDDING_LATENCY_HISTOGRAM = histogram("semantrix_ollama_embedding_latency_seconds", "Ollama embedding latency distribution")
OLLAMA_LOCAL_MODEL_PERFORMANCE_GAUGE = gauge("semantrix_ollama_local_model_performance", "Ollama local model performance metrics")

LANGCHAIN_EMBEDDING_OPERATIONS_COUNTER = counter("semantrix_langchain_embedding_operations_total", "Total number of LangChain embedding operations")
LANGCHAIN_EMBEDDING_LATENCY_HISTOGRAM = histogram("semantrix_langchain_embedding_latency_seconds", "LangChain embedding latency distribution")
LANGCHAIN_INTEGRATION_PERFORMANCE_GAUGE = gauge("semantrix_langchain_integration_performance", "LangChain integration performance metrics")

# Business Intelligence Metrics
CACHE_HIT_RATE_GAUGE = gauge("semantrix_cache_hit_rate_percentage", "Cache hit rate as a percentage")
CACHE_EFFECTIVENESS_SCORE_GAUGE = gauge("semantrix_cache_effectiveness_score", "Cache effectiveness score (0-1)")
SEMANTIC_SIMILARITY_DISTRIBUTION_HISTOGRAM = histogram("semantrix_semantic_similarity_distribution", "Distribution of semantic similarity scores")
CACHE_EVICTION_REASONS_COUNTER = counter("semantrix_cache_eviction_reasons_total", "Total cache evictions by reason")

# User Experience Metrics
RESPONSE_TIME_PERCENTILE_95_HISTOGRAM = histogram("semantrix_response_time_percentile_95_seconds", "95th percentile response time distribution")
RESPONSE_TIME_PERCENTILE_99_HISTOGRAM = histogram("semantrix_response_time_percentile_99_seconds", "99th percentile response time distribution")
USER_SATISFACTION_SCORE_GAUGE = gauge("semantrix_user_satisfaction_score", "User satisfaction score (0-1)")
QUERY_COMPLEXITY_SCORE_HISTOGRAM = histogram("semantrix_query_complexity_score", "Query complexity score distribution")

# Resource Utilization Metrics
MEMORY_USAGE_GAUGE = gauge("semantrix_memory_usage_bytes", "Memory usage in bytes")
CPU_USAGE_GAUGE = gauge("semantrix_cpu_usage_percentage", "CPU usage percentage")
DISK_IO_OPERATIONS_COUNTER = counter("semantrix_disk_io_operations_total", "Total disk I/O operations")
NETWORK_BANDWIDTH_GAUGE = gauge("semantrix_network_bandwidth_bytes", "Network bandwidth usage in bytes")

# Application Resource Metrics
ACTIVE_CONNECTIONS_TOTAL_GAUGE = gauge("semantrix_active_connections_total", "Total number of active connections")
CONNECTION_POOL_UTILIZATION_GAUGE = gauge("semantrix_connection_pool_utilization", "Connection pool utilization percentage")
THREAD_POOL_SIZE_GAUGE = gauge("semantrix_thread_pool_size", "Thread pool size")
QUEUE_DEPTH_GAUGE = gauge("semantrix_queue_depth", "Current queue depth")
