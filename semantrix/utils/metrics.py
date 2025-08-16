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


# Pre-defined metrics for common operations
REQUEST_COUNTER = counter("requests_total", "Total number of requests")
ERROR_COUNTER = counter("errors_total", "Total number of errors")
CACHE_HIT_COUNTER = counter("cache_hits_total", "Total number of cache hits")
CACHE_MISS_COUNTER = counter("cache_misses_total", "Total number of cache misses")
ACTIVE_CONNECTIONS_GAUGE = gauge("active_connections", "Number of active connections")
REQUEST_DURATION_HISTOGRAM = histogram("request_duration_seconds", "Request duration distribution")
CACHE_OPERATION_DURATION_HISTOGRAM = histogram("cache_operation_duration_seconds", "Cache operation duration distribution")
