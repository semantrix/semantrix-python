"""
OpenTelemetry integration for Semantrix.

This module provides comprehensive OpenTelemetry integration including:
- Metrics bridging from Semantrix metrics to OpenTelemetry
- Distributed tracing for operations
- Automatic instrumentation of reliability features
- Configurable exporters and sampling
"""
import asyncio
import functools
import inspect
import os
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar, Union, cast
from urllib.parse import urlparse

# OpenTelemetry imports
try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.zipkin.json import ZipkinExporter
    # Optional instrumentation imports
    try:
        from opentelemetry.instrumentation.aiohttp import AioHttpClientInstrumentor
    except ImportError:
        AioHttpClientInstrumentor = None
    
    try:
        from opentelemetry.instrumentation.logging import LoggingInstrumentor
    except ImportError:
        LoggingInstrumentor = None
    
    try:
        from opentelemetry.instrumentation.requests import RequestsInstrumentor
    except ImportError:
        RequestsInstrumentor = None
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.trace.sampling import ParentBasedTraceIdRatio
    from opentelemetry.trace import Span, Status, StatusCode
    from opentelemetry.trace.span import Span as SpanType
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    # Create dummy classes for type hints when OpenTelemetry is not available
    class Span:
        pass
    
    class SpanType:
        pass
    
    class Status:
        pass
    
    class StatusCode:
        pass
    
    class Resource:
        pass
    
    class TracerProvider:
        pass
    
    class MeterProvider:
        pass
    
    class BatchSpanProcessor:
        pass
    
    class ConsoleSpanExporter:
        pass
    
    class ConsoleMetricExporter:
        pass
    
    class PeriodicExportingMetricReader:
        pass
    
    class ParentBasedTraceIdRatio:
        pass
    
    class JaegerExporter:
        pass
    
    class OTLPSpanExporter:
        pass
    
    class OTLPMetricExporter:
        pass
    
    class ZipkinExporter:
        pass
    
    class AioHttpClientInstrumentor:
        pass
    
    class LoggingInstrumentor:
        pass
    
    class RequestsInstrumentor:
        pass
    
    # Dummy trace and metrics modules
    class trace:
        pass
    
    class metrics:
        pass

from semantrix.exceptions import ConfigurationError
from semantrix.utils.logging import get_logger
from semantrix.utils.metrics import Counter, Gauge, Histogram, Timer, MetricsRegistry

T = TypeVar("T")
P = TypeVar("P")
R = TypeVar("R")

logger = get_logger("semantrix.opentelemetry")


class OpenTelemetryNotAvailableError(ConfigurationError):
    """Raised when OpenTelemetry is not available."""
    pass


class OpenTelemetryManager:
    """
    Central manager for OpenTelemetry configuration and integration.
    
    This class manages the lifecycle of OpenTelemetry components including
    trace providers, meter providers, and exporters.
    """
    
    def __init__(self):
        """Initialize the OpenTelemetry manager."""
        if not OPENTELEMETRY_AVAILABLE:
            raise OpenTelemetryNotAvailableError(
                "OpenTelemetry is not available. Install with: pip install opentelemetry-api opentelemetry-sdk"
            )
        
        self._initialized = False
        self._tracer_provider: Optional[TracerProvider] = None
        self._meter_provider: Optional[MeterProvider] = None
        self._tracer = None
        self._meter = None
        self._metrics_bridge: Optional["MetricsBridge"] = None
        
    def initialize(
        self,
        service_name: str = "semantrix",
        service_version: str = "1.0.0",
        environment: str = "development",
        traces_endpoint: Optional[str] = None,
        metrics_endpoint: Optional[str] = None,
        jaeger_endpoint: Optional[str] = None,
        zipkin_endpoint: Optional[str] = None,
        sampling_rate: float = 1.0,
        enable_console_export: bool = False,
        enable_auto_instrumentation: bool = True,
        resource_attributes: Optional[Dict[str, str]] = None,
    ) -> None:
        """
        Initialize OpenTelemetry with the specified configuration.
        
        Args:
            service_name: Name of the service
            service_version: Version of the service
            environment: Environment (development, staging, production)
            traces_endpoint: OTLP traces endpoint URL
            metrics_endpoint: OTLP metrics endpoint URL
            jaeger_endpoint: Jaeger endpoint URL
            zipkin_endpoint: Zipkin endpoint URL
            sampling_rate: Sampling rate for traces (0.0 to 1.0)
            enable_console_export: Enable console export for debugging
            enable_auto_instrumentation: Enable automatic instrumentation
            resource_attributes: Additional resource attributes
        """
        if self._initialized:
            logger.warning("OpenTelemetry already initialized")
            return
        
        # Create resource
        resource_attrs = {
            "service.name": service_name,
            "service.version": service_version,
            "deployment.environment": environment,
            **(resource_attributes or {})
        }
        resource = Resource.create(resource_attrs)
        
        # Initialize trace provider
        self._initialize_trace_provider(
            resource=resource,
            traces_endpoint=traces_endpoint,
            jaeger_endpoint=jaeger_endpoint,
            zipkin_endpoint=zipkin_endpoint,
            sampling_rate=sampling_rate,
            enable_console_export=enable_console_export
        )
        
        # Initialize meter provider
        self._initialize_meter_provider(
            resource=resource,
            metrics_endpoint=metrics_endpoint,
            enable_console_export=enable_console_export
        )
        
        # Get tracer and meter
        self._tracer = trace.get_tracer(service_name, service_version)
        self._meter = metrics.get_meter(service_name, service_version)
        
        # Initialize metrics bridge
        self._metrics_bridge = MetricsBridge(self._meter)
        
        # Enable auto-instrumentation if requested
        if enable_auto_instrumentation:
            self._enable_auto_instrumentation()
        
        self._initialized = True
        logger.info(f"OpenTelemetry initialized for service: {service_name} v{service_version}")
    
    def _initialize_trace_provider(
        self,
        resource: Resource,
        traces_endpoint: Optional[str],
        jaeger_endpoint: Optional[str],
        zipkin_endpoint: Optional[str],
        sampling_rate: float,
        enable_console_export: bool
    ) -> None:
        """Initialize the trace provider with exporters."""
        exporters = []
        
        # Add console exporter for debugging
        if enable_console_export:
            exporters.append(ConsoleSpanExporter())
        
        # Add OTLP exporter
        if traces_endpoint:
            try:
                otlp_exporter = OTLPSpanExporter(endpoint=traces_endpoint)
                exporters.append(otlp_exporter)
                logger.info(f"Added OTLP trace exporter: {traces_endpoint}")
            except Exception as e:
                logger.warning(f"Failed to create OTLP trace exporter: {e}")
        
        # Add Jaeger exporter
        if jaeger_endpoint:
            try:
                jaeger_exporter = JaegerExporter(
                    agent_host_name=urlparse(jaeger_endpoint).hostname or "localhost",
                    agent_port=urlparse(jaeger_endpoint).port or 6831
                )
                exporters.append(jaeger_exporter)
                logger.info(f"Added Jaeger trace exporter: {jaeger_endpoint}")
            except Exception as e:
                logger.warning(f"Failed to create Jaeger trace exporter: {e}")
        
        # Add Zipkin exporter
        if zipkin_endpoint:
            try:
                zipkin_exporter = ZipkinExporter(endpoint=zipkin_endpoint)
                exporters.append(zipkin_exporter)
                logger.info(f"Added Zipkin trace exporter: {zipkin_endpoint}")
            except Exception as e:
                logger.warning(f"Failed to create Zipkin trace exporter: {e}")
        
        # Create trace provider
        self._tracer_provider = TracerProvider(
            resource=resource,
            sampler=ParentBasedTraceIdRatio(sampling_rate)
        )
        
        # Add span processors
        for exporter in exporters:
            self._tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
        
        # Set as global provider
        trace.set_tracer_provider(self._tracer_provider)
    
    def _initialize_meter_provider(
        self,
        resource: Resource,
        metrics_endpoint: Optional[str],
        enable_console_export: bool
    ) -> None:
        """Initialize the meter provider with exporters."""
        readers = []
        
        # Add console reader for debugging
        if enable_console_export:
            readers.append(PeriodicExportingMetricReader(ConsoleMetricExporter()))
        
        # Add OTLP reader
        if metrics_endpoint:
            try:
                otlp_exporter = OTLPMetricExporter(endpoint=metrics_endpoint)
                readers.append(PeriodicExportingMetricReader(otlp_exporter))
                logger.info(f"Added OTLP metrics exporter: {metrics_endpoint}")
            except Exception as e:
                logger.warning(f"Failed to create OTLP metrics exporter: {e}")
        
        # Create meter provider
        self._meter_provider = MeterProvider(resource=resource, metric_readers=readers)
        
        # Set as global provider
        metrics.set_meter_provider(self._meter_provider)
    
    def _enable_auto_instrumentation(self) -> None:
        """Enable automatic instrumentation of common libraries."""
        try:
            LoggingInstrumentor().instrument()
            logger.info("Enabled logging instrumentation")
        except Exception as e:
            logger.warning(f"Failed to enable logging instrumentation: {e}")
        
        try:
            RequestsInstrumentor().instrument()
            logger.info("Enabled requests instrumentation")
        except Exception as e:
            logger.warning(f"Failed to enable requests instrumentation: {e}")
        
        try:
            AioHttpClientInstrumentor().instrument()
            logger.info("Enabled aiohttp instrumentation")
        except Exception as e:
            logger.warning(f"Failed to enable aiohttp instrumentation: {e}")
    
    @property
    def tracer(self):
        """Get the OpenTelemetry tracer."""
        if not self._initialized:
            raise RuntimeError("OpenTelemetry not initialized. Call initialize() first.")
        return self._tracer
    
    @property
    def meter(self):
        """Get the OpenTelemetry meter."""
        if not self._initialized:
            raise RuntimeError("OpenTelemetry not initialized. Call initialize() first.")
        return self._meter
    
    @property
    def metrics_bridge(self) -> "MetricsBridge":
        """Get the metrics bridge."""
        if not self._initialized:
            raise RuntimeError("OpenTelemetry not initialized. Call initialize() first.")
        return self._metrics_bridge
    
    def shutdown(self) -> None:
        """Shutdown OpenTelemetry components."""
        if self._tracer_provider:
            self._tracer_provider.shutdown()
        if self._meter_provider:
            self._meter_provider.shutdown()
        self._initialized = False
        logger.info("OpenTelemetry shutdown complete")


class MetricsBridge:
    """
    Bridge between Semantrix metrics and OpenTelemetry metrics.
    
    This class provides automatic conversion of Semantrix metrics to
    OpenTelemetry format for external monitoring systems.
    """
    
    def __init__(self, meter):
        """Initialize the metrics bridge."""
        self.meter = meter
        self._counters: Dict[str, Any] = {}
        self._gauges: Dict[str, Any] = {}
        self._histograms: Dict[str, Any] = {}
    
    def register_counter(self, name: str, description: str = "", unit: str = "1") -> Any:
        """Register an OpenTelemetry counter."""
        if name not in self._counters:
            self._counters[name] = self.meter.create_counter(
                name=name,
                description=description,
                unit=unit
            )
        return self._counters[name]
    
    def register_gauge(self, name: str, description: str = "", unit: str = "1") -> Any:
        """Register an OpenTelemetry gauge."""
        if name not in self._gauges:
            self._gauges[name] = self.meter.create_up_down_counter(
                name=name,
                description=description,
                unit=unit
            )
        return self._gauges[name]
    
    def register_histogram(self, name: str, description: str = "", unit: str = "1") -> Any:
        """Register an OpenTelemetry histogram."""
        if name not in self._histograms:
            self._histograms[name] = self.meter.create_histogram(
                name=name,
                description=description,
                unit=unit
            )
        return self._histograms[name]
    
    def sync_metrics(self, metrics_registry: MetricsRegistry) -> None:
        """Synchronize Semantrix metrics to OpenTelemetry."""
        metrics_data = metrics_registry.collect_metrics()
        
        # Sync counters
        for name, counter_data in metrics_data.get('counters', {}).items():
            otel_counter = self.register_counter(name, counter_data.get('description', ''))
            otel_counter.add(
                counter_data['value'],
                attributes=counter_data.get('labels', {})
            )
        
        # Sync gauges
        for name, gauge_data in metrics_data.get('gauges', {}).items():
            otel_gauge = self.register_gauge(name, gauge_data.get('description', ''))
            otel_gauge.add(
                gauge_data['value'],
                attributes=gauge_data.get('labels', {})
            )
        
        # Sync histograms
        for name, histogram_data in metrics_data.get('histograms', {}).items():
            otel_histogram = self.register_histogram(name, histogram_data.get('description', ''))
            summary = histogram_data.get('summary', {})
            if summary.get('count', 0) > 0:
                # Record the mean value as a sample
                otel_histogram.record(
                    summary['mean'],
                    attributes=histogram_data.get('labels', {})
                )


# Global OpenTelemetry manager instance
_ot_manager: Optional[OpenTelemetryManager] = None


def initialize_opentelemetry(**kwargs) -> OpenTelemetryManager:
    """
    Initialize OpenTelemetry integration.
    
    Args:
        **kwargs: Configuration parameters for OpenTelemetryManager.initialize()
        
    Returns:
        OpenTelemetryManager instance
        
    Raises:
        OpenTelemetryNotAvailableError: If OpenTelemetry is not available
    """
    global _ot_manager
    
    if _ot_manager is None:
        _ot_manager = OpenTelemetryManager()
    
    _ot_manager.initialize(**kwargs)
    return _ot_manager


def get_opentelemetry_manager() -> Optional[OpenTelemetryManager]:
    """Get the global OpenTelemetry manager instance."""
    return _ot_manager


def trace_operation(
    operation_name: str,
    attributes: Optional[Dict[str, Any]] = None,
    record_exceptions: bool = True,
    record_return_value: bool = False
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Decorator to trace operations with OpenTelemetry.
    
    Args:
        operation_name: Name of the operation to trace
        attributes: Additional attributes to add to the span
        record_exceptions: Whether to record exceptions in the span
        record_return_value: Whether to record the return value as an attribute
        
    Returns:
        Decorated function with tracing
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        if not OPENTELEMETRY_AVAILABLE or _ot_manager is None:
            return func
        
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> R:
            return _trace_sync_operation(
                func, operation_name, attributes, record_exceptions, record_return_value, *args, **kwargs
            )
        
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> R:
            return await _trace_async_operation(
                func, operation_name, attributes, record_exceptions, record_return_value, *args, **kwargs
            )
        
        if asyncio.iscoroutinefunction(func):
            return cast(Callable[P, R], async_wrapper)
        else:
            return cast(Callable[P, R], sync_wrapper)
    
    return decorator


def _trace_sync_operation(
    func: Callable[..., R],
    operation_name: str,
    attributes: Optional[Dict[str, Any]],
    record_exceptions: bool,
    record_return_value: bool,
    *args: Any,
    **kwargs: Any
) -> R:
    """Trace a synchronous operation."""
    if _ot_manager is None:
        return func(*args, **kwargs)
    
    tracer = _ot_manager.tracer
    with tracer.start_as_current_span(operation_name) as span:
        # Add attributes
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        
        # Add function arguments as attributes
        span.set_attribute("function.name", func.__name__)
        span.set_attribute("function.module", func.__module__)
        
        try:
            result = func(*args, **kwargs)
            
            if record_return_value:
                span.set_attribute("function.return_value", str(result))
            
            span.set_status(Status(StatusCode.OK))
            return result
            
        except Exception as e:
            if record_exceptions:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


async def _trace_async_operation(
    func: Callable[..., R],
    operation_name: str,
    attributes: Optional[Dict[str, Any]],
    record_exceptions: bool,
    record_return_value: bool,
    *args: Any,
    **kwargs: Any
) -> R:
    """Trace an asynchronous operation."""
    if _ot_manager is None:
        return await func(*args, **kwargs)
    
    tracer = _ot_manager.tracer
    with tracer.start_as_current_span(operation_name) as span:
        # Add attributes
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        
        # Add function arguments as attributes
        span.set_attribute("function.name", func.__name__)
        span.set_attribute("function.module", func.__module__)
        
        try:
            result = await func(*args, **kwargs)
            
            if record_return_value:
                span.set_attribute("function.return_value", str(result))
            
            span.set_status(Status(StatusCode.OK))
            return result
            
        except Exception as e:
            if record_exceptions:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


@contextmanager
def trace_span(
    span_name: str,
    attributes: Optional[Dict[str, Any]] = None,
    record_exceptions: bool = True
):
    """
    Context manager for manual span creation.
    
    Args:
        span_name: Name of the span
        attributes: Additional attributes to add to the span
        record_exceptions: Whether to record exceptions in the span
        
    Yields:
        OpenTelemetry span object
    """
    if not OPENTELEMETRY_AVAILABLE or _ot_manager is None:
        yield None
        return
    
    tracer = _ot_manager.tracer
    with tracer.start_as_current_span(span_name) as span:
        # Add attributes
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        
        try:
            yield span
        except Exception as e:
            if record_exceptions:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


@asynccontextmanager
async def trace_span_async(
    span_name: str,
    attributes: Optional[Dict[str, Any]] = None,
    record_exceptions: bool = True
):
    """
    Async context manager for manual span creation.
    
    Args:
        span_name: Name of the span
        attributes: Additional attributes to add to the span
        record_exceptions: Whether to record exceptions in the span
        
    Yields:
        OpenTelemetry span object
    """
    if not OPENTELEMETRY_AVAILABLE or _ot_manager is None:
        yield None
        return
    
    tracer = _ot_manager.tracer
    with tracer.start_as_current_span(span_name) as span:
        # Add attributes
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        
        try:
            yield span
        except Exception as e:
            if record_exceptions:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
            raise


def add_span_event(event_name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
    """
    Add an event to the current span.
    
    Args:
        event_name: Name of the event
        attributes: Event attributes
    """
    if not OPENTELEMETRY_AVAILABLE or _ot_manager is None:
        return
    
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        current_span.add_event(event_name, attributes or {})


def set_span_attribute(key: str, value: Any) -> None:
    """
    Set an attribute on the current span.
    
    Args:
        key: Attribute key
        value: Attribute value
    """
    if not OPENTELEMETRY_AVAILABLE or _ot_manager is None:
        return
    
    current_span = trace.get_current_span()
    if current_span and current_span.is_recording():
        current_span.set_attribute(key, str(value))


def sync_metrics_to_opentelemetry(metrics_registry: Optional[MetricsRegistry] = None) -> None:
    """
    Synchronize Semantrix metrics to OpenTelemetry.
    
    Args:
        metrics_registry: Metrics registry to sync (uses global if None)
    """
    if not OPENTELEMETRY_AVAILABLE or _ot_manager is None:
        return
    
    if metrics_registry is None:
        from semantrix.utils.metrics import get_metrics_registry
        metrics_registry = get_metrics_registry()
    
    _ot_manager.metrics_bridge.sync_metrics(metrics_registry)


def shutdown_opentelemetry() -> None:
    """Shutdown OpenTelemetry integration."""
    global _ot_manager
    
    if _ot_manager:
        _ot_manager.shutdown()
        _ot_manager = None
        logger.info("OpenTelemetry integration shutdown complete")


# Convenience functions for common tracing patterns
def trace_cache_operation(operation: str):
    """Decorator for tracing cache operations."""
    return trace_operation(
        f"cache.{operation}",
        attributes={"cache.operation": operation},
        record_exceptions=True
    )


def trace_vector_store_operation(operation: str):
    """Decorator for tracing vector store operations."""
    return trace_operation(
        f"vector_store.{operation}",
        attributes={"vector_store.operation": operation},
        record_exceptions=True
    )


def trace_embedding_operation(operation: str):
    """Decorator for tracing embedding operations."""
    return trace_operation(
        f"embedding.{operation}",
        attributes={"embedding.operation": operation},
        record_exceptions=True
    )


# Integration with existing reliability features
def trace_with_reliability_features(
    operation_name: str,
    enable_retry_tracing: bool = True,
    enable_circuit_breaker_tracing: bool = True,
    enable_timeout_tracing: bool = True
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """
    Enhanced tracing decorator that integrates with reliability features.
    
    Args:
        operation_name: Name of the operation
        enable_retry_tracing: Enable retry attempt tracing
        enable_circuit_breaker_tracing: Enable circuit breaker state tracing
        enable_timeout_tracing: Enable timeout tracing
        
    Returns:
        Decorated function with enhanced tracing
    """
    def decorator(func: Callable[P, R]) -> Callable[P, R]:
        @trace_operation(operation_name, record_exceptions=True)
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> R:
            return _trace_with_reliability_sync(
                func, enable_retry_tracing, enable_circuit_breaker_tracing, 
                enable_timeout_tracing, *args, **kwargs
            )
        
        @trace_operation(operation_name, record_exceptions=True)
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> R:
            return await _trace_with_reliability_async(
                func, enable_retry_tracing, enable_circuit_breaker_tracing,
                enable_timeout_tracing, *args, **kwargs
            )
        
        if asyncio.iscoroutinefunction(func):
            return cast(Callable[P, R], async_wrapper)
        else:
            return cast(Callable[P, R], sync_wrapper)
    
    return decorator


def _trace_with_reliability_sync(
    func: Callable[..., R],
    enable_retry_tracing: bool,
    enable_circuit_breaker_tracing: bool,
    enable_timeout_tracing: bool,
    *args: Any,
    **kwargs: Any
) -> R:
    """Trace synchronous operation with reliability features."""
    start_time = time.time()
    
    try:
        result = func(*args, **kwargs)
        
        # Record success metrics
        duration = time.time() - start_time
        set_span_attribute("operation.duration", duration)
        set_span_attribute("operation.success", True)
        
        return result
        
    except Exception as e:
        # Record failure metrics
        duration = time.time() - start_time
        set_span_attribute("operation.duration", duration)
        set_span_attribute("operation.success", False)
        set_span_attribute("operation.error_type", type(e).__name__)
        set_span_attribute("operation.error_message", str(e))
        
        raise


async def _trace_with_reliability_async(
    func: Callable[..., R],
    enable_retry_tracing: bool,
    enable_circuit_breaker_tracing: bool,
    enable_timeout_tracing: bool,
    *args: Any,
    **kwargs: Any
) -> R:
    """Trace asynchronous operation with reliability features."""
    start_time = time.time()
    
    try:
        result = await func(*args, **kwargs)
        
        # Record success metrics
        duration = time.time() - start_time
        set_span_attribute("operation.duration", duration)
        set_span_attribute("operation.success", True)
        
        return result
        
    except Exception as e:
        # Record failure metrics
        duration = time.time() - start_time
        set_span_attribute("operation.duration", duration)
        set_span_attribute("operation.success", False)
        set_span_attribute("operation.error_type", type(e).__name__)
        set_span_attribute("operation.error_message", str(e))
        
        raise 