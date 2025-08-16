"""
Tests for OpenTelemetry integration functionality.
"""
import asyncio
import time
import unittest
from unittest import mock
from typing import Optional

# Test with and without OpenTelemetry available
try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

from semantrix.integrations.opentelemetry import (
    OpenTelemetryManager,
    OpenTelemetryNotAvailableError,
    MetricsBridge,
    initialize_opentelemetry,
    get_opentelemetry_manager,
    trace_operation,
    trace_span,
    trace_span_async,
    add_span_event,
    set_span_attribute,
    sync_metrics_to_opentelemetry,
    shutdown_opentelemetry,
    trace_cache_operation,
    trace_vector_store_operation,
    trace_embedding_operation,
    trace_with_reliability_features,
)
from semantrix.utils.metrics import MetricsRegistry, Counter, Gauge, Histogram


class TestOpenTelemetryAvailability(unittest.TestCase):
    """Test OpenTelemetry availability detection."""

    def test_opentelemetry_availability(self):
        """Test that OpenTelemetry availability is correctly detected."""
        # This test will pass regardless of whether OpenTelemetry is available
        # The import should not fail
        from semantrix.integrations.opentelemetry import OPENTELEMETRY_AVAILABLE
        self.assertIsInstance(OPENTELEMETRY_AVAILABLE, bool)


class TestOpenTelemetryManager(unittest.TestCase):
    """Test cases for OpenTelemetryManager class."""

    def setUp(self):
        """Set up test fixtures."""
        # Clean up any existing manager
        shutdown_opentelemetry()

    def tearDown(self):
        """Clean up test fixtures."""
        shutdown_opentelemetry()

    @unittest.skipUnless(OPENTELEMETRY_AVAILABLE, "OpenTelemetry not available")
    def test_manager_initialization(self):
        """Test OpenTelemetryManager initialization."""
        manager = OpenTelemetryManager()
        self.assertIsInstance(manager, OpenTelemetryManager)
        self.assertFalse(manager._initialized)

    @unittest.skipUnless(OPENTELEMETRY_AVAILABLE, "OpenTelemetry not available")
    def test_manager_initialize_basic(self):
        """Test basic manager initialization."""
        manager = OpenTelemetryManager()
        manager.initialize(
            service_name="test-service",
            service_version="1.0.0",
            enable_console_export=True
        )
        
        self.assertTrue(manager._initialized)
        self.assertIsNotNone(manager.tracer)
        self.assertIsNotNone(manager.meter)
        self.assertIsNotNone(manager.metrics_bridge)

    @unittest.skipUnless(OPENTELEMETRY_AVAILABLE, "OpenTelemetry not available")
    def test_manager_initialize_with_endpoints(self):
        """Test manager initialization with endpoints."""
        manager = OpenTelemetryManager()
        
        # Test with mock endpoints (these won't actually connect)
        manager.initialize(
            service_name="test-service",
            service_version="1.0.0",
            traces_endpoint="http://localhost:4317",
            metrics_endpoint="http://localhost:4318",
            enable_console_export=True
        )
        
        self.assertTrue(manager._initialized)

    @unittest.skipUnless(OPENTELEMETRY_AVAILABLE, "OpenTelemetry not available")
    def test_manager_double_initialization(self):
        """Test that double initialization is handled gracefully."""
        manager = OpenTelemetryManager()
        manager.initialize(enable_console_export=True)
        
        # Second initialization should log a warning but not fail
        manager.initialize(enable_console_export=True)
        self.assertTrue(manager._initialized)

    @unittest.skipUnless(OPENTELEMETRY_AVAILABLE, "OpenTelemetry not available")
    def test_manager_shutdown(self):
        """Test manager shutdown."""
        manager = OpenTelemetryManager()
        manager.initialize(enable_console_export=True)
        
        manager.shutdown()
        self.assertFalse(manager._initialized)

    @unittest.skipUnless(OPENTELEMETRY_AVAILABLE, "OpenTelemetry not available")
    def test_access_before_initialization(self):
        """Test that accessing properties before initialization raises error."""
        manager = OpenTelemetryManager()
        
        with self.assertRaises(RuntimeError):
            _ = manager.tracer
        
        with self.assertRaises(RuntimeError):
            _ = manager.meter
        
        with self.assertRaises(RuntimeError):
            _ = manager.metrics_bridge


class TestMetricsBridge(unittest.TestCase):
    """Test cases for MetricsBridge class."""

    @unittest.skipUnless(OPENTELEMETRY_AVAILABLE, "OpenTelemetry not available")
    def test_metrics_bridge_creation(self):
        """Test MetricsBridge creation."""
        manager = OpenTelemetryManager()
        manager.initialize(enable_console_export=True)
        
        bridge = manager.metrics_bridge
        self.assertIsInstance(bridge, MetricsBridge)
        self.assertEqual(bridge.meter, manager.meter)

    @unittest.skipUnless(OPENTELEMETRY_AVAILABLE, "OpenTelemetry not available")
    def test_metrics_bridge_registration(self):
        """Test metric registration in bridge."""
        manager = OpenTelemetryManager()
        manager.initialize(enable_console_export=True)
        bridge = manager.metrics_bridge
        
        # Test counter registration
        counter = bridge.register_counter("test_counter", "Test counter")
        self.assertIsNotNone(counter)
        
        # Test gauge registration
        gauge = bridge.register_gauge("test_gauge", "Test gauge")
        self.assertIsNotNone(gauge)
        
        # Test histogram registration
        histogram = bridge.register_histogram("test_histogram", "Test histogram")
        self.assertIsNotNone(histogram)

    @unittest.skipUnless(OPENTELEMETRY_AVAILABLE, "OpenTelemetry not available")
    def test_metrics_bridge_sync(self):
        """Test metrics synchronization."""
        manager = OpenTelemetryManager()
        manager.initialize(enable_console_export=True)
        bridge = manager.metrics_bridge
        
        # Create a test metrics registry
        registry = MetricsRegistry()
        counter = registry.counter("test_counter", "Test counter")
        gauge = registry.gauge("test_gauge", "Test gauge")
        histogram = registry.histogram("test_histogram", "Test histogram")
        
        # Add some values
        counter.increment(5)
        gauge.set(42.0)
        histogram.observe(1.5)
        
        # Sync metrics
        bridge.sync_metrics(registry)
        
        # Verify that the bridge has registered the metrics
        self.assertIn("test_counter", bridge._counters)
        self.assertIn("test_gauge", bridge._gauges)
        self.assertIn("test_histogram", bridge._histograms)


class TestGlobalFunctions(unittest.TestCase):
    """Test cases for global OpenTelemetry functions."""

    def setUp(self):
        """Set up test fixtures."""
        shutdown_opentelemetry()

    def tearDown(self):
        """Clean up test fixtures."""
        shutdown_opentelemetry()

    @unittest.skipUnless(OPENTELEMETRY_AVAILABLE, "OpenTelemetry not available")
    def test_initialize_opentelemetry(self):
        """Test global initialization function."""
        manager = initialize_opentelemetry(
            service_name="test-service",
            enable_console_export=True
        )
        
        self.assertIsInstance(manager, OpenTelemetryManager)
        self.assertTrue(manager._initialized)

    @unittest.skipUnless(OPENTELEMETRY_AVAILABLE, "OpenTelemetry not available")
    def test_get_opentelemetry_manager(self):
        """Test getting the global manager."""
        # Should return None before initialization
        self.assertIsNone(get_opentelemetry_manager())
        
        # Initialize and check
        initialize_opentelemetry(enable_console_export=True)
        manager = get_opentelemetry_manager()
        self.assertIsInstance(manager, OpenTelemetryManager)

    @unittest.skipUnless(OPENTELEMETRY_AVAILABLE, "OpenTelemetry not available")
    def test_shutdown_opentelemetry(self):
        """Test global shutdown function."""
        initialize_opentelemetry(enable_console_export=True)
        self.assertIsNotNone(get_opentelemetry_manager())
        
        shutdown_opentelemetry()
        self.assertIsNone(get_opentelemetry_manager())


class TestTracingDecorators(unittest.TestCase):
    """Test cases for tracing decorators."""

    def setUp(self):
        """Set up test fixtures."""
        shutdown_opentelemetry()

    def tearDown(self):
        """Clean up test fixtures."""
        shutdown_opentelemetry()

    @unittest.skipUnless(OPENTELEMETRY_AVAILABLE, "OpenTelemetry not available")
    def test_trace_operation_sync(self):
        """Test synchronous operation tracing."""
        initialize_opentelemetry(enable_console_export=True)
        
        @trace_operation("test_operation")
        def test_function(x: int) -> int:
            return x * 2
        
        result = test_function(5)
        self.assertEqual(result, 10)

    @unittest.skipUnless(OPENTELEMETRY_AVAILABLE, "OpenTelemetry not available")
    async def test_trace_operation_async(self):
        """Test asynchronous operation tracing."""
        initialize_opentelemetry(enable_console_export=True)
        
        @trace_operation("test_async_operation")
        async def test_async_function(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2
        
        result = await test_async_function(5)
        self.assertEqual(result, 10)

    @unittest.skipUnless(OPENTELEMETRY_AVAILABLE, "OpenTelemetry not available")
    def test_trace_operation_with_attributes(self):
        """Test operation tracing with attributes."""
        initialize_opentelemetry(enable_console_export=True)
        
        @trace_operation("test_operation", attributes={"test.attr": "value"})
        def test_function():
            return "success"
        
        result = test_function()
        self.assertEqual(result, "success")

    @unittest.skipUnless(OPENTELEMETRY_AVAILABLE, "OpenTelemetry not available")
    def test_trace_operation_with_exception(self):
        """Test operation tracing with exception handling."""
        initialize_opentelemetry(enable_console_export=True)
        
        @trace_operation("test_operation", record_exceptions=True)
        def test_function():
            raise ValueError("Test error")
        
        with self.assertRaises(ValueError):
            test_function()

    @unittest.skipUnless(OPENTELEMETRY_AVAILABLE, "OpenTelemetry not available")
    def test_trace_cache_operation(self):
        """Test cache operation tracing decorator."""
        initialize_opentelemetry(enable_console_export=True)
        
        @trace_cache_operation("get")
        def get_from_cache(key: str) -> str:
            return f"value_for_{key}"
        
        result = get_from_cache("test_key")
        self.assertEqual(result, "value_for_test_key")

    @unittest.skipUnless(OPENTELEMETRY_AVAILABLE, "OpenTelemetry not available")
    def test_trace_vector_store_operation(self):
        """Test vector store operation tracing decorator."""
        initialize_opentelemetry(enable_console_export=True)
        
        @trace_vector_store_operation("search")
        def search_vectors(query: str) -> list:
            return [{"id": 1, "score": 0.9}]
        
        result = search_vectors("test query")
        self.assertEqual(len(result), 1)

    @unittest.skipUnless(OPENTELEMETRY_AVAILABLE, "OpenTelemetry not available")
    def test_trace_embedding_operation(self):
        """Test embedding operation tracing decorator."""
        initialize_opentelemetry(enable_console_export=True)
        
        @trace_embedding_operation("encode")
        def encode_text(text: str) -> list:
            return [0.1, 0.2, 0.3]
        
        result = encode_text("test text")
        self.assertEqual(len(result), 3)


class TestTracingContextManagers(unittest.TestCase):
    """Test cases for tracing context managers."""

    def setUp(self):
        """Set up test fixtures."""
        shutdown_opentelemetry()

    def tearDown(self):
        """Clean up test fixtures."""
        shutdown_opentelemetry()

    @unittest.skipUnless(OPENTELEMETRY_AVAILABLE, "OpenTelemetry not available")
    def test_trace_span_sync(self):
        """Test synchronous span context manager."""
        initialize_opentelemetry(enable_console_export=True)
        
        with trace_span("test_span", attributes={"test.attr": "value"}) as span:
            self.assertIsNotNone(span)
            # Add some work
            time.sleep(0.01)

    @unittest.skipUnless(OPENTELEMETRY_AVAILABLE, "OpenTelemetry not available")
    async def test_trace_span_async(self):
        """Test asynchronous span context manager."""
        initialize_opentelemetry(enable_console_export=True)
        
        async with trace_span_async("test_async_span", attributes={"test.attr": "value"}) as span:
            self.assertIsNotNone(span)
            # Add some work
            await asyncio.sleep(0.01)

    @unittest.skipUnless(OPENTELEMETRY_AVAILABLE, "OpenTelemetry not available")
    def test_trace_span_with_exception(self):
        """Test span context manager with exception handling."""
        initialize_opentelemetry(enable_console_export=True)
        
        with self.assertRaises(ValueError):
            with trace_span("test_span", record_exceptions=True) as span:
                raise ValueError("Test error")


class TestSpanUtilities(unittest.TestCase):
    """Test cases for span utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        shutdown_opentelemetry()

    def tearDown(self):
        """Clean up test fixtures."""
        shutdown_opentelemetry()

    @unittest.skipUnless(OPENTELEMETRY_AVAILABLE, "OpenTelemetry not available")
    def test_add_span_event(self):
        """Test adding events to spans."""
        initialize_opentelemetry(enable_console_export=True)
        
        with trace_span("test_span") as span:
            add_span_event("test_event", {"event.attr": "value"})
            # Should not raise any exceptions

    @unittest.skipUnless(OPENTELEMETRY_AVAILABLE, "OpenTelemetry not available")
    def test_set_span_attribute(self):
        """Test setting span attributes."""
        initialize_opentelemetry(enable_console_export=True)
        
        with trace_span("test_span") as span:
            set_span_attribute("test.attr", "value")
            # Should not raise any exceptions

    @unittest.skipUnless(OPENTELEMETRY_AVAILABLE, "OpenTelemetry not available")
    def test_span_utilities_outside_span(self):
        """Test span utilities when no span is active."""
        initialize_opentelemetry(enable_console_export=True)
        
        # Should not raise exceptions when no span is active
        add_span_event("test_event", {"event.attr": "value"})
        set_span_attribute("test.attr", "value")


class TestMetricsIntegration(unittest.TestCase):
    """Test cases for metrics integration."""

    def setUp(self):
        """Set up test fixtures."""
        shutdown_opentelemetry()

    def tearDown(self):
        """Clean up test fixtures."""
        shutdown_opentelemetry()

    @unittest.skipUnless(OPENTELEMETRY_AVAILABLE, "OpenTelemetry not available")
    def test_sync_metrics_to_opentelemetry(self):
        """Test metrics synchronization to OpenTelemetry."""
        initialize_opentelemetry(enable_console_export=True)
        
        # Create metrics registry and add some metrics
        registry = MetricsRegistry()
        counter = registry.counter("test_counter", "Test counter")
        gauge = registry.gauge("test_gauge", "Test gauge")
        histogram = registry.histogram("test_histogram", "Test histogram")
        
        # Add values
        counter.increment(10)
        gauge.set(100.0)
        histogram.observe(2.5)
        
        # Sync to OpenTelemetry
        sync_metrics_to_opentelemetry(registry)
        
        # Should not raise any exceptions

    @unittest.skipUnless(OPENTELEMETRY_AVAILABLE, "OpenTelemetry not available")
    def test_sync_metrics_without_manager(self):
        """Test metrics sync when OpenTelemetry is not initialized."""
        # Should not raise exceptions when OpenTelemetry is not initialized
        registry = MetricsRegistry()
        sync_metrics_to_opentelemetry(registry)


class TestReliabilityFeaturesIntegration(unittest.TestCase):
    """Test cases for reliability features integration."""

    def setUp(self):
        """Set up test fixtures."""
        shutdown_opentelemetry()

    def tearDown(self):
        """Clean up test fixtures."""
        shutdown_opentelemetry()

    @unittest.skipUnless(OPENTELEMETRY_AVAILABLE, "OpenTelemetry not available")
    def test_trace_with_reliability_features_sync(self):
        """Test tracing with reliability features for sync functions."""
        initialize_opentelemetry(enable_console_export=True)
        
        @trace_with_reliability_features("test_operation")
        def test_function(x: int) -> int:
            return x * 2
        
        result = test_function(5)
        self.assertEqual(result, 10)

    @unittest.skipUnless(OPENTELEMETRY_AVAILABLE, "OpenTelemetry not available")
    async def test_trace_with_reliability_features_async(self):
        """Test tracing with reliability features for async functions."""
        initialize_opentelemetry(enable_console_export=True)
        
        @trace_with_reliability_features("test_async_operation")
        async def test_async_function(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2
        
        result = await test_async_function(5)
        self.assertEqual(result, 10)

    @unittest.skipUnless(OPENTELEMETRY_AVAILABLE, "OpenTelemetry not available")
    def test_trace_with_reliability_features_exception(self):
        """Test tracing with reliability features and exceptions."""
        initialize_opentelemetry(enable_console_export=True)
        
        @trace_with_reliability_features("test_operation")
        def test_function():
            raise ValueError("Test error")
        
        with self.assertRaises(ValueError):
            test_function()


class TestOpenTelemetryNotAvailable(unittest.TestCase):
    """Test behavior when OpenTelemetry is not available."""

    def test_import_without_opentelemetry(self):
        """Test that imports work even without OpenTelemetry."""
        # This should not raise any exceptions
        from semantrix.integrations.opentelemetry import (
            OpenTelemetryNotAvailableError,
            trace_operation,
            trace_span,
            add_span_event,
            set_span_attribute,
        )

    def test_trace_operation_without_opentelemetry(self):
        """Test that tracing decorators work without OpenTelemetry."""
        @trace_operation("test_operation")
        def test_function():
            return "success"
        
        result = test_function()
        self.assertEqual(result, "success")

    def test_trace_span_without_opentelemetry(self):
        """Test that span context managers work without OpenTelemetry."""
        with trace_span("test_span") as span:
            self.assertIsNone(span)

    def test_span_utilities_without_opentelemetry(self):
        """Test that span utilities work without OpenTelemetry."""
        add_span_event("test_event", {"attr": "value"})
        set_span_attribute("test.attr", "value")
        # Should not raise any exceptions


if __name__ == "__main__":
    unittest.main()
