"""
Distributed Tracing with OpenTelemetry
Enables end-to-end request tracing across services
"""

import logging
from typing import Optional
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

logger = logging.getLogger(__name__)


class TracingManager:
    """
    OpenTelemetry tracing manager

    Features:
    - Automatic request tracing
    - Custom span creation
    - Context propagation
    - Export to Jaeger/Zipkin/etc.
    """

    def __init__(
        self,
        service_name: str = "transparency-api",
        otlp_endpoint: Optional[str] = None,
        enable_console: bool = False
    ):
        """
        Initialize tracing

        Args:
            service_name: Service name for traces
            otlp_endpoint: OTLP collector endpoint (e.g., "http://localhost:4317")
            enable_console: Enable console exporter for debugging
        """
        self.service_name = service_name
        self.otlp_endpoint = otlp_endpoint
        self.enable_console = enable_console
        self.tracer_provider: Optional[TracerProvider] = None
        self.tracer: Optional[trace.Tracer] = None

    def setup(self):
        """Setup OpenTelemetry tracing"""
        # Create resource with service name
        resource = Resource.create({
            SERVICE_NAME: self.service_name,
            "service.version": "1.0.0",
            "environment": "production"
        })

        # Create tracer provider
        self.tracer_provider = TracerProvider(resource=resource)

        # Add console exporter (for debugging)
        if self.enable_console:
            console_exporter = ConsoleSpanExporter()
            self.tracer_provider.add_span_processor(
                BatchSpanProcessor(console_exporter)
            )
            logger.info("Console span exporter enabled")

        # Add OTLP exporter (for production)
        if self.otlp_endpoint:
            try:
                otlp_exporter = OTLPSpanExporter(endpoint=self.otlp_endpoint)
                self.tracer_provider.add_span_processor(
                    BatchSpanProcessor(otlp_exporter)
                )
                logger.info(f"OTLP exporter configured: {self.otlp_endpoint}")
            except Exception as e:
                logger.error(f"Failed to setup OTLP exporter: {e}")

        # Set global tracer provider
        trace.set_tracer_provider(self.tracer_provider)

        # Get tracer
        self.tracer = trace.get_tracer(__name__)

        logger.info(f"Tracing initialized for service: {self.service_name}")

    def instrument_fastapi(self, app):
        """Instrument FastAPI app for automatic tracing"""
        if not self.tracer_provider:
            logger.warning("Tracer not initialized, call setup() first")
            return

        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI instrumented for tracing")

    def create_span(self, name: str, attributes: Optional[dict] = None):
        """
        Create a custom span

        Usage:
            with tracing_manager.create_span("database_query", {"table": "trades"}):
                # your code here
                pass
        """
        if not self.tracer:
            logger.warning("Tracer not initialized")
            return DummySpan()

        span = self.tracer.start_span(name)
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        return span

    def shutdown(self):
        """Shutdown tracing and flush spans"""
        if self.tracer_provider:
            self.tracer_provider.shutdown()
            logger.info("Tracing shutdown complete")


class DummySpan:
    """Dummy span for when tracing is not initialized"""
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def set_attribute(self, key, value):
        pass


# Global tracing instance
_tracing_manager: Optional[TracingManager] = None


def get_tracing_manager() -> TracingManager:
    """Get global tracing manager"""
    global _tracing_manager
    if _tracing_manager is None:
        from src.security.secrets_manager import get_secrets_manager
        secrets = get_secrets_manager()

        _tracing_manager = TracingManager(
            service_name="transparency-api",
            otlp_endpoint=secrets.get_secret("OTLP_ENDPOINT"),
            enable_console=secrets.is_development()
        )
        _tracing_manager.setup()

    return _tracing_manager


def setup_tracing(app):
    """Setup tracing for FastAPI app"""
    tracing = get_tracing_manager()
    tracing.instrument_fastapi(app)
