"""OpenTelemetry instrumentation — opt-in via OTEL_ENABLED.

Imports are local so the project runs without otel deps installed.
"""
from __future__ import annotations

from core.config import get_settings
from core.logger import get_logger

logger = get_logger(__name__)

_TRACING_INSTALLED = False


def init_tracing(service_name: str) -> None:
    """Configure tracer provider + OTLP exporter when enabled."""
    global _TRACING_INSTALLED
    s = get_settings()
    if not s.otel_enabled or _TRACING_INSTALLED:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning(
            "otel_packages_missing",
            hint="install with: pip install opentelemetry-sdk opentelemetry-exporter-otlp opentelemetry-instrumentation-fastapi opentelemetry-instrumentation-httpx",
        )
        return

    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=s.otel_exporter_otlp_endpoint) if s.otel_exporter_otlp_endpoint else OTLPSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _TRACING_INSTALLED = True
    logger.info("otel_tracing_initialized", endpoint=s.otel_exporter_otlp_endpoint or "default")


def instrument_app(app) -> None:
    """Wrap FastAPI + httpx with OTel instrumentation when available."""
    s = get_settings()
    if not s.otel_enabled:
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    except ImportError:
        return
    FastAPIInstrumentor.instrument_app(app)
    HTTPXClientInstrumentor().instrument()
