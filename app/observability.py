"""Structured logging, per-node timing and token accounting, optional tracing."""

import hashlib
import json
import logging
import sys
import threading
import time
from collections.abc import Callable
from contextlib import nullcontext
from datetime import UTC, datetime
from functools import wraps

from app.config import get_settings
from app.state import AgentState, TraceStep

_tracer = None


class JsonFormatter(logging.Formatter):
    """Render each log record as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        """Serialise the record, merging any `extra` fields.

        Args:
            record: The log record.

        Returns:
            A JSON string without newlines.
        """
        payload = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging() -> None:
    """Install the JSON formatter on the root logger."""
    root = logging.getLogger()
    if not any(isinstance(existing.formatter, JsonFormatter) for existing in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        root.addHandler(handler)
    root.setLevel(get_settings().log_level.upper())
    logging.getLogger("uvicorn.access").disabled = True
    # Third-party clients log one line per HTTP call; the request log already covers it.
    for noisy in ("httpx", "httpx2", "google_genai", "google_genai.models", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_event(logger: logging.Logger, message: str, *, level: int = logging.INFO, **fields) -> None:
    """Log a message with structured fields.

    Args:
        logger: Target logger.
        message: Human-readable event name.
        level: Logging level; INFO by default.
        **fields: Key-value pairs added to the JSON line.
    """
    logger.log(level, message, extra={"extra_fields": fields})


def merchant_ref(merchant_id: str) -> str:
    """Return the pseudonymous reference under which a merchant appears in logs.

    Args:
        merchant_id: Real merchant identifier; never written to logs itself.

    Returns:
        Twelve hex characters of a salted SHA-256 digest.
    """
    salt = get_settings().log_salt
    return hashlib.sha256(f"{salt}:{merchant_id}".encode()).hexdigest()[:12]


def usage_of(message) -> tuple[int, int]:
    """Read the token usage reported with a model response.

    Args:
        message: AIMessage returned by the model.

    Returns:
        A tuple of (input_tokens, output_tokens); zeros when the model did not report.
    """
    usage = getattr(message, "usage_metadata", None) or {}
    return int(usage.get("input_tokens", 0) or 0), int(usage.get("output_tokens", 0) or 0)


def model_of(message) -> str:
    """Return the model id reported with a model response, or an empty string."""
    metadata = getattr(message, "response_metadata", None) or {}
    return str(metadata.get("model_name") or "")


class Metrics:
    """In-process counters exposed at `/metrics` in the Prometheus text format.

    Counters are labelled by low-cardinality values only (status, agent, event); no
    merchant identifier ever becomes a label.
    """

    def __init__(self) -> None:
        """Start with empty counters."""
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}

    def inc(self, name: str, value: float = 1.0, **labels: str) -> None:
        """Add `value` to the counter `name` with the given labels."""
        key = (name, tuple(sorted((k, str(v)) for k, v in labels.items())))
        with self._lock:
            self._counters[key] = self._counters.get(key, 0.0) + value

    def render(self) -> str:
        """Return every counter as Prometheus exposition lines."""
        lines = []
        with self._lock:
            items = sorted(self._counters.items())
        for (name, labels), value in items:
            label_text = ",".join(f'{k}="{v}"' for k, v in labels)
            suffix = f"{{{label_text}}}" if label_text else ""
            lines.append(f"{name}{suffix} {value:g}")
        return "\n".join(lines) + "\n"


metrics = Metrics()


def configure_tracing(app) -> bool:
    """Enable OpenTelemetry export when an OTLP endpoint is configured.

    The OpenTelemetry packages are imported only here, so the `otel` extra stays
    optional and nothing is loaded when the endpoint is unset.

    Args:
        app: The FastAPI application to instrument.

    Returns:
        Whether tracing was enabled.
    """
    global _tracer
    endpoint = get_settings().otel_exporter_otlp_endpoint
    if not endpoint:
        return False
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    provider = TracerProvider(resource=Resource.create({"service.name": "getnet-agents"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
    _tracer = trace.get_tracer("getnet.graph")
    return True


def _span(name: str):
    return _tracer.start_as_current_span(f"node.{name}") if _tracer else nullcontext()


def timed_node(name: str) -> Callable:
    """Wrap a graph node so that its duration and token usage are appended to the trace.

    Nodes report tokens by returning `tokens_in` and `tokens_out` in their update, and the
    model that answered as `model`; all three are moved into the trace step so that the API
    can sum tokens per request and show which model served each node.

    Args:
        name: Node name as it should appear in the trace.

    Returns:
        A decorator for node functions that take and return an `AgentState` update.
    """

    def decorator(func: Callable[[AgentState], dict]) -> Callable[[AgentState], dict]:
        @wraps(func)
        def wrapper(state: AgentState) -> dict:
            started = time.perf_counter()
            with _span(name) as span:
                update = func(state) or {}
                elapsed = int((time.perf_counter() - started) * 1000)
                detail = str(update.pop("trace_detail", ""))
                tokens_in = int(update.pop("tokens_in", 0))
                tokens_out = int(update.pop("tokens_out", 0))
                model = str(update.pop("model", ""))
                if span is not None and hasattr(span, "set_attribute"):
                    span.set_attribute("node.ms", elapsed)
                    span.set_attribute("node.detail", detail)
                    span.set_attribute("node.tokens_in", tokens_in)
                    span.set_attribute("node.tokens_out", tokens_out)
                    span.set_attribute("node.model", model)
            # `trace` uses an additive reducer, so each node contributes only its own
            # step; steps are numbered once the graph has finished.
            update["trace"] = [
                TraceStep(
                    step=0,
                    node=name,
                    ms=elapsed,
                    detail=detail,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    model=model,
                )
            ]
            return update

        return wrapper

    return decorator
