"""Structured logging and per-node timing."""

import json
import logging
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from functools import wraps

from app.config import get_settings
from app.state import AgentState, TraceStep


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
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(get_settings().log_level.upper())
    logging.getLogger("uvicorn.access").disabled = True
    # Third-party clients log one line per HTTP call; the request log already covers it.
    for noisy in ("httpx", "httpx2", "google_genai", "google_genai.models", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_event(logger: logging.Logger, message: str, **fields) -> None:
    """Log a message with structured fields.

    Args:
        logger: Target logger.
        message: Human-readable event name.
        **fields: Key-value pairs added to the JSON line.
    """
    logger.info(message, extra={"extra_fields": fields})


def timed_node(name: str) -> Callable:
    """Wrap a graph node so that its duration is appended to the trace.

    Args:
        name: Node name as it should appear in the trace.

    Returns:
        A decorator for node functions that take and return an `AgentState` update.
    """

    def decorator(func: Callable[[AgentState], dict]) -> Callable[[AgentState], dict]:
        @wraps(func)
        def wrapper(state: AgentState) -> dict:
            started = time.perf_counter()
            update = func(state) or {}
            elapsed = int((time.perf_counter() - started) * 1000)
            detail = str(update.pop("trace_detail", ""))
            # `trace` uses an additive reducer, so each node contributes only its own
            # step; steps are numbered once the graph has finished.
            update["trace"] = [TraceStep(step=0, node=name, ms=elapsed, detail=detail)]
            return update

        return wrapper

    return decorator
