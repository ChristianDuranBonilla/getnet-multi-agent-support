"""FastAPI application exposing the multi-agent system."""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from app import graph as graph_module
from app.config import get_settings
from app.observability import configure_logging, log_event
from app.rag import retriever as rag
from app.state import ChatRequest, ChatResponse, initial_state

logger = logging.getLogger("getnet.api")
STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Configure logging once at startup."""
    configure_logging()
    log_event(logger, "startup", model=get_settings().llm_model)
    yield


app = FastAPI(title="Getnet Multi-Agent Support", version="0.1.0", lifespan=lifespan)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a request id to every request and log its outcome."""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    request.state.request_id = request_id
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    if request.url.path not in ("/", "/health"):
        log_event(
            logger,
            "request",
            request_id=request_id,
            path=request.url.path,
            status=response.status_code,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
    return response


@app.get("/")
def index() -> FileResponse:
    """Serve the minimal chat UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    """Report service status, index size and the configured model."""
    try:
        chunks = rag.get_retriever().count()
    except Exception:
        chunks = 0
    return {"status": "ok", "index_chunks": chunks, "model": get_settings().llm_model}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    """Run the orchestration graph for one message.

    Args:
        payload: Message and merchant id.
        request: Incoming request, used for the correlation id.

    Returns:
        The agent's answer with routing metadata and the execution trace.
    """
    request_id = request.state.request_id
    started = time.perf_counter()
    state = initial_state(request_id, payload.user_id, payload.message)
    final = graph_module.get_graph().invoke(state)
    verdict = final.get("guardrail")
    agent = graph_module.resolved_agent(final)
    route = final.get("route") if agent else None
    trace = [
        step.model_copy(update={"step": i}) for i, step in enumerate(final.get("trace", []), 1)
    ]
    response = ChatResponse(
        request_id=request_id,
        user_id=payload.user_id,
        agent=agent,
        intent=route.intent if route else (verdict.category if verdict else None),
        guardrail=verdict.category if verdict else None,
        response=final.get("answer") or "",
        confidence=round(float(final.get("confidence", 0.0)), 3) if agent else 0.0,
        handoff=bool(final.get("handoff", False)) if agent else False,
        ticket_id=final.get("ticket_id"),
        sources=final.get("sources", []),
        trace=trace,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
    log_event(
        logger,
        "chat",
        request_id=request_id,
        user_id=payload.user_id,
        agent=response.agent,
        intent=response.intent,
        confidence=response.confidence,
        handoff=response.handoff,
        ticket_id=response.ticket_id,
        latency_ms=response.latency_ms,
        tool_calls=[call.name for call in final.get("tool_calls", [])],
    )
    return response


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    """Return a 500 envelope with the request id and never a traceback."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        "unhandled error", exc_info=exc, extra={"extra_fields": {"request_id": request_id}}
    )
    return JSONResponse(
        status_code=500,
        content={"detail": {"message": "Internal error", "request_id": request_id}},
    )
