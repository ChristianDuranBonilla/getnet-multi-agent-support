"""FastAPI application exposing the multi-agent system."""

import logging
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from langgraph.types import Command
from pydantic import BaseModel, Field

from app import graph as graph_module
from app import repositories
from app.agents.escalation import operator_message
from app.auth import (
    AuthError,
    Principal,
    authenticated_principal,
    merchant_principal,
    operator_key_configured,
    operator_principal,
)
from app.config import get_settings
from app.observability import (
    configure_logging,
    configure_tracing,
    log_event,
    merchant_ref,
    metrics,
)
from app.rag import retriever as rag
from app.repositories.protocols import ConflictError
from app.state import ChatRequest, ChatResponse, Conversation, Turn, Usage, initial_state

logger = logging.getLogger("getnet.api")
STATIC_DIR = Path(__file__).resolve().parent / "static"
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
MERCHANT_TICKET_FIELDS = (
    "ticket_id",
    "status",
    "priority",
    "summary",
    "created_at",
    "resolved_at",
    "operator_reply",
    "resolution",
)


class TicketReply(BaseModel):
    """Payload of `POST /tickets/{id}/reply`."""

    message: str = Field(min_length=1, max_length=2000)
    resolution: Literal["resolved", "needs_merchant"] = "resolved"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Configure logging and tracing once at startup."""
    configure_logging()
    settings = get_settings()
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    tracing = configure_tracing(app)
    log_event(
        logger,
        "startup",
        model=settings.llm_model,
        tracing=tracing,
        operator_configured=operator_key_configured(),
    )
    if settings.uses_default_secrets:
        log_event(
            logger,
            "development defaults in use; set OPERATOR_API_KEY and LOG_SALT outside development",
            level=logging.WARNING,
        )
    yield


app = FastAPI(title="Getnet Multi-Agent Support", version="0.2.0", lifespan=lifespan)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Attach a request id to every request and log its outcome."""
    supplied = request.headers.get("x-request-id", "")
    request_id = supplied if REQUEST_ID_PATTERN.match(supplied) else uuid.uuid4().hex
    request.state.request_id = request_id
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    if request.url.path not in ("/", "/health", "/ready", "/metrics"):
        metrics.inc("http_requests_total", path=request.url.path, status=response.status_code)
        log_event(
            logger,
            "request",
            request_id=request_id,
            path=request.url.path,
            status=response.status_code,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
    return response


def error_response(status: int, code: str, request: Request, **extra) -> JSONResponse:
    """Build the error envelope shared by every non-2xx response."""
    request_id = getattr(request.state, "request_id", "unknown")
    headers = {}
    retry_after = extra.pop("retry_after", None)
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)
    return JSONResponse(
        status_code=status,
        content={"detail": {"error": code, "request_id": request_id, **extra}},
        headers=headers,
    )


@app.exception_handler(AuthError)
async def auth_error(request: Request, exc: AuthError) -> JSONResponse:
    """Render authentication and authorization failures."""
    return error_response(exc.status, exc.code, request, retry_after=exc.retry_after)


@app.exception_handler(ConflictError)
async def conflict_error(request: Request, exc: ConflictError) -> JSONResponse:
    """Render an invalid ticket state transition as 409."""
    return error_response(409, exc.code, request)


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    """Return a 500 envelope with the request id and never a traceback."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(
        "unhandled error", exc_info=exc, extra={"extra_fields": {"request_id": request_id}}
    )
    return error_response(500, "internal_error", request, message="Internal error")


@app.get("/")
def index() -> FileResponse:
    """Serve the minimal chat UI."""
    return FileResponse(STATIC_DIR / "index.html")


def index_size() -> int:
    """Return the number of indexed chunks, 0 when the index is unavailable."""
    try:
        return rag.get_retriever().count()
    except Exception:
        return 0


def credentials_present() -> bool:
    """Return whether Gemini credentials are configured for this process.

    On Vertex AI that is a service-account file or application default credentials; on
    the Gemini Developer API (`GOOGLE_GENAI_USE_VERTEXAI=false`) it is an API key.
    """
    if not get_settings().google_genai_use_vertexai:
        return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    adc = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    return bool(credentials and Path(credentials).exists()) or adc.exists()


def readiness_checks() -> dict[str, bool]:
    """Evaluate the conditions under which the service can answer a merchant.

    The state directory is probed with a real write, which is why this runs only for
    `/ready`; `/health` stays cheap.
    """
    settings = get_settings()
    try:
        settings.state_dir.mkdir(parents=True, exist_ok=True)
        probe = settings.state_dir / ".write-probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        state_writable = True
    except OSError:
        state_writable = False
    return {
        "index": index_size() > 0,
        "state_dir": state_writable,
        "credentials": credentials_present(),
    }


@app.get("/health")
def health() -> dict:
    """Liveness: service status, index size, model and a cheap readiness indication.

    Unauthenticated on purpose (load balancers call it), so it never reports anything
    about credentials or operators; those go to the startup log.
    """
    chunks = index_size()
    return {
        "status": "ok",
        "index_chunks": chunks,
        "model": get_settings().llm_model,
        "ready": chunks > 0 and credentials_present(),
    }


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics(
    request: Request, principal: Annotated[Principal, Depends(operator_principal)]
) -> str:
    """Operator view: process counters in the Prometheus text format."""
    return metrics.render()


@app.get("/ready")
def ready(request: Request):
    """Readiness: 200 only when the index, the state directory and credentials are in place."""
    checks = readiness_checks()
    if all(checks.values()):
        return {"status": "ready", "checks": checks}
    return JSONResponse(status_code=503, content={"status": "not_ready", "checks": checks})


def _resolve_conversation(payload: ChatRequest, principal: Principal) -> tuple[str, int]:
    conversations = repositories.get_conversation_repository()
    if payload.conversation_id is None:
        return conversations.create(principal.merchant_id), 1
    conversation = conversations.get(payload.conversation_id, principal.merchant_id)
    if conversation is None:
        raise AuthError(404, "conversation_not_found")
    merchant_turns = sum(1 for turn in conversation.turns if turn.role == "merchant")
    return conversation.conversation_id, merchant_turns + 1


def _usage(trace) -> Usage:
    settings = get_settings()
    tokens_in = sum(step.tokens_in for step in trace)
    tokens_out = sum(step.tokens_out for step in trace)
    cost = (
        tokens_in * settings.llm_price_input_per_mtok
        + tokens_out * settings.llm_price_output_per_mtok
    ) / 1_000_000
    return Usage(
        input_tokens=tokens_in, output_tokens=tokens_out, estimated_cost_usd=round(cost, 6)
    )


@app.post("/chat", response_model=ChatResponse)
def chat(
    payload: ChatRequest,
    request: Request,
    principal: Annotated[Principal, Depends(merchant_principal)],
) -> ChatResponse:
    """Run the orchestration graph for one turn of the authenticated merchant.

    Args:
        payload: Message, optional conversation id and the optional `user_id` assertion.
        request: Incoming request, used for the correlation id.
        principal: Authenticated merchant.

    Returns:
        The agent's answer with routing metadata and the execution trace.

    Raises:
        AuthError: 403 when `user_id` is present and differs from the principal.
    """
    if payload.user_id is not None and payload.user_id != principal.merchant_id:
        log_event(
            logger,
            "identity_mismatch",
            level=logging.WARNING,
            request_id=request.state.request_id,
            key_id=principal.key_id,
        )
        raise AuthError(403, "identity_mismatch")
    request_id = request.state.request_id
    started = time.perf_counter()
    conversation_id, turn = _resolve_conversation(payload, principal)
    thread_id = graph_module.new_thread_id(conversation_id, turn)
    state = initial_state(request_id, principal, payload.message, conversation_id, turn, thread_id)
    graph = graph_module.get_graph()
    final = graph.invoke(state, config=graph_module.thread_config(thread_id))
    verdict = final.get("guardrail")
    agent = graph_module.resolved_agent(final)
    route = final.get("route") if agent else None
    trace = [
        step.model_copy(update={"step": i}) for i, step in enumerate(final.get("trace", []), 1)
    ]
    response = ChatResponse(
        request_id=request_id,
        conversation_id=conversation_id,
        turn=turn,
        user_id=principal.merchant_id,
        agent=agent,
        plan=list(final.get("plan") or []) if agent else [],
        hops=int(final.get("hops", 0)) if agent else 0,
        intent=route.intent if route else (verdict.category if verdict else None),
        router_rationale=route.rationale if route else None,
        hop_question=final.get("hop_question") if agent else None,
        guardrail=verdict.category if verdict else None,
        response=final.get("answer") or "",
        confidence=round(float(final.get("confidence", 0.0)), 3) if agent else 0.0,
        handoff=bool(final.get("handoff", False)) if agent else False,
        ticket_id=final.get("ticket_id"),
        sources=final.get("sources", []),
        tool_calls=[call.name for call in final.get("tool_calls", [])],
        security_events=list(final.get("security_events", [])),
        usage=_usage(trace),
        evidence=list(final.get("evidence", [])) if payload.include_evidence else None,
        trace=trace,
        latency_ms=int((time.perf_counter() - started) * 1000),
    )
    metrics.inc("chat_turns_total", agent=response.agent or "guardrail", handoff=response.handoff)
    metrics.inc("chat_latency_ms_sum", response.latency_ms)
    metrics.inc("chat_tokens_total", response.usage.input_tokens, direction="input")
    metrics.inc("chat_tokens_total", response.usage.output_tokens, direction="output")
    for event in response.security_events:
        metrics.inc("security_events_total", event=event.split(":")[0])
    for step in trace:
        if step.model:
            metrics.inc("model_calls_total", node=step.node, model=step.model)
    log_event(
        logger,
        "chat",
        request_id=request_id,
        merchant_ref=merchant_ref(principal.merchant_id),
        key_id=principal.key_id,
        conversation_id=conversation_id,
        turn=turn,
        agent=response.agent,
        plan=response.plan,
        hops=response.hops,
        intent=response.intent,
        confidence=response.confidence,
        handoff=response.handoff,
        ticket_id=response.ticket_id,
        security_events=response.security_events,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_ms=response.latency_ms,
        tool_calls=[call.name for call in final.get("tool_calls", [])],
    )
    return response


@app.get("/conversations/{conversation_id}", response_model=Conversation)
def get_conversation(
    conversation_id: str,
    request: Request,
    principal: Annotated[Principal, Depends(merchant_principal)],
) -> Conversation:
    """Return one of the merchant's conversations with its turns and summary."""
    conversation = repositories.get_conversation_repository().get(
        conversation_id, principal.merchant_id
    )
    if conversation is None:
        raise AuthError(404, "conversation_not_found")
    return conversation


@app.get("/tickets")
def list_tickets(
    request: Request,
    principal: Annotated[Principal, Depends(operator_principal)],
    status: Literal["open", "claimed", "waiting_merchant", "resolved"] | None = None,
) -> list[dict]:
    """Operator view: list tickets, newest first."""
    return repositories.get_ticket_repository().list_tickets(status)


@app.get("/tickets/{ticket_id}")
def get_ticket(
    ticket_id: str,
    request: Request,
    principal: Annotated[Principal, Depends(authenticated_principal)],
) -> dict:
    """Return a ticket: any ticket for an operator, only their own for a merchant."""
    if principal.has_scope("operator"):
        ticket = repositories.get_ticket_repository().get_ticket(ticket_id)
    else:
        ticket = repositories.get_ticket_repository().get_ticket(ticket_id, principal.merchant_id)
        if ticket is not None:
            ticket = {field: ticket.get(field) for field in MERCHANT_TICKET_FIELDS}
    if ticket is None:
        raise AuthError(404, "ticket_not_found")
    return ticket


@app.post("/tickets/{ticket_id}/claim")
def claim_ticket(
    ticket_id: str,
    request: Request,
    principal: Annotated[Principal, Depends(operator_principal)],
) -> dict:
    """Assign the ticket to the calling operator."""
    try:
        return repositories.get_ticket_repository().claim(ticket_id, principal.key_id)
    except LookupError as exc:
        raise AuthError(404, "ticket_not_found") from exc


@app.post("/tickets/{ticket_id}/reply")
def reply_ticket(
    ticket_id: str,
    payload: TicketReply,
    request: Request,
    principal: Annotated[Principal, Depends(operator_principal)],
) -> dict:
    """Store the operator's reply and resume the merchant's turn.

    The interrupted graph thread of the escalated turn is resumed with the reply, which
    appends an operator turn to the conversation. When the thread has already finished
    (a second reply after `needs_merchant`, or a lost checkpoint) the reply is appended
    to the conversation directly so the merchant still sees it.
    """
    tickets = repositories.get_ticket_repository()
    try:
        ticket = tickets.reply(ticket_id, principal.key_id, payload.message, payload.resolution)
    except LookupError as exc:
        raise AuthError(404, "ticket_not_found") from exc
    resumed = resume_turn(ticket, payload.message, payload.resolution)
    log_event(
        logger,
        "ticket_reply",
        request_id=request.state.request_id,
        ticket_id=ticket_id,
        key_id=principal.key_id,
        merchant_ref=merchant_ref(ticket["merchant_id"]),
        resumed=resumed,
    )
    return {**ticket, "resumed": resumed}


def resume_turn(ticket: dict, message: str, resolution: str) -> bool:
    """Resume the escalated turn with the operator's reply.

    Args:
        ticket: Ticket row with `thread_id`, `conversation_id` and `merchant_id`.
        message: Operator's message.
        resolution: Operator's resolution code.

    Returns:
        Whether the graph thread was resumed; `False` when the reply was stored directly.
    """
    config = {"configurable": {"thread_id": ticket["thread_id"]}}
    graph = graph_module.get_graph()
    try:
        if graph_module.awaiting_operator(graph, config):
            graph.invoke(
                Command(resume={"message": message, "resolution": resolution}), config=config
            )
            return True
        log_event(
            logger,
            "resume_skipped",
            ticket_id=ticket["ticket_id"],
            reason="thread not waiting for an operator",
        )
    except Exception as exc:
        log_event(
            logger,
            "resume_failed",
            level=logging.WARNING,
            ticket_id=ticket["ticket_id"],
            error=type(exc).__name__,
        )
    conversations = repositories.get_conversation_repository()
    conversation = conversations.get(ticket["conversation_id"], ticket["merchant_id"])
    language = conversation_language(conversation) if conversation else "en"
    conversations.append(
        ticket["conversation_id"],
        Turn(
            role="operator",
            content=operator_message(language, message),
            agent="operator",
            created_at=datetime.now(UTC),
        ),
    )
    return False


def conversation_language(conversation) -> str:
    """Guess the conversation language from the stored handoff notice, defaulting to English."""
    for turn in reversed(conversation.turns):
        if turn.role == "assistant" and "chamado" in turn.content:
            return "pt"
        if turn.role == "assistant" and "ticket" in turn.content and "Abrí" in turn.content:
            return "es"
    return "en"
