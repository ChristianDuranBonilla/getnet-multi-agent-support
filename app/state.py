"""Graph state and the data models shared by the API and the agents."""

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field

AgentName = Literal["knowledge", "support"]
Intent = Literal[
    "product_info", "general_question", "account_or_transaction", "device_issue", "other"
]
GuardrailCategory = Literal["ok", "off_topic", "pii", "unsafe", "prompt_injection"]


class GuardrailResult(BaseModel):
    """Outcome of the input guardrail."""

    allowed: bool
    category: GuardrailCategory
    language: str = Field(default="en", max_length=8)
    reason: str = Field(default="", max_length=200)


class RouterDecision(BaseModel):
    """Routing decision produced by the Router agent."""

    agent: AgentName
    intent: Intent
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(default="", max_length=200)


class Source(BaseModel):
    """Reference that grounds an answer."""

    type: Literal["rag", "web", "tool"]
    ref: str


class ToolCall(BaseModel):
    """Record of one tool invocation made by the Support agent."""

    name: str
    args: dict
    ok: bool
    summary: str = ""


class TraceStep(BaseModel):
    """Timing record of one graph node."""

    step: int
    node: str
    ms: int
    detail: str = ""


class ChatRequest(BaseModel):
    """Payload accepted by `POST /chat`."""

    message: str = Field(min_length=1, max_length=2000)
    user_id: str = Field(min_length=1, max_length=64)


class ChatResponse(BaseModel):
    """Payload returned by `POST /chat`."""

    request_id: str
    user_id: str
    agent: str | None
    intent: str | None
    guardrail: str | None
    response: str
    confidence: float
    handoff: bool
    ticket_id: str | None
    sources: list[Source]
    trace: list[TraceStep]
    latency_ms: int


class AgentState(TypedDict, total=False):
    """Shared state that flows through the LangGraph nodes."""

    request_id: str
    user_id: str
    message: str
    language: str
    pii_detected: bool
    guardrail: GuardrailResult | None
    route: RouterDecision | None
    answer: str | None
    sources: list[Source]
    tool_calls: list[ToolCall]
    confidence: float
    handoff: bool
    handoff_reason: str
    ticket_id: str | None
    trace: Annotated[list[TraceStep], operator.add]


def initial_state(request_id: str, user_id: str, message: str) -> AgentState:
    """Build the state for a new request.

    Args:
        request_id: Correlation id assigned by the API middleware.
        user_id: Merchant identifier sent by the client.
        message: Raw user message.

    Returns:
        A fully initialised state with empty collections.
    """
    return AgentState(
        request_id=request_id,
        user_id=user_id,
        message=message,
        language="en",
        pii_detected=False,
        guardrail=None,
        route=None,
        answer=None,
        sources=[],
        tool_calls=[],
        confidence=0.0,
        handoff=False,
        handoff_reason="",
        ticket_id=None,
        trace=[],
    )
