"""Graph state and the data models shared by the API and the agents."""

import operator
from datetime import datetime
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field, model_validator

from app.auth import Principal

AgentName = Literal["knowledge", "support"]
Intent = Literal[
    "product_info", "general_question", "account_or_transaction", "device_issue", "other"
]
GuardrailCategory = Literal["ok", "off_topic", "pii", "unsafe", "prompt_injection", "unavailable"]
TurnRole = Literal["merchant", "assistant", "operator"]


class GuardrailResult(BaseModel):
    """Outcome of the input guardrail."""

    allowed: bool
    category: GuardrailCategory
    language: str = Field(default="en", max_length=8)
    reason: str = Field(default="", max_length=200)


class RouterDecision(BaseModel):
    """Routing decision produced by the Router agent.

    `plan` lists the specialists to run in order. It has one element for an ordinary
    question and two when the request needs both the merchant's data and product
    knowledge; the first element always equals `agent`.
    """

    agent: AgentName
    intent: Intent
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(default="", max_length=200)
    plan: list[AgentName] = Field(default_factory=list, max_length=2)

    @model_validator(mode="after")
    def _normalise_plan(self) -> "RouterDecision":
        if not self.plan or self.plan[0] != self.agent:
            self.plan = [self.agent, *[step for step in self.plan if step != self.agent]][:2]
        if len(self.plan) == 2 and self.plan[0] == self.plan[1]:
            self.plan = self.plan[:1]
        return self


class Source(BaseModel):
    """Reference that grounds an answer."""

    type: Literal["rag", "web", "tool", "operator"]
    ref: str
    title: str | None = None
    """Display name of a web result (its domain), when the grounding metadata gives one."""


class Evidence(BaseModel):
    """Material the answer was produced from, returned on request for evaluation."""

    type: Literal["rag", "web", "tool", "operator"]
    ref: str
    excerpt: str = Field(default="", max_length=600)
    distance: float | None = None


class ToolCall(BaseModel):
    """Record of one tool invocation made by the Support agent."""

    name: str
    args: dict
    ok: bool
    summary: str = ""
    account_status: str | None = None
    """Account status reported by the tool, when it reported one (policy input)."""


class TraceStep(BaseModel):
    """Timing record of one graph node."""

    step: int
    node: str
    ms: int
    detail: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""


class Usage(BaseModel):
    """Token consumption and estimated cost of one request."""

    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0


class Turn(BaseModel):
    """One message of a conversation."""

    role: TurnRole
    content: str
    agent: str | None = None
    created_at: datetime


class Conversation(BaseModel):
    """A merchant-owned sequence of turns with a rolling summary."""

    conversation_id: str
    merchant_id: str
    turns: list[Turn] = Field(default_factory=list)
    summary: str = ""
    summary_through_turn: int = 0
    created_at: datetime


class ConversationSummary(BaseModel):
    """Structured output of the rolling-summary call."""

    summary: str = Field(max_length=900)


class PartialAnswer(BaseModel):
    """Answer produced by a specialist before a hop to another one."""

    agent: str
    answer: str
    tool_summaries: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)


class ChatRequest(BaseModel):
    """Payload accepted by `POST /chat`.

    `user_id` is kept for compatibility with the challenge contract. It is an assertion
    that must match the authenticated merchant, never the source of identity.
    """

    message: str = Field(min_length=1, max_length=2000)
    user_id: str | None = Field(default=None, min_length=1, max_length=64)
    conversation_id: str | None = Field(default=None, min_length=1, max_length=64)
    include_evidence: bool = False


class ChatResponse(BaseModel):
    """Payload returned by `POST /chat`."""

    request_id: str
    conversation_id: str
    turn: int
    user_id: str
    agent: str | None
    plan: list[str]
    hops: int
    intent: str | None
    router_rationale: str | None = None
    hop_question: str | None = None
    guardrail: str | None
    response: str
    confidence: float
    handoff: bool
    ticket_id: str | None
    sources: list[Source]
    tool_calls: list[str]
    security_events: list[str]
    usage: Usage
    evidence: list[Evidence] | None = None
    trace: list[TraceStep]
    latency_ms: int


class AgentState(TypedDict, total=False):
    """Shared state that flows through the LangGraph nodes during one turn.

    Durable conversation memory lives in the conversation repository; `history` and
    `summary` are loaded into the state by the `recall` node at the start of a turn.
    """

    request_id: str
    principal: Principal
    conversation_id: str
    turn: int
    thread_id: str
    message: str
    language: str
    history: list[Turn]
    summary: str
    pii_detected: bool
    guardrail: GuardrailResult | None
    route: RouterDecision | None
    plan: list[str]
    current_agent: str | None
    next_node: str
    hops: int
    hop_request: dict | None
    hop_question: str | None
    partial_answers: list[PartialAnswer]
    answer: str | None
    sources: list[Source]
    evidence: Annotated[list[Evidence], operator.add]
    tool_calls: list[ToolCall]
    confidence: float
    handoff: bool
    handoff_reason: str
    ticket_id: str | None
    ticket_reused: bool
    operator_reply: str | None
    security_events: Annotated[list[str], operator.add]
    trace: Annotated[list[TraceStep], operator.add]


def initial_state(
    request_id: str,
    principal: Principal,
    message: str,
    conversation_id: str = "",
    turn: int = 1,
    thread_id: str = "",
) -> AgentState:
    """Build the state for a new turn.

    Every per-turn field is reset here so that nothing leaks between turns; the
    conversation memory is loaded by the `recall` node.

    Args:
        request_id: Correlation id assigned by the API middleware.
        principal: Authenticated merchant.
        message: Raw user message.
        conversation_id: Conversation this turn belongs to.
        turn: 1-based index of the turn within the conversation.
        thread_id: Checkpoint thread of this turn; defaults to `conversation_id:turn`.

    Returns:
        A fully initialised state with empty collections.
    """
    return AgentState(
        request_id=request_id,
        principal=principal,
        conversation_id=conversation_id,
        turn=turn,
        thread_id=thread_id or f"{conversation_id}:{turn}",
        message=message,
        language="en",
        history=[],
        summary="",
        pii_detected=False,
        guardrail=None,
        route=None,
        plan=[],
        current_agent=None,
        next_node="finalizer",
        hops=0,
        hop_request=None,
        hop_question=None,
        partial_answers=[],
        answer=None,
        sources=[],
        evidence=[],
        tool_calls=[],
        confidence=0.0,
        handoff=False,
        handoff_reason="",
        ticket_id=None,
        operator_reply=None,
        security_events=[],
        trace=[],
    )
