"""Replanner: decides what happens after a specialist has run.

The policy is deterministic code, not a model call: it adds no latency, it is tested
exhaustively, and the model never chooses a policy outcome. It observes the state the
specialist left behind (handoff, hop request, remaining plan) and picks the next node.
"""

from app import repositories
from app.config import get_settings
from app.observability import timed_node
from app.state import AgentState, PartialAnswer

SPECIALIST_NAMES = ("knowledge", "support")
HOP_TOOL_NAME = "request_product_knowledge"
BLOCKING_ACCOUNT_STATUSES = ("under_review", "blocked", "suspended")


def data_policy_reason(state: AgentState) -> str | None:
    """Return why the merchant's data requires a human, or `None`.

    The Support agent only reports what the tools returned; whether a failed lookup or
    an account under review must go to a person is decided here, next to the other
    policies, so the model never owns that decision.
    """
    for call in state.get("tool_calls", []):
        if call.name == HOP_TOOL_NAME:
            continue
        if not call.ok:
            return f"tool {call.name} failed: {call.summary}"
        if call.account_status in BLOCKING_ACCOUNT_STATUSES:
            return f"account {call.account_status.replace('_', ' ')}"
    return None


def resolved_by_operator(reason: str, tickets: list[dict]) -> bool:
    """Tell whether an operator already resolved a ticket for this cause in the conversation."""
    return any(
        ticket.get("reason") == reason and ticket.get("status") == "resolved" for ticket in tickets
    )


def decide(state: AgentState, tickets: list[dict] | None = None) -> tuple[str, str]:
    """Apply the replanning rules in order.

    1. A handoff requested by a node (model error, exhausted loop, empty answer) goes to
       the ticket node.
    2. A failed merchant-data lookup or an account under review goes to a human, unless
       an operator already resolved a ticket for the same cause in this conversation: a
       human decision stands over the automatic trigger, and the ticket node re-uses a
       ticket that is still open instead of opening a second one.
    3. A Knowledge answer with no evidence behind it (no retrieved chunk, no web
       grounding) is an answer from the model's memory: it is escalated instead of sent.
    4. A hop requested at runtime by the specialist runs the other specialist.
    5. A remaining step in the Router's plan runs next.
    6. Otherwise the turn is complete.

    Hops 4 and 5 are only taken while `hops < MAX_HOPS`.

    Args:
        state: State after a specialist ran.
        tickets: Tickets of the conversation, for rule 2; `None` when there are none.

    Returns:
        The next node (`"ticket"`, a specialist name, or `"finalizer"`) and, for a
        ticket, the reason.
    """
    if state.get("handoff"):
        return "ticket", state.get("handoff_reason") or "handoff requested"
    reason = data_policy_reason(state) if state.get("current_agent") == "support" else None
    if reason and not resolved_by_operator(reason, tickets or []):
        return "ticket", reason
    if state.get("current_agent") == "knowledge" and not knowledge_evidence(state):
        return "ticket", "answer without sources"
    hops = int(state.get("hops", 0))
    if hops >= get_settings().max_hops:
        return "finalizer", ""
    current = state.get("current_agent")
    request = state.get("hop_request") or {}
    target = request.get("agent")
    if target in SPECIALIST_NAMES and target != current:
        return target, ""
    plan = list(state.get("plan") or [])
    position = hops + 1
    if position < len(plan) and plan[position] in SPECIALIST_NAMES and plan[position] != current:
        return plan[position], ""
    return "finalizer", ""


def next_step(state: AgentState) -> str:
    """Return only the next node name (see `decide`)."""
    return decide(state)[0]


def knowledge_evidence(state: AgentState) -> bool:
    """Return whether the turn holds retrieval or web evidence for a Knowledge answer."""
    return any(item.type in ("rag", "web") for item in state.get("evidence", []))


def after_replan(state: AgentState) -> str:
    """Conditional edge: follow the decision stored by `replan_node`."""
    return state.get("next_node") or "finalizer"


@timed_node("replan")
def replan_node(state: AgentState) -> dict:
    """Record the next node and, on a hop, move the current answer aside.

    Args:
        state: State after a specialist ran.

    Returns:
        State update with `next_node` and, for a hop, `hops`, `partial_answers`,
        `hop_question` and cleared `answer`, `sources` and `hop_request`.
    """
    conversation_id = state.get("conversation_id") or ""
    tickets = (
        repositories.get_ticket_repository().for_conversation(conversation_id)
        if conversation_id
        else []
    )
    decision, reason = decide(state, tickets)
    if decision not in SPECIALIST_NAMES:
        update = {"next_node": decision, "hop_request": None, "trace_detail": decision}
        if decision == "ticket" and not state.get("handoff"):
            update.update(handoff=True, handoff_reason=reason, trace_detail=f"ticket ({reason})")
        return update
    request = state.get("hop_request") or {}
    partial = PartialAnswer(
        agent=state.get("current_agent") or "unknown",
        answer=state.get("answer") or "",
        tool_summaries=[
            f"{call.name}: {call.summary}"
            for call in state.get("tool_calls", [])
            if call.name != HOP_TOOL_NAME
        ],
        sources=list(state.get("sources", [])),
    )
    return {
        "next_node": decision,
        "hops": int(state.get("hops", 0)) + 1,
        "partial_answers": [*state.get("partial_answers", []), partial],
        "hop_question": request.get("question") or None,
        "hop_request": None,
        "answer": None,
        "sources": [],
        "trace_detail": f"hop->{decision}",
    }
