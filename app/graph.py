"""LangGraph wiring of the multi-agent orchestration.

Flow of one turn:

    recall -> sanitize -> (guardrail || router) -> dispatch -> specialist -> replan
    replan -> specialist (one bounded hop) | ticket -> finalizer | finalizer
    finalizer -> remember -> [await_operator] -> END

The Guardrail classifier and the Router are independent model calls over the same
sanitized message, so they run in one parallel superstep. `await_operator` interrupts
the graph when a ticket was opened; the operator's reply resumes the same thread.
"""

import secrets
import sqlite3
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.agents.escalation import await_operator_node, ticket_node
from app.agents.finalizer import finalizer_node
from app.agents.guardrail import BLOCKED_CATEGORIES, guardrail_node, sanitize_node
from app.agents.knowledge import knowledge_node
from app.agents.memory import recall_node, remember_node
from app.agents.replan import after_replan, replan_node
from app.agents.router import router_node
from app.agents.support import support_node
from app.config import get_settings
from app.state import AgentState

SPECIALISTS = {"knowledge": knowledge_node, "support": support_node}
CLASSIFIERS = ["guardrail", "router"]
BLOCKING_CATEGORIES = BLOCKED_CATEGORIES


def after_sanitize(state: AgentState) -> str | list[str]:
    """Skip the model calls for deterministic blocks, otherwise fan out to both classifiers."""
    return "finalizer" if state.get("answer") else CLASSIFIERS


def after_dispatch(state: AgentState) -> str:
    """Stop when the guardrail produced a reply, otherwise follow the Router."""
    if state.get("answer"):
        return "finalizer"
    route = state.get("route")
    return route.agent if route and route.agent in SPECIALISTS else "knowledge"


def after_remember(state: AgentState) -> str:
    """Wait for an operator when a ticket was opened in this turn and nobody answered yet.

    A turn that pointed the merchant to a ticket already open ends normally: the
    operator's reply will resume the turn that opened that ticket.
    """
    if (
        state.get("ticket_id")
        and state.get("operator_reply") is None
        and not state.get("ticket_reused")
    ):
        return "await_operator"
    return END


def dispatch_node(state: AgentState) -> dict:
    """Join point for the parallel classifiers; carries no logic of its own."""
    return {}


def guardrail_blocked(state: AgentState) -> bool:
    """Return whether the guardrail answered on behalf of the agents."""
    verdict = state.get("guardrail")
    return bool(verdict and verdict.category in BLOCKING_CATEGORIES)


def resolved_agent(state: AgentState) -> str | None:
    """Return the agent that produced the answer, or `None` when the guardrail did.

    The Router runs in parallel with the Guardrail, so a routing decision exists even
    for blocked messages; it is reported only when a specialist actually ran.
    """
    if guardrail_blocked(state):
        return None
    if state.get("current_agent"):
        return state["current_agent"]
    route = state.get("route")
    return route.agent if route else None


def build_graph(checkpointer=None):
    """Assemble and compile the orchestration graph.

    Args:
        checkpointer: LangGraph checkpointer. Without one the graph still runs every
            node, but the operator wait is skipped because `interrupt` needs a
            checkpoint to resume from.

    Returns:
        A compiled LangGraph application that accepts an `AgentState`.
    """
    graph = StateGraph(AgentState)
    graph.add_node("recall", recall_node)
    graph.add_node("sanitize", sanitize_node)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("router", router_node)
    graph.add_node("dispatch", dispatch_node)
    for name, node in SPECIALISTS.items():
        graph.add_node(name, node)
    graph.add_node("replan", replan_node)
    graph.add_node("ticket", ticket_node)
    graph.add_node("finalizer", finalizer_node)
    graph.add_node("remember", remember_node)

    graph.add_edge(START, "recall")
    graph.add_edge("recall", "sanitize")
    graph.add_conditional_edges(
        "sanitize", after_sanitize, {"finalizer": "finalizer", **{n: n for n in CLASSIFIERS}}
    )
    for name in CLASSIFIERS:
        graph.add_edge(name, "dispatch")
    graph.add_conditional_edges(
        "dispatch", after_dispatch, {"finalizer": "finalizer", **{n: n for n in SPECIALISTS}}
    )
    for name in SPECIALISTS:
        graph.add_edge(name, "replan")
    graph.add_conditional_edges(
        "replan",
        after_replan,
        {"ticket": "ticket", "finalizer": "finalizer", **{n: n for n in SPECIALISTS}},
    )
    graph.add_edge("ticket", "finalizer")
    graph.add_edge("finalizer", "remember")
    if checkpointer is None:
        graph.add_edge("remember", END)
        return graph.compile()
    graph.add_node("await_operator", await_operator_node)
    graph.add_conditional_edges(
        "remember", after_remember, {"await_operator": "await_operator", END: END}
    )
    graph.add_edge("await_operator", "remember")
    return graph.compile(checkpointer=checkpointer)


def new_thread_id(conversation_id: str, turn: int) -> str:
    """Return a fresh checkpoint thread id for a turn.

    The id embeds the merchant-owned conversation and the turn number for readability and
    a random suffix so that two concurrent or retried requests never share a thread.
    """
    return f"{conversation_id}:{turn}:{secrets.token_hex(4)}"


def thread_config(thread_id: str) -> dict:
    """Return the LangGraph config that selects a checkpoint thread."""
    return {"configurable": {"thread_id": thread_id}}


def awaiting_operator(graph, config: dict) -> bool:
    """Return whether the thread is parked at `await_operator` with a pending interrupt.

    A finished thread accepts `Command(resume=...)` without doing anything, so callers
    check this first and store a late operator reply directly when it is false.
    """
    snapshot = graph.get_state(config)
    if not snapshot.next or "await_operator" not in snapshot.next:
        return False
    return any(getattr(task, "interrupts", ()) for task in snapshot.tasks)


CHECKPOINT_TYPES = (
    ("app.auth", "Principal"),
    ("app.state", "Evidence"),
    ("app.state", "GuardrailResult"),
    ("app.state", "PartialAnswer"),
    ("app.state", "RouterDecision"),
    ("app.state", "Source"),
    ("app.state", "ToolCall"),
    ("app.state", "TraceStep"),
    ("app.state", "Turn"),
)


def checkpoint_serializer():
    """Return the serializer with the application's state types registered explicitly.

    LangGraph deserialises only registered types in strict mode; listing ours keeps the
    checkpoints readable after the library tightens its default, and documents exactly
    which objects a checkpoint may contain.
    """
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return JsonPlusSerializer(allowed_msgpack_modules=list(CHECKPOINT_TYPES))


@lru_cache
def get_checkpointer():
    """Return the process-wide SQLite checkpointer under `state_dir`."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    settings = get_settings()
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(settings.state_dir / "checkpoints.sqlite", check_same_thread=False)
    return SqliteSaver(connection, serde=checkpoint_serializer())


@lru_cache
def get_graph():
    """Return the process-wide compiled graph with the persistent checkpointer."""
    return build_graph(get_checkpointer())
