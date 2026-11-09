"""LangGraph wiring of the multi-agent orchestration.

Flow: sanitize -> (guardrail || router) -> dispatch -> specialist -> [escalation] -> finalizer.
The Guardrail classifier and the Router are independent model calls over the same
sanitized message, so they run in one parallel superstep.
"""

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.agents.escalation import escalation_node
from app.agents.finalizer import finalizer_node
from app.agents.guardrail import guardrail_node, sanitize_node
from app.agents.knowledge import knowledge_node
from app.agents.router import router_node
from app.agents.support import support_node
from app.state import AgentState

SPECIALISTS = {"knowledge": knowledge_node, "support": support_node}
CLASSIFIERS = ["guardrail", "router"]
BLOCKING_CATEGORIES = ("unsafe", "prompt_injection", "off_topic")


def after_sanitize(state: AgentState) -> str | list[str]:
    """Skip the model calls for deterministic blocks, otherwise fan out to both classifiers."""
    return "finalizer" if state.get("answer") else CLASSIFIERS


def after_dispatch(state: AgentState) -> str:
    """Stop when the guardrail produced a reply, otherwise follow the Router."""
    if state.get("answer"):
        return "finalizer"
    route = state.get("route")
    return route.agent if route and route.agent in SPECIALISTS else "knowledge"


def after_specialist(state: AgentState) -> str:
    """Escalate when any node asked for a human handoff."""
    return "escalation" if state.get("handoff") else "finalizer"


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
    route = state.get("route")
    if route is None or guardrail_blocked(state):
        return None
    return route.agent


def build_graph():
    """Assemble and compile the orchestration graph.

    Returns:
        A compiled LangGraph application that accepts an `AgentState`.
    """
    graph = StateGraph(AgentState)
    graph.add_node("sanitize", sanitize_node)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("router", router_node)
    graph.add_node("dispatch", dispatch_node)
    for name, node in SPECIALISTS.items():
        graph.add_node(name, node)
    graph.add_node("escalation", escalation_node)
    graph.add_node("finalizer", finalizer_node)

    graph.add_edge(START, "sanitize")
    graph.add_conditional_edges(
        "sanitize", after_sanitize, {"finalizer": "finalizer", **{n: n for n in CLASSIFIERS}}
    )
    for name in CLASSIFIERS:
        graph.add_edge(name, "dispatch")
    graph.add_conditional_edges(
        "dispatch", after_dispatch, {"finalizer": "finalizer", **{n: n for n in SPECIALISTS}}
    )
    for name in SPECIALISTS:
        graph.add_conditional_edges(
            name, after_specialist, {"escalation": "escalation", "finalizer": "finalizer"}
        )
    graph.add_edge("escalation", "finalizer")
    graph.add_edge("finalizer", END)
    return graph.compile()


@lru_cache
def get_graph():
    """Return the process-wide compiled graph."""
    return build_graph()
