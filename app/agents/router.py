"""Router agent: decides which specialist handles the message."""

from langchain_core.messages import HumanMessage, SystemMessage

from app import llm
from app.agents.prompts import load_prompt
from app.observability import timed_node
from app.state import AgentState, RouterDecision

LOW_CONFIDENCE = 0.5


def decide(message: str) -> RouterDecision:
    """Ask the model for a routing decision.

    Args:
        message: Guardrail-cleaned user message.

    Returns:
        The decision, or a low-confidence default when the model fails.
    """
    try:
        result = llm.structured(RouterDecision).invoke(
            [SystemMessage(content=load_prompt("router")), HumanMessage(content=message)]
        )
    except Exception:
        return RouterDecision(
            agent="knowledge", intent="other", confidence=0.0, rationale="router unavailable"
        )
    if isinstance(result, dict):
        result = RouterDecision.model_validate(result)
    return result


@timed_node("router")
def router_node(state: AgentState) -> dict:
    """Store the routing decision and flag low confidence for escalation.

    Args:
        state: Current graph state.

    Returns:
        State update with `route`, `confidence` and possibly `handoff`.
    """
    decision = decide(state["message"])
    update = {
        "route": decision,
        "confidence": decision.confidence,
        "trace_detail": f"{decision.agent}/{decision.intent}",
    }
    if decision.confidence < LOW_CONFIDENCE:
        update["route"] = decision.model_copy(update={"agent": "knowledge"})
        update["handoff"] = True
        update["handoff_reason"] = "low routing confidence"
    return update
