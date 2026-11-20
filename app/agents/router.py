"""Router agent: decides which specialists handle the message and in which order."""

from langchain_core.messages import HumanMessage, SystemMessage

from app import llm
from app.agents.prompts import render_history, render_prompt
from app.observability import timed_node
from app.state import AgentState, RouterDecision

LOW_CONFIDENCE = 0.5


def decide(message: str, history_block: str) -> tuple[RouterDecision, tuple[int, int], str]:
    """Ask the model for a routing decision.

    Args:
        message: Guardrail-cleaned user message.
        history_block: Rendered conversation memory.

    Returns:
        The decision (a low-confidence default when the model fails), the token usage and
        the model that answered.
    """
    try:
        model = llm.structured(RouterDecision)
        result = llm.invoke_resilient(
            model,
            [
                SystemMessage(content=render_prompt("router", history=history_block)),
                HumanMessage(content=message),
            ],
        )
    except Exception:
        fallback = RouterDecision(
            agent="knowledge", intent="other", confidence=0.0, rationale="router unavailable"
        )
        return fallback, (0, 0), ""
    if isinstance(result, dict):
        result = RouterDecision.model_validate(result)
    return result, getattr(model, "last_usage", (0, 0)), getattr(model, "last_model", "")


@timed_node("router")
def router_node(state: AgentState) -> dict:
    """Store the routing decision and flag low confidence for escalation.

    Args:
        state: Current graph state.

    Returns:
        State update with `route`, `plan`, `confidence` and possibly `handoff`.
    """
    history_block = render_history(state.get("summary", ""), state.get("history", []))
    decision, (tokens_in, tokens_out), model = decide(state["message"], history_block)
    update = {
        "route": decision,
        "plan": list(decision.plan),
        "confidence": decision.confidence,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "model": model,
        "trace_detail": "/".join([*decision.plan, decision.intent]),
    }
    if decision.confidence < LOW_CONFIDENCE:
        decision = decision.model_copy(update={"agent": "knowledge", "plan": ["knowledge"]})
        update["route"] = decision
        update["plan"] = ["knowledge"]
        update["handoff"] = True
        update["handoff_reason"] = "low routing confidence"
    return update
