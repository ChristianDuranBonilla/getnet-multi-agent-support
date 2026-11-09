"""Escalation agent: opens a ticket and hands the conversation to a human."""

from langchain_core.messages import HumanMessage, SystemMessage

from app import llm, repositories
from app.agents.prompts import render_prompt
from app.observability import timed_node
from app.state import AgentState

HANDOFF_TEMPLATES = {
    "en": "I've opened support ticket {ticket_id} and a human agent will follow up with you.",
    "pt": "Abri o chamado {ticket_id} e um atendente humano entrará em contato com você.",
    "es": "Abrí el ticket {ticket_id} y un agente humano se pondrá en contacto contigo.",
}


def handoff_sentence(language: str, ticket_id: str) -> str:
    """Return the templated handoff sentence in the user's language."""
    template = HANDOFF_TEMPLATES.get(language, HANDOFF_TEMPLATES["en"])
    return template.format(ticket_id=ticket_id)


def compose(partial: str | None, language: str, ticket_id: str) -> str:
    """Merge a partial answer with the handoff notice.

    When the specialist produced an answer, the templated sentence is appended as is:
    that is exactly what the model would be asked to do, and it saves a model call of
    two to four seconds on every handoff. The model only writes the notice when there
    is no partial answer to attach it to.

    Args:
        partial: Answer produced by the specialist, if any.
        language: ISO code of the user's language.
        ticket_id: Ticket just created.

    Returns:
        Final text for the user.
    """
    fallback = f"{partial.strip()}\n\n" if partial else ""
    fallback += handoff_sentence(language, ticket_id)
    if partial:
        return fallback
    try:
        system = render_prompt("escalation", language=language, ticket_id=ticket_id)
        user = "(no partial answer; write only the handoff sentence)"
        response = llm.chat(thinking="low").invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        )
        text = llm.message_text(response.content).strip()
    except Exception:
        return fallback
    return text if ticket_id in text else fallback


@timed_node("escalation")
def escalation_node(state: AgentState) -> dict:
    """Create a ticket and append the human-handoff notice.

    Args:
        state: Current graph state with `handoff` set.

    Returns:
        State update with `ticket_id` and the final `answer`.
    """
    reason = state.get("handoff_reason") or "unresolved request"
    priority = "high" if "under review" in reason or "error" in reason else "medium"
    summary = f"{reason}: {state['message'][:160]}"
    ticket_id = repositories.get_repository().create_ticket(state["user_id"], summary, priority)
    answer = compose(state.get("answer"), state.get("language", "en"), ticket_id)
    return {"ticket_id": ticket_id, "answer": answer, "trace_detail": reason}
