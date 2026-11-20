"""Escalation: open a persistent ticket, hand off to a human, resume with their reply."""

import json

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import interrupt

from app import llm, repositories
from app.agents.prompts import render_prompt
from app.observability import merchant_ref, model_of, timed_node, usage_of
from app.state import AgentState, Evidence

HANDOFF_TEMPLATES = {
    "en": "I've opened support ticket {ticket_id} and a human agent will follow up with you.",
    "pt": "Abri o chamado {ticket_id} e um atendente humano entrará em contato com você.",
    "es": "Abrí el ticket {ticket_id} y un agente humano se pondrá en contacto contigo.",
}
REUSE_TEMPLATES = {
    "en": "Support ticket {ticket_id} is already open for this; our team will follow up with you.",
    "pt": "O chamado {ticket_id} já está aberto para isso; nossa equipe entrará em contato.",
    "es": "El ticket {ticket_id} ya está abierto para esto; nuestro equipo se pondrá en contacto.",
}
OPERATOR_PREFIX = {
    "en": "Reply from our support team",
    "pt": "Resposta da nossa equipe de suporte",
    "es": "Respuesta de nuestro equipo de soporte",
}


def handoff_sentence(language: str, ticket_id: str) -> str:
    """Return the templated handoff sentence in the user's language."""
    template = HANDOFF_TEMPLATES.get(language, HANDOFF_TEMPLATES["en"])
    return template.format(ticket_id=ticket_id)


def reuse_sentence(language: str, ticket_id: str) -> str:
    """Return the sentence that points the merchant to the ticket already open."""
    template = REUSE_TEMPLATES.get(language, REUSE_TEMPLATES["en"])
    return template.format(ticket_id=ticket_id)


def open_ticket_for(tickets: list[dict], reason: str) -> dict | None:
    """Return the unresolved ticket of a conversation opened for the same reason, if any."""
    for ticket in tickets:
        if ticket.get("reason") == reason and ticket.get("status") != "resolved":
            return ticket
    return None


def compose(partial: str | None, language: str, ticket_id: str) -> tuple[str, tuple[int, int], str]:
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
        Final text for the user, the token usage and the model that wrote it, if any.
    """
    fallback = f"{partial.strip()}\n\n" if partial else ""
    fallback += handoff_sentence(language, ticket_id)
    if partial:
        return fallback, (0, 0), ""
    try:
        system = render_prompt("escalation", language=language, ticket_id=ticket_id)
        user = "(no partial answer; write only the handoff sentence)"
        response = llm.invoke_resilient(
            llm.chat(thinking="low"), [SystemMessage(content=system), HumanMessage(content=user)]
        )
        text = llm.message_text(response.content).strip()
    except Exception:
        return fallback, (0, 0), ""
    return (text if ticket_id in text else fallback), usage_of(response), model_of(response)


def operator_message(language: str, message: str) -> str:
    """Return the operator's reply as shown to the merchant."""
    prefix = OPERATOR_PREFIX.get(language, OPERATOR_PREFIX["en"])
    return f"{prefix}: {message.strip()}"


@timed_node("ticket")
def ticket_node(state: AgentState) -> dict:
    """Create the ticket (idempotently) and append the human-handoff notice.

    A conversation gets one ticket per cause: when an unresolved ticket with the same
    reason already exists, the merchant is pointed to it and no second ticket (and no
    second wait for an operator) is opened.

    Args:
        state: Current graph state with `handoff` set.

    Returns:
        State update with `ticket_id` and the final `answer`.
    """
    reason = state.get("handoff_reason") or "unresolved request"
    priority = (
        "high" if any(word in reason for word in ("under review", "error", "failed")) else "medium"
    )
    summary = f"{reason}: {state['message'][:160]}"
    conversation_id = state.get("conversation_id", "")
    language = state.get("language", "en")
    tickets = repositories.get_ticket_repository()
    existing = open_ticket_for(tickets.for_conversation(conversation_id), reason)
    if existing is not None:
        ticket_id = existing["ticket_id"]
        partial = (state.get("answer") or "").strip()
        answer = f"{partial}\n\n" if partial else ""
        answer += reuse_sentence(language, ticket_id)
        record = {"ticket_id": ticket_id, "reason": reason, "status": existing["status"]}
        return {
            "ticket_id": ticket_id,
            "ticket_reused": True,
            "evidence": [Evidence(type="tool", ref="existing_ticket", excerpt=json.dumps(record))],
            "answer": answer,
            "trace_detail": f"{reason} (ticket {ticket_id} already open)",
        }
    ticket_id = tickets.create_ticket(
        merchant_id=state["principal"].merchant_id,
        summary=summary,
        priority=priority,
        conversation_id=conversation_id,
        request_id=state["request_id"],
        thread_id=state.get("thread_id") or f"{conversation_id}:{state.get('turn', 1)}",
        reason=reason,
    )
    answer, (tokens_in, tokens_out), model = compose(state.get("answer"), language, ticket_id)
    record = {"ticket_id": ticket_id, "priority": priority, "reason": reason, "status": "open"}
    return {
        "ticket_id": ticket_id,
        "ticket_reused": False,
        "evidence": [Evidence(type="tool", ref="create_ticket", excerpt=json.dumps(record))],
        "answer": answer,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "model": model,
        "trace_detail": reason,
    }


@timed_node("await_operator")
def await_operator_node(state: AgentState) -> dict:
    """Suspend the turn until an operator answers the ticket, then carry their reply.

    On the first run `interrupt` stops the graph; the API answers the merchant with the
    handoff notice already in the state. When the operator replies, the graph is resumed
    with `Command(resume=...)`, this node runs again, receives the reply and the turn
    continues to `remember`, which stores it as an operator turn.

    Args:
        state: Current graph state with `ticket_id` set.

    Returns:
        State update with `operator_reply` and the merchant-facing `answer`.
    """
    reply = interrupt(
        {
            "ticket_id": state.get("ticket_id"),
            "merchant_ref": merchant_ref(state["principal"].merchant_id),
        }
    )
    message = str(reply.get("message", "")) if isinstance(reply, dict) else str(reply)
    return {
        "operator_reply": message,
        "answer": operator_message(state.get("language", "en"), message),
        "trace_detail": "resumed",
    }
