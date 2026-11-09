"""Finalizer: last node, normalises the answer before the API builds the response."""

from app.observability import timed_node
from app.state import AgentState

PII_NOTICES = {
    "en": (
        "For your security I masked the card or document number you shared; "
        "please never send full card numbers."
    ),
    "pt": (
        "Por segurança, ocultei o número de cartão ou documento que você enviou; "
        "nunca envie números completos."
    ),
    "es": (
        "Por seguridad oculté el número de tarjeta o documento que enviaste; "
        "nunca envíes números completos."
    ),
}
FALLBACK = {
    "en": "I couldn't complete your request right now.",
    "pt": "Não consegui concluir sua solicitação agora.",
    "es": "No pude completar tu solicitud en este momento.",
}


@timed_node("finalizer")
def finalizer_node(state: AgentState) -> dict:
    """Guarantee a non-empty answer and add the PII notice when applicable.

    Args:
        state: Current graph state.

    Returns:
        State update with the final `answer`.
    """
    language = state.get("language", "en")
    answer = (state.get("answer") or "").strip() or FALLBACK.get(language, FALLBACK["en"])
    verdict = state.get("guardrail")
    if verdict and verdict.category == "pii":
        answer = f"{answer}\n\n{PII_NOTICES.get(language, PII_NOTICES['en'])}"
    return {"answer": answer}
