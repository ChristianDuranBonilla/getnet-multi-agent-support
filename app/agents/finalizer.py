"""Finalizer: last node before memory, normalises the answer and merges the sources."""

from app.agents.guardrail import mask_pii
from app.observability import timed_node
from app.state import AgentState, Source

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
SOURCES_HEADING = {"en": "Sources", "pt": "Fontes", "es": "Fuentes"}
WEB_SEARCH = {"en": "web search", "pt": "busca na web", "es": "búsqueda web"}
NO_ATTRIBUTION = {
    "en": "no result attribution returned",
    "pt": "sem atribuição de resultados",
    "es": "sin atribución de resultados",
}
QUERY_PREFIX = "google_search: "


def render_sources(sources: list[Source], language: str) -> str:
    """Render the verified sources as the block appended to the answer text.

    Retrieved pages are listed by URL. Web results are listed by the domain the grounding
    metadata attributes them to (the redirect URI stays in the structured `sources`); a
    search that ran without per-result attribution is listed by its query. Tool sources
    are not listed: the merchant's own data needs no citation.

    Args:
        sources: Merged sources of the turn.
        language: ISO code of the user's language.

    Returns:
        The block, or an empty string when there is nothing to cite.
    """
    search = WEB_SEARCH.get(language, WEB_SEARCH["en"])
    lines = []
    for source in sources:
        if source.type == "rag":
            lines.append(f"- {source.ref}")
        elif source.type == "web" and source.ref.startswith(QUERY_PREFIX):
            query = source.ref[len(QUERY_PREFIX) :]
            lines.append(
                f'- {search}: "{query}" ({NO_ATTRIBUTION.get(language, NO_ATTRIBUTION["en"])})'
            )
        elif source.type == "web":
            lines.append(f"- {search}: {source.title or source.ref}")
    if not lines:
        return ""
    heading = SOURCES_HEADING.get(language, SOURCES_HEADING["en"])
    unique = list(dict.fromkeys(lines))
    return f"{heading}:\n" + "\n".join(unique)


def merge_sources(state: AgentState) -> list[Source]:
    """Return the sources of every specialist that ran in this turn, deduplicated, in order."""
    merged: list[Source] = []
    seen: set[tuple[str, str]] = set()
    partials = [
        source for partial in state.get("partial_answers", []) for source in partial.sources
    ]
    for source in [*partials, *state.get("sources", [])]:
        key = (source.type, source.ref)
        if key not in seen:
            seen.add(key)
            merged.append(source)
    return merged


@timed_node("finalizer")
def finalizer_node(state: AgentState) -> dict:
    """Guarantee a non-empty answer, mask PII in it, append the sources and the notice.

    The output is masked with the same deterministic rules as the input, so a card
    number that reached the answer through a tool result or a model slip never leaves
    the service.

    Args:
        state: Current graph state.

    Returns:
        State update with the final `answer` and `sources`.
    """
    language = state.get("language", "en")
    answer = (state.get("answer") or "").strip() or FALLBACK.get(language, FALLBACK["en"])
    answer, _ = mask_pii(answer)
    sources = merge_sources(state)
    block = render_sources(sources, language)
    if block:
        answer = f"{answer}\n\n{block}"
    verdict = state.get("guardrail")
    if verdict and verdict.category == "pii":
        answer = f"{answer}\n\n{PII_NOTICES.get(language, PII_NOTICES['en'])}"
    return {"answer": answer, "sources": sources}
