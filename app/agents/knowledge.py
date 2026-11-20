"""Knowledge agent: RAG over Getnet pages with web search as fallback."""

import re

from langchain_core.messages import HumanMessage, SystemMessage

from app import llm
from app.agents.prompts import render_history, render_partial_answers, render_prompt
from app.config import get_settings
from app.observability import model_of, timed_node, usage_of
from app.rag import retriever as rag
from app.state import AgentState, Evidence, Source

WEB_INTENTS = ("general_question",)
FORCE_SEARCH = (
    "Your previous draft had no search results behind it. Use the google_search tool now and"
    " answer only from what it returns, naming the facts that come from the search."
)


def build_context(
    hits: list[tuple[str, dict, float]], threshold: float
) -> tuple[str, list[Source], list[Evidence]]:
    """Keep the chunks under the distance threshold and format them for the prompt.

    Args:
        hits: Retriever results as (text, metadata, distance).
        threshold: Maximum cosine distance to accept.

    Returns:
        A tuple of (context_text, sources, evidence).
    """
    kept = [(text, meta, distance) for text, meta, distance in hits if distance <= threshold]
    if not kept:
        return "", [], []
    blocks = []
    urls: list[str] = []
    evidence: list[Evidence] = []
    for index, (text, meta, distance) in enumerate(kept, start=1):
        url = meta.get("url", "")
        if url and url not in urls:
            urls.append(url)
        title = meta.get("title") or meta.get("section") or url
        blocks.append(f"[{index}] {title}\nURL: {url}\n{text}")
        evidence.append(Evidence(type="rag", ref=url, excerpt=text[:600], distance=distance))
    return "\n\n".join(blocks), [Source(type="rag", ref=url) for url in urls], evidence


SOURCES_BLOCK = re.compile(
    r"\n[ \t]*(?:sources?|fontes?|fuentes?|referências|referencias|references)\s*:[\s\S]*\Z",
    re.IGNORECASE,
)


def strip_sources_block(answer: str) -> str:
    """Remove a trailing "Sources:" block written by the model.

    The service appends the sources it can verify (retrieved pages, grounding
    attributions) in the finalizer; a list written by the model may contain URLs it
    remembered rather than retrieved, so it never reaches the merchant.
    """
    return SOURCES_BLOCK.sub("", "\n" + answer).strip()


QUERY_PREFIX = "google_search: "


def attributed(sources: list[Source]) -> bool:
    """Tell whether any web source is a result attribution rather than an executed query."""
    return any(
        source.type == "web" and not source.ref.startswith(QUERY_PREFIX) for source in sources
    )


def needs_search_retry(web_only: bool, sources: list[Source]) -> bool:
    """Decide whether the answer should be asked again with the search forced.

    Args:
        web_only: Whether the question must be answered from the web (no usable context
            or an intent that needs current facts).
        sources: Web sources extracted from the first response.

    Returns:
        True when a web-only question came back with no grounding at all, or when a
        search ran but returned only its queries, so no claim can be checked against a
        result.
    """
    if web_only and not sources:
        return True
    return bool(sources) and not attributed(sources)


def web_sources(response) -> tuple[list[Source], list[Evidence]]:
    """Extract grounding references and the supported text from a google_search response.

    Args:
        response: AIMessage returned by the model.

    Returns:
        Web sources found in the grounding metadata (result URIs, or the executed search
        queries when the endpoint returns no per-result attribution), and one evidence item
        per source whose excerpt is the answer text the result supports; both empty when the
        model did not search. Nothing is fabricated: an answer without grounding metadata
        reports no web source.
    """
    metadata = getattr(response, "response_metadata", {}) or {}
    grounding = metadata.get("grounding_metadata") or {}
    chunks = grounding.get("grounding_chunks", []) or []
    supported: dict[int, list[str]] = {}
    for support in grounding.get("grounding_supports", []) or []:
        if not isinstance(support, dict):
            continue
        text = str((support.get("segment") or {}).get("text") or "").strip()
        for index in support.get("grounding_chunk_indices", []) or []:
            if text:
                supported.setdefault(int(index), []).append(text)
    sources: list[Source] = []
    evidence: list[Evidence] = []
    for index, chunk in enumerate(chunks):
        web = chunk.get("web") if isinstance(chunk, dict) else None
        if not (web and web.get("uri")):
            continue
        title = str(web.get("title") or web.get("domain") or "").strip() or None
        sources.append(Source(type="web", ref=web["uri"], title=title))
        excerpt = " ".join(supported.get(index, []))[:600]
        evidence.append(Evidence(type="web", ref=web["uri"], excerpt=excerpt))
    if not sources:
        # Gemini sometimes runs the search and reports the queries without per-result
        # attributions. The queries are still proof of a search, so they are reported as
        # the source; nothing is invented beyond what the metadata states.
        for query in grounding.get("web_search_queries", []) or []:
            ref = f"{QUERY_PREFIX}{query}"
            sources.append(Source(type="web", ref=ref))
            evidence.append(
                Evidence(type="web", ref=ref, excerpt="search executed; no result attribution")
            )
    return sources, evidence


@timed_node("knowledge")
def knowledge_node(state: AgentState) -> dict:
    """Answer from the index when it is relevant, complemented by web search.

    The model always has google_search available: the index covers Getnet's own
    pages, while specifications and world facts come from the web. When another
    specialist ran first in this turn, the question is the one it asked and the
    answer merges both.

    Args:
        state: Current graph state.

    Returns:
        State update with `answer`, `sources`, `evidence` and possibly `handoff`.
    """
    settings = get_settings()
    route = state.get("route")
    intent = route.intent if route else "other"
    question = state.get("hop_question") or state["message"]
    try:
        hits = rag.get_retriever().search(question, settings.rag_top_k)
    except Exception:
        hits = []
    context, rag_refs, evidence = build_context(hits, settings.rag_distance_threshold)
    web_only = (intent in WEB_INTENTS and not state.get("hop_question")) or not context
    system = render_prompt(
        "knowledge",
        language=state.get("language", "en"),
        history=render_history(state.get("summary", ""), state.get("history", [])),
        partial_answers=render_partial_answers(state.get("partial_answers", [])),
        context=context or "(empty)",
    )
    user = question
    if state.get("hop_question"):
        user = f"Original message: {state['message']}\nProduct question: {question}"
    model = llm.chat(tools=[llm.GOOGLE_SEARCH_TOOL])
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    retry_tokens = (0, 0)
    try:
        response = llm.invoke_resilient(model, messages)
        if needs_search_retry(web_only, web_sources(response)[0]):
            # A world question answered without grounding is an answer from memory, and a
            # search that ran without result attributions leaves its claims unverifiable;
            # ask once more with the search made explicit before giving up on sources.
            retry_tokens = usage_of(response)
            response = llm.invoke_resilient(model, [*messages, SystemMessage(content=FORCE_SEARCH)])
    except Exception as exc:
        return {
            "current_agent": "knowledge",
            "answer": None,
            "sources": rag_refs,
            "evidence": evidence,
            "handoff": True,
            "handoff_reason": f"knowledge model error: {type(exc).__name__}",
            "trace_detail": "model error",
        }
    tokens_in, tokens_out = usage_of(response)
    tokens_in, tokens_out = tokens_in + retry_tokens[0], tokens_out + retry_tokens[1]
    answer = strip_sources_block(llm.message_text(response.content))
    web_refs, web_evidence = web_sources(response)
    evidence.extend(web_evidence)
    sources = (rag_refs if not web_only else []) + web_refs
    if web_only:
        detail = "web" if web_refs else "web (no grounding returned)"
    else:
        detail = f"rag:{len(rag_refs)}" + ("+web" if web_refs else "")
    update = {
        "current_agent": "knowledge",
        "answer": answer,
        "sources": sources,
        "evidence": evidence,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "model": model_of(response),
        "trace_detail": detail,
    }
    if not answer.strip():
        update.update(handoff=True, handoff_reason="empty answer")
    return update
