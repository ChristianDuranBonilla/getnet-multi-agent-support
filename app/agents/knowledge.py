"""Knowledge agent: RAG over Getnet pages with web search as fallback."""

from langchain_core.messages import HumanMessage, SystemMessage

from app import llm
from app.agents.prompts import render_prompt
from app.config import get_settings
from app.observability import timed_node
from app.rag import retriever as rag
from app.state import AgentState, Source

WEB_INTENTS = ("general_question",)


def build_context(
    hits: list[tuple[str, dict, float]], threshold: float
) -> tuple[str, list[Source]]:
    """Keep the chunks under the distance threshold and format them for the prompt.

    Args:
        hits: Retriever results as (text, metadata, distance).
        threshold: Maximum cosine distance to accept.

    Returns:
        A tuple of (context_text, sources).
    """
    kept = [(text, meta) for text, meta, distance in hits if distance <= threshold]
    if not kept:
        return "", []
    blocks = []
    urls: list[str] = []
    for index, (text, meta) in enumerate(kept, start=1):
        url = meta.get("url", "")
        if url and url not in urls:
            urls.append(url)
        title = meta.get("title") or meta.get("section") or url
        blocks.append(f"[{index}] {title}\nURL: {url}\n{text}")
    return "\n\n".join(blocks), [Source(type="rag", ref=url) for url in urls]


def web_sources(response) -> list[Source]:
    """Extract grounding references from a google_search response.

    Args:
        response: AIMessage returned by the model.

    Returns:
        Web sources found in the grounding metadata; empty when search was not used.
    """
    metadata = getattr(response, "response_metadata", {}) or {}
    grounding = metadata.get("grounding_metadata") or {}
    sources = []
    for chunk in grounding.get("grounding_chunks", []) or []:
        web = chunk.get("web") if isinstance(chunk, dict) else None
        if web and web.get("uri"):
            sources.append(Source(type="web", ref=web["uri"]))
    return sources


@timed_node("knowledge")
def knowledge_node(state: AgentState) -> dict:
    """Answer from the index when it is relevant, complemented by web search.

    The model always has google_search available: the index covers Getnet's own
    pages, while specifications and world facts come from the web.

    Args:
        state: Current graph state.

    Returns:
        State update with `answer`, `sources` and possibly `handoff`.
    """
    settings = get_settings()
    route = state.get("route")
    intent = route.intent if route else "other"
    try:
        hits = rag.get_retriever().search(state["message"], settings.rag_top_k)
    except Exception:
        hits = []
    context, rag_refs = build_context(hits, settings.rag_distance_threshold)
    web_only = intent in WEB_INTENTS or not context
    system = render_prompt(
        "knowledge", language=state.get("language", "en"), context=context or "(empty)"
    )
    model = llm.chat(tools=[llm.GOOGLE_SEARCH_TOOL])
    try:
        response = model.invoke(
            [SystemMessage(content=system), HumanMessage(content=state["message"])]
        )
    except Exception as exc:
        return {
            "answer": None,
            "sources": rag_refs,
            "handoff": True,
            "handoff_reason": f"knowledge model error: {type(exc).__name__}",
            "trace_detail": "model error",
        }
    answer = llm.message_text(response.content)
    web_refs = web_sources(response)
    if web_only and not web_refs:
        web_refs = [Source(type="web", ref="google_search")]
    sources = (rag_refs if not web_only else []) + web_refs
    detail = "web" if web_only else f"rag:{len(rag_refs)}" + ("+web" if web_refs else "")
    update = {"answer": answer, "sources": sources, "trace_detail": detail}
    if not answer.strip():
        update.update(handoff=True, handoff_reason="empty answer")
    return update
