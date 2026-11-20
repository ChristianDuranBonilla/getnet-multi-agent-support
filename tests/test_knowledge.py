"""Knowledge agent: context building, web grounding extraction, honest sourcing."""

from langchain_core.messages import AIMessage

from app.agents.finalizer import finalizer_node, render_sources
from app.agents.knowledge import (
    build_context,
    knowledge_node,
    needs_search_retry,
    strip_sources_block,
    web_sources,
)
from app.rag import retriever as rag
from app.state import RouterDecision, Source, initial_state
from tests.conftest import FakeRetriever, principal_for


def test_build_context_applies_threshold_and_collects_evidence():
    hits = [
        ("relevant text", {"url": "https://a", "title": "A"}, 0.2),
        ("borderline", {"url": "https://b", "title": "B"}, 0.37),
        ("unrelated", {"url": "https://c", "title": "C"}, 0.5),
    ]
    context, sources, evidence = build_context(hits, 0.38)
    assert "[1] A" in context and "[2] B" in context and "unrelated" not in context
    assert [source.ref for source in sources] == ["https://a", "https://b"]
    assert [item.distance for item in evidence] == [0.2, 0.37]


def test_web_sources_reads_grounding_chunks_and_supports():
    response = AIMessage(
        content="Tomorrow 24 degrees and sunny.",
        response_metadata={
            "grounding_metadata": {
                "grounding_chunks": [
                    {"web": {"uri": "https://vertexaisearch/abc", "title": "Weather"}},
                    {"web": {"uri": "https://vertexaisearch/def", "title": "Other"}},
                ],
                "grounding_supports": [
                    {
                        "segment": {
                            "start_index": 0,
                            "end_index": 30,
                            "text": "Tomorrow 24 degrees",
                        },
                        "grounding_chunk_indices": [0],
                    }
                ],
            }
        },
    )
    sources, evidence = web_sources(response)
    assert [source.ref for source in sources] == [
        "https://vertexaisearch/abc",
        "https://vertexaisearch/def",
    ]
    assert evidence[0].excerpt == "Tomorrow 24 degrees" and evidence[1].excerpt == ""


def test_no_web_source_is_fabricated_without_grounding(monkeypatch, use_chat):
    monkeypatch.setattr(rag, "get_retriever", lambda: FakeRetriever([]))
    use_chat(AIMessage(content="It will probably rain."))
    state = initial_state("r1", principal_for("cliente1988"), "Weather in Porto Alegre tomorrow?")
    state["route"] = RouterDecision(agent="knowledge", intent="general_question", confidence=0.9)
    update = knowledge_node(state)
    assert update["sources"] == []
    assert update["evidence"] == []
    assert update["trace"][-1].detail == "web (no grounding returned)"


def test_web_sources_reports_executed_queries_without_attributions():
    response = AIMessage(
        content="1 EUR = 6.46 BRL.",
        response_metadata={
            "grounding_metadata": {
                "grounding_chunks": [],
                "grounding_supports": [],
                "web_search_queries": ["euro exchange rate today BRL"],
            }
        },
    )
    sources, evidence = web_sources(response)
    assert [source.ref for source in sources] == ["google_search: euro exchange rate today BRL"]
    assert evidence[0].excerpt.startswith("search executed")


def test_model_written_source_lists_are_dropped_and_verified_ones_appended():
    """Only sources the service can verify reach the merchant, in the user's language."""
    answer = "1 EUR is about 6.35 BRL today.\n\nSources:\nhttps://www.xe.com\nhttps://wise.com"
    assert strip_sources_block(answer) == "1 EUR is about 6.35 BRL today."
    assert strip_sources_block("Resposta.\n\nFontes:\n- https://x") == "Resposta."
    assert strip_sources_block("Respuesta.\nFuentes: https://y") == "Respuesta."
    assert strip_sources_block("No sources here. Fontes de renda são outra coisa.") == (
        "No sources here. Fontes de renda são outra coisa."
    )
    sources = [
        Source(type="tool", ref="get_settlement_status"),
        Source(type="rag", ref="https://site.getnet.com.br/duvidas/pix/"),
        Source(type="web", ref="https://vertexaisearch/abc", title="climatempo.com.br"),
        Source(type="web", ref="google_search: euro exchange rate BRL"),
    ]
    block = render_sources(sources, "en")
    assert block.splitlines() == [
        "Sources:",
        "- https://site.getnet.com.br/duvidas/pix/",
        "- web search: climatempo.com.br",
        '- web search: "euro exchange rate BRL" (no result attribution returned)',
    ]
    assert "get_settlement_status" not in block
    assert render_sources(sources[:1], "pt") == ""
    state = initial_state("r1", principal_for("cliente1988"), "Pix?")
    state.update(answer="Sim, precisa de conta.", sources=sources[1:3], language="pt")
    final = finalizer_node(state)
    assert final["answer"].endswith(
        "Fontes:\n- https://site.getnet.com.br/duvidas/pix/\n- busca na web: climatempo.com.br"
    )
    assert [source.type for source in final["sources"]] == ["rag", "web"]


def test_search_retry_only_when_nothing_can_be_verified():
    query_only = [Source(type="web", ref="google_search: Get Classica")]
    attributed = [Source(type="web", ref="https://vertexaisearch/abc", title="getnet.com.br")]
    assert needs_search_retry(True, []) is True
    assert needs_search_retry(False, []) is False
    assert needs_search_retry(False, query_only) is True
    assert needs_search_retry(True, query_only) is True
    assert needs_search_retry(False, attributed) is False
    assert needs_search_retry(False, query_only + attributed) is False
