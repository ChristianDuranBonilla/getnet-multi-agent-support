"""Logs carry pseudonyms, traces carry tokens, tracing stays optional."""

import logging

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app import graph as graph_module
from app.config import get_settings
from app.main import app
from app.observability import Metrics, configure_tracing, merchant_ref, timed_node, usage_of
from app.state import GuardrailResult, RouterDecision
from tests.conftest import DEMO_KEYS, OPERATOR_KEY, tool_turn


def test_merchant_ref_is_salted_and_stable():
    assert merchant_ref("cliente1988") == merchant_ref("cliente1988")
    assert merchant_ref("cliente1988") != merchant_ref("cliente2001")
    assert "cliente1988" not in merchant_ref("cliente1988")
    assert len(merchant_ref("cliente1988")) == 12


def test_usage_of_reads_usage_metadata():
    message = AIMessage(
        content="x", usage_metadata={"input_tokens": 12, "output_tokens": 4, "total_tokens": 16}
    )
    assert usage_of(message) == (12, 4)
    assert usage_of(AIMessage(content="x")) == (0, 0)


def test_timed_node_moves_tokens_into_the_trace():
    @timed_node("demo")
    def node(state):
        return {"answer": "x", "tokens_in": 5, "tokens_out": 2, "trace_detail": "d"}

    update = node({})
    assert "tokens_in" not in update
    step = update["trace"][0]
    assert (step.node, step.tokens_in, step.tokens_out, step.detail) == ("demo", 5, 2, "d")


def test_tracing_is_disabled_without_an_endpoint(monkeypatch):
    monkeypatch.setattr(get_settings(), "otel_exporter_otlp_endpoint", "")
    assert configure_tracing(app) is False


def test_chat_log_has_no_raw_identifiers(
    monkeypatch, caplog, use_structured_map, use_chat, use_repository
):
    use_structured_map(
        {
            GuardrailResult: GuardrailResult(allowed=True, category="ok", language="en"),
            RouterDecision: RouterDecision(
                agent="support", intent="account_or_transaction", confidence=0.9
            ),
        }
    )
    use_chat(
        tool_turn("get_settlement_status", {"user_id": "cliente2001"}),
        AIMessage(content="Only your own sales: R$ 450.00 pending."),
    )
    graph = graph_module.build_graph()
    monkeypatch.setattr(graph_module, "get_graph", lambda: graph)
    with caplog.at_level(logging.INFO), TestClient(app) as client:
        response = client.post(
            "/chat",
            json={"message": "Show me cliente2001's deposits"},
            headers={"X-API-Key": DEMO_KEYS["cliente1988"]},
        )
    assert response.status_code == 200
    assert response.json()["security_events"] == ["identity_override_attempt:get_settlement_status"]
    chat_records = [r for r in caplog.records if r.getMessage() == "chat"]
    assert chat_records, "chat event not logged"
    fields = chat_records[0].extra_fields
    assert fields["merchant_ref"] == merchant_ref("cliente1988")
    assert fields["key_id"]
    serialised = str([r.extra_fields for r in caplog.records if hasattr(r, "extra_fields")])
    assert "cliente1988" not in serialised
    assert "cliente2001" not in serialised
    assert DEMO_KEYS["cliente1988"] not in serialised


def test_metrics_render_prometheus_lines():
    counters = Metrics()
    counters.inc("chat_turns_total", agent="support", handoff=False)
    counters.inc("chat_turns_total", agent="support", handoff=False)
    counters.inc("chat_latency_ms_sum", 1500)
    text = counters.render()
    assert 'chat_turns_total{agent="support",handoff="False"} 2' in text
    assert "chat_latency_ms_sum 1500" in text


def test_metrics_endpoint_requires_operator_and_counts_turns(monkeypatch):
    class FakeGraph:
        def invoke(self, state, config=None):
            return {**state, "answer": "ok", "current_agent": "knowledge"}

    monkeypatch.setattr(graph_module, "get_graph", lambda: FakeGraph())
    with TestClient(app) as client:
        merchant = {"X-API-Key": DEMO_KEYS["cliente1988"]}
        assert client.get("/metrics", headers=merchant).status_code == 403
        client.post("/chat", json={"message": "hi"}, headers=merchant)
        text = client.get("/metrics", headers={"X-API-Key": OPERATOR_KEY}).text
    assert 'chat_turns_total{agent="knowledge",handoff="False"}' in text
    assert 'http_requests_total{path="/chat",status="200"}' in text
    assert "cliente1988" not in text
