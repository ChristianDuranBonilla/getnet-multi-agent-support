from fastapi.testclient import TestClient

from app import graph as graph_module
from app.main import app
from app.rag import retriever as rag
from app.state import RouterDecision, Source, TraceStep
from tests.conftest import FakeRetriever


class FakeGraph:
    def __init__(self):
        self.configs = []

    def invoke(self, state, config=None):
        self.configs.append(config)
        return {
            **state,
            "route": RouterDecision(agent="knowledge", intent="product_info", confidence=0.9),
            "plan": ["knowledge"],
            "current_agent": "knowledge",
            "answer": "stub answer",
            "confidence": 0.9,
            "sources": [Source(type="rag", ref="https://site.getnet.com.br/pix/")],
            "trace": [
                TraceStep(step=1, node="guardrail", ms=1, tokens_in=100, tokens_out=10),
                TraceStep(step=2, node="router", ms=1, tokens_in=100, tokens_out=10),
            ],
        }


def test_health_reports_index_size_and_readiness(monkeypatch):
    monkeypatch.setattr(rag, "get_retriever", lambda: FakeRetriever(size=42))
    with TestClient(app) as client:
        body = client.get("/health").json()
        ready = client.get("/ready")
    assert body["status"] == "ok"
    assert body["index_chunks"] == 42
    assert body["model"]
    assert "ready" in body
    assert ready.status_code in (200, 503)
    assert set(ready.json()["checks"]) == {"index", "state_dir", "credentials"}


def test_chat_rejects_empty_message(demo_key):
    with TestClient(app, headers={"X-API-Key": demo_key}) as client:
        response = client.post("/chat", json={"message": "", "user_id": "cliente1988"})
    assert response.status_code == 422


def test_chat_returns_contract(monkeypatch, demo_key):
    fake = FakeGraph()
    monkeypatch.setattr(graph_module, "get_graph", lambda: fake)
    with TestClient(app, headers={"X-API-Key": demo_key}) as client:
        response = client.post(
            "/chat", json={"message": "How does Pix work?", "user_id": "cliente1988"}
        )
    assert response.status_code == 200
    body = response.json()
    assert body["agent"] == "knowledge"
    assert body["plan"] == ["knowledge"]
    assert body["hops"] == 0
    assert body["guardrail"] is None
    assert body["response"] == "stub answer"
    assert body["handoff"] is False
    assert body["sources"][0]["type"] == "rag"
    assert body["user_id"] == "cliente1988"
    assert body["conversation_id"].startswith("cnv_")
    assert body["turn"] == 1
    assert body["usage"]["input_tokens"] == 200
    assert body["usage"]["output_tokens"] == 20
    assert body["usage"]["estimated_cost_usd"] > 0
    assert body["evidence"] is None
    assert [step["node"] for step in body["trace"]] == ["guardrail", "router"]
    assert response.headers["x-request-id"] == body["request_id"]
    assert fake.configs[0]["configurable"]["thread_id"].startswith(f"{body['conversation_id']}:1:")


def test_chat_without_user_id_uses_the_principal(monkeypatch, demo_key):
    monkeypatch.setattr(graph_module, "get_graph", lambda: FakeGraph())
    with TestClient(app, headers={"X-API-Key": demo_key}) as client:
        response = client.post("/chat", json={"message": "How does Pix work?"})
    assert response.status_code == 200
    assert response.json()["user_id"] == "cliente1988"


def test_index_serves_ui_and_no_credentials_endpoint_exists():
    with TestClient(app) as client:
        response = client.get("/")
        keys = client.get("/demo-keys")
    assert response.status_code == 200
    assert "Getnet" in response.text
    assert "X-API-Key" in response.text
    assert keys.status_code == 404


def test_errors_never_leak_tracebacks(monkeypatch, demo_key):
    class Broken:
        def invoke(self, state, config=None):
            raise RuntimeError("boom")

    monkeypatch.setattr(graph_module, "get_graph", lambda: Broken())
    with TestClient(app, headers={"X-API-Key": demo_key}, raise_server_exceptions=False) as client:
        response = client.post("/chat", json={"message": "hi"})
    assert response.status_code == 500
    assert response.json()["detail"]["error"] == "internal_error"
    assert "boom" not in response.text
