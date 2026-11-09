from fastapi.testclient import TestClient

from app import graph as graph_module
from app.main import app
from app.rag import retriever as rag
from app.state import RouterDecision, Source, TraceStep
from tests.conftest import FakeRetriever


class FakeGraph:
    def invoke(self, state):
        return {
            **state,
            "route": RouterDecision(agent="knowledge", intent="product_info", confidence=0.9),
            "answer": "stub answer",
            "confidence": 0.9,
            "sources": [Source(type="rag", ref="https://site.getnet.com.br/pix/")],
            "trace": [
                TraceStep(step=1, node="guardrail", ms=1),
                TraceStep(step=2, node="router", ms=1),
            ],
        }


def test_health_reports_index_size(monkeypatch):
    monkeypatch.setattr(rag, "get_retriever", lambda: FakeRetriever(size=42))
    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["index_chunks"] == 42
    assert body["model"]


def test_chat_rejects_empty_message():
    with TestClient(app) as client:
        response = client.post("/chat", json={"message": "", "user_id": "cliente1988"})
    assert response.status_code == 422


def test_chat_returns_contract(monkeypatch):
    monkeypatch.setattr(graph_module, "get_graph", lambda: FakeGraph())
    with TestClient(app) as client:
        response = client.post(
            "/chat", json={"message": "How does Pix work?", "user_id": "cliente1988"}
        )
    assert response.status_code == 200
    body = response.json()
    assert body["agent"] == "knowledge"
    assert body["guardrail"] is None
    assert body["response"] == "stub answer"
    assert body["handoff"] is False
    assert body["sources"][0]["type"] == "rag"
    assert [step["node"] for step in body["trace"]] == ["guardrail", "router"]
    assert response.headers["x-request-id"] == body["request_id"]


def test_index_serves_ui():
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "Getnet" in response.text
