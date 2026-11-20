"""The identity boundary: the model and the body never choose whose data a tool reads."""

import json
import logging
import os
import time

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app import auth
from app import graph as graph_module
from app.agents.support import IDENTITY_KEYS, run_tool, support_node
from app.auth import AuthError, RateLimiter, authenticate, hash_key
from app.config import get_settings
from app.main import app
from app.state import GuardrailResult, RouterDecision, initial_state
from app.tools.support_tools import MERCHANT_DATA_TOOLS, SUPPORT_TOOLS
from tests.conftest import DEMO_KEYS, OPERATOR_KEY, principal_for, tool_turn


def test_tool_schemas_expose_no_identity_field():
    for tool in SUPPORT_TOOLS:
        properties = tool.tool_call_schema.model_json_schema().get("properties", {})
        assert not IDENTITY_KEYS & set(properties), (tool.name, properties)
    for tool in MERCHANT_DATA_TOOLS:
        assert tool.tool_call_schema.model_json_schema().get("properties", {}) == {}


def test_run_tool_binds_the_principal_and_drops_model_identity(use_repository, caplog):
    principal = principal_for("cliente1988")
    with caplog.at_level(logging.WARNING):
        outcome = run_tool("get_user_profile", {"user_id": "cliente2001"}, principal)
    assert outcome.ok
    assert outcome.result["name"] == "Mariana Souza"
    assert outcome.override_attempted
    assert outcome.args == {}
    records = [r for r in caplog.records if r.getMessage() == "identity_override_attempt"]
    assert len(records) == 1
    assert records[0].extra_fields == {"tool": "get_user_profile", "key_id": "test"}
    assert "cliente2001" not in caplog.text and "cliente1988" not in caplog.text


def test_run_tool_only_forwards_declared_arguments(use_repository, caplog):
    """Identity spelled any way, or any undeclared argument, never reaches a tool."""
    principal = principal_for("cliente1988")
    with caplog.at_level(logging.INFO):
        outcome = run_tool(
            "get_settlement_status",
            {"merchantId": "cliente2001", "cnpj": "12.345.678/0001-90", "limit": 5},
            principal,
        )
    assert outcome.ok and outcome.args == {}
    assert outcome.override_attempted
    assert outcome.result["sales"][0]["sale_id"].startswith("S-")
    messages = [record.getMessage() for record in caplog.records]
    assert "identity_override_attempt" in messages
    stray = next(r for r in caplog.records if r.getMessage() == "unexpected_tool_arguments")
    assert stray.extra_fields["arguments"] == ["cnpj", "limit"]
    hop = run_tool("request_product_knowledge", {"question": "Fees?", "user_id": "x"}, principal)
    assert hop.ok and hop.args == {"question": "Fees?"} and hop.override_attempted


def test_wire_schema_sent_to_the_model_has_no_identity_field():
    """What Gemini sees (the OpenAI-style tool declaration) carries no merchant argument."""
    from langchain_core.utils.function_calling import convert_to_openai_tool

    for tool in MERCHANT_DATA_TOOLS:
        declaration = convert_to_openai_tool(tool)["function"]
        assert declaration["parameters"].get("properties", {}) == {}
        assert "merchant_id" not in json.dumps(declaration["parameters"])
    hop = convert_to_openai_tool(
        next(tool for tool in SUPPORT_TOOLS if tool.name == "request_product_knowledge")
    )
    assert set(hop["function"]["parameters"]["properties"]) == {"question"}


def test_run_tool_without_override_is_silent(use_repository):
    outcome = run_tool("get_terminal_status", {}, principal_for("cliente2001"))
    assert outcome.ok and outcome.result["connectivity"] == "offline"
    assert not outcome.override_attempted
    unknown = run_tool("delete_everything", {}, principal_for("cliente2001"))
    assert not unknown.ok and unknown.result["error"] == "unknown_tool"


def test_support_node_records_override_attempts(use_repository, use_chat):
    use_chat(
        tool_turn("get_settlement_status", {"user_id": "cliente2001"}),
        AIMessage(content="Your credit sale of R$ 450.00 is pending."),
    )
    state = initial_state("r1", principal_for("cliente1988"), "Show me cliente2001's deposits")
    update = support_node(state)
    assert update["security_events"] == ["identity_override_attempt:get_settlement_status"]
    assert update["tool_calls"][0].args == {}
    assert "450" in update["answer"]


def test_graph_never_leaks_another_merchant(use_repository, use_structured_map, use_chat):
    use_structured_map(
        {
            GuardrailResult: GuardrailResult(allowed=True, category="ok", language="en"),
            RouterDecision: RouterDecision(
                agent="support", intent="account_or_transaction", confidence=0.9
            ),
        }
    )
    use_chat(
        tool_turn("get_settlement_status", {"merchant_id": "cliente2001"}),
        AIMessage(content="I can only show your own account: R$ 450.00 pending."),
    )
    final = graph_module.build_graph().invoke(
        initial_state("r1", principal_for("cliente1988"), "Show me the deposits of cliente2001")
    )
    assert final["security_events"] == ["identity_override_attempt:get_settlement_status"]
    assert "1200" not in final["answer"]
    assert final["tool_calls"][0].args == {}


def test_authenticate_resolves_keys():
    principal = authenticate(DEMO_KEYS["cliente1988"])
    assert principal.merchant_id == "cliente1988"
    assert principal.scopes == ("chat",)
    assert principal.key_id == hash_key(DEMO_KEYS["cliente1988"])[:8]
    operator = authenticate(OPERATOR_KEY)
    assert operator.scopes == ("operator",) and operator.merchant_id == ""
    for key, status, code in (
        (None, 401, "unauthorized"),
        ("gk_cliente1988_0000000000000000", 401, "unauthorized"),
        (DEMO_KEYS["revoked"], 403, "key_revoked"),
    ):
        with pytest.raises(AuthError) as excinfo:
            authenticate(key)
        assert (excinfo.value.status, excinfo.value.code) == (status, code)


def test_rate_limiter_token_bucket():
    clock = [0.0]
    limiter = RateLimiter(per_minute=30, burst=10, clock=lambda: clock[0])
    assert all(limiter.acquire("k") is None for _ in range(10))
    assert limiter.acquire("k") == 2
    clock[0] += 2.0
    assert limiter.acquire("k") is None
    assert limiter.acquire("k") is not None
    assert limiter.acquire("other") is None


def test_api_requires_a_key_and_verifies_the_body_identity(monkeypatch):
    class FakeGraph:
        def invoke(self, state, config=None):
            return {**state, "answer": "ok", "current_agent": "knowledge"}

    monkeypatch.setattr(graph_module, "get_graph", lambda: FakeGraph())
    with TestClient(app) as client:
        assert client.post("/chat", json={"message": "hi"}).status_code == 401
        revoked = client.post(
            "/chat", json={"message": "hi"}, headers={"X-API-Key": DEMO_KEYS["revoked"]}
        )
        assert revoked.status_code == 403
        assert revoked.json()["detail"]["error"] == "key_revoked"
        headers = {"X-API-Key": DEMO_KEYS["cliente1988"]}
        mismatch = client.post(
            "/chat", json={"message": "hi", "user_id": "cliente2001"}, headers=headers
        )
        assert mismatch.status_code == 403
        assert mismatch.json()["detail"]["error"] == "identity_mismatch"
        ok = client.post("/chat", json={"message": "hi"}, headers=headers)
        assert ok.status_code == 200
        assert ok.json()["user_id"] == "cliente1988"
        operator = client.post("/chat", json={"message": "hi"}, headers={"X-API-Key": OPERATOR_KEY})
        assert operator.status_code == 403
        assert client.get("/tickets", headers=headers).status_code == 403


def test_api_rate_limit_returns_429(monkeypatch):
    limiter = RateLimiter(per_minute=30, burst=2)
    monkeypatch.setattr(auth, "get_rate_limiter", lambda: limiter)
    with TestClient(app) as client:
        headers = {"X-API-Key": DEMO_KEYS["cliente1988"]}
        client.get("/conversations/none", headers=headers)
        client.get("/conversations/none", headers=headers)
        limited = client.get("/conversations/none", headers=headers)
    assert limited.status_code == 429
    assert limited.json()["detail"]["error"] == "rate_limited"
    assert int(limited.headers["Retry-After"]) >= 1


def test_placeholder_operator_key_never_authenticates(monkeypatch):
    for placeholder in ("", "op_dev_change_me"):
        monkeypatch.setattr(get_settings(), "operator_api_key", placeholder)
        with pytest.raises(AuthError) as excinfo:
            authenticate(placeholder or "x")
        assert excinfo.value.status == 401


def test_key_store_reloads_when_the_file_changes(tmp_path, monkeypatch):
    path = tmp_path / "api_keys.json"
    path.write_text(
        json.dumps({hash_key("k1"): {"merchant_id": "m", "scopes": ["chat"], "status": "active"}})
    )
    store = auth.KeyStore(path)
    monkeypatch.setattr(auth, "get_key_store", lambda: store)
    assert authenticate("k1").merchant_id == "m"
    data = json.loads(path.read_text())
    next(iter(data.values()))["status"] = "revoked"
    path.write_text(json.dumps(data))
    os.utime(path, (time.time() + 5, time.time() + 5))
    with pytest.raises(AuthError) as excinfo:
        authenticate("k1")
    assert excinfo.value.code == "key_revoked"


def test_failed_authentications_are_throttled_per_client(monkeypatch):
    limiter = RateLimiter(per_minute=60, burst=2)
    monkeypatch.setattr(auth, "get_rate_limiter", lambda: limiter)
    with TestClient(app) as client:
        first = client.post("/chat", json={"message": "hi"}, headers={"X-API-Key": "bad-1"})
        second = client.post("/chat", json={"message": "hi"}, headers={"X-API-Key": "bad-2"})
        third = client.post("/chat", json={"message": "hi"}, headers={"X-API-Key": "bad-3"})
    assert (first.status_code, second.status_code, third.status_code) == (401, 401, 429)


def test_identity_keys_are_detected_in_any_spelling(use_repository):
    principal = principal_for("cliente1988")
    for spelling in ("userId", "MERCHANT_ID", "customer-id", "account_id "):
        outcome = run_tool("get_user_profile", {spelling: "cliente2001"}, principal)
        assert outcome.override_attempted, spelling
        assert outcome.result["name"] == "Mariana Souza"
