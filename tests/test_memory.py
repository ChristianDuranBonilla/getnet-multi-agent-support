"""Conversation memory: ownership, recall, remember, rolling summary, API surface."""

from datetime import UTC, datetime

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app import graph as graph_module
from app.agents.memory import recall_node, remember_node
from app.agents.prompts import render_history
from app.config import get_settings
from app.main import app
from app.state import (
    ConversationSummary,
    GuardrailResult,
    RouterDecision,
    Turn,
    initial_state,
)
from tests.conftest import DEMO_KEYS, principal_for, tool_turn

OK = GuardrailResult(allowed=True, category="ok", language="en")


def turn(role, content, agent=None):
    return Turn(role=role, content=content, agent=agent, created_at=datetime.now(UTC))


def test_repository_enforces_ownership(conversations):
    conversation_id = conversations.create("cliente1988")
    conversations.append(conversation_id, turn("merchant", "hi"))
    assert conversations.get(conversation_id, "cliente2001") is None
    assert conversations.get("cnv_missing", "cliente1988") is None
    owned = conversations.get(conversation_id, "cliente1988")
    assert [item.role for item in owned.turns] == ["merchant"]


def test_conversations_survive_a_restart(tmp_path):
    from app.repositories.sqlite_repository import SqliteConversationRepository, SqliteStore

    path = tmp_path / "restart.sqlite"
    first = SqliteConversationRepository(SqliteStore(path))
    conversation_id = first.create("cliente1988")
    first.append(conversation_id, turn("merchant", "before restart"))
    second = SqliteConversationRepository(SqliteStore(path))
    restored = second.get(conversation_id, "cliente1988")
    assert [item.content for item in restored.turns] == ["before restart"]


def test_render_history_block():
    assert render_history("", []) == "Conversation so far: this is the first message."
    block = render_history("Device offline.", [turn("merchant", "It's the Smart.")])
    assert block.startswith("Conversation so far:")
    assert "Summary of earlier turns: Device offline." in block
    assert "merchant: It's the Smart." in block


def test_recall_loads_window_and_summary(conversations, monkeypatch):
    monkeypatch.setattr(get_settings(), "memory_window_turns", 1)
    conversation_id = conversations.create("cliente1988")
    for index in range(4):
        conversations.append(conversation_id, turn("merchant", f"m{index}"))
        conversations.append(conversation_id, turn("assistant", f"a{index}", "support"))
    conversations.update_summary(conversation_id, "earlier", 6)
    state = initial_state("r1", principal_for("cliente1988"), "next", conversation_id, 5)
    update = recall_node(state)
    assert [item.content for item in update["history"]] == ["m3", "a3"]
    assert update["summary"] == "earlier"


def test_remember_stores_sanitized_turns_and_skips_foreign_conversation(conversations):
    conversation_id = conversations.create("cliente1988")
    state = initial_state("r1", principal_for("cliente1988"), "card **** 1111", conversation_id, 1)
    state.update(answer="Masked.", current_agent="support")
    update = remember_node(state)
    stored = conversations.get(conversation_id, "cliente1988")
    assert [(item.role, item.content, item.agent) for item in stored.turns] == [
        ("merchant", "card **** 1111", None),
        ("assistant", "Masked.", "support"),
    ]
    assert update["trace"][-1].detail == "2 turns"
    foreign = initial_state("r2", principal_for("cliente2001"), "hi", conversation_id, 1)
    foreign["answer"] = "no"
    assert remember_node(foreign)["trace"][-1].detail == "no conversation"
    assert len(conversations.get(conversation_id, "cliente1988").turns) == 2


def test_rolling_summary_triggers_after_threshold(conversations, use_structured, monkeypatch):
    monkeypatch.setattr(get_settings(), "memory_summary_every", 4)
    fake = use_structured(ConversationSummary(summary="Merchant asked twice about deposits."))
    conversation_id = conversations.create("cliente1988")
    for index in range(2):
        state = initial_state("r", principal_for("cliente1988"), f"q{index}", conversation_id, 1)
        state.update(answer=f"a{index}", current_agent="support")
        remember_node(state)
    stored = conversations.get(conversation_id, "cliente1988")
    assert stored.summary == "Merchant asked twice about deposits."
    assert stored.summary_through_turn == 4
    assert len(fake.calls) == 1
    assert "q0" in fake.calls[0][1].content


def test_second_turn_carries_history_to_router(
    conversations, use_structured_map, use_chat, use_repository
):
    fakes = use_structured_map(
        {
            GuardrailResult: OK,
            RouterDecision: RouterDecision(agent="support", intent="device_issue", confidence=0.9),
        }
    )
    use_chat(tool_turn("get_terminal_status", {}), AIMessage(content="Your Get Smart is online."))
    graph = graph_module.build_graph()
    principal = principal_for("cliente1988")
    conversation_id = conversations.create("cliente1988")
    first = graph.invoke(
        initial_state("r1", principal, "My card machine is offline.", conversation_id, 1)
    )
    assert first["answer"]
    use_chat(tool_turn("get_terminal_status", {}), AIMessage(content="The Smart is online on 4G."))
    second = graph.invoke(initial_state("r2", principal, "It's the Smart.", conversation_id, 2))
    assert [item.content for item in second["history"]][:2] == [
        "My card machine is offline.",
        "Your Get Smart is online.",
    ]
    router_prompt = fakes[RouterDecision].calls[-1][0].content
    assert "My card machine is offline." in router_prompt
    assert len(conversations.get(conversation_id, "cliente1988").turns) == 4


def test_api_conversation_lifecycle(monkeypatch, conversations):
    class FakeGraph:
        def invoke(self, state, config=None):
            conversations.append(state["conversation_id"], turn("merchant", state["message"]))
            conversations.append(state["conversation_id"], turn("assistant", "ok", "knowledge"))
            return {**state, "answer": "ok", "current_agent": "knowledge"}

    monkeypatch.setattr(graph_module, "get_graph", lambda: FakeGraph())
    headers = {"X-API-Key": DEMO_KEYS["cliente1988"]}
    with TestClient(app) as client:
        first = client.post("/chat", json={"message": "hi"}, headers=headers).json()
        second = client.post(
            "/chat",
            json={"message": "again", "conversation_id": first["conversation_id"]},
            headers=headers,
        ).json()
        assert (first["turn"], second["turn"]) == (1, 2)
        mine = client.get(f"/conversations/{first['conversation_id']}", headers=headers)
        assert mine.status_code == 200
        assert [item["role"] for item in mine.json()["turns"]] == [
            "merchant",
            "assistant",
            "merchant",
            "assistant",
        ]
        foreign = client.get(
            f"/conversations/{first['conversation_id']}",
            headers={"X-API-Key": DEMO_KEYS["cliente2001"]},
        )
        assert foreign.status_code == 404
        assert foreign.json()["detail"]["error"] == "conversation_not_found"
        missing = client.post(
            "/chat", json={"message": "x", "conversation_id": "cnv_missing"}, headers=headers
        )
        assert missing.status_code == 404


def test_summary_failure_keeps_the_cursor(conversations, use_structured, monkeypatch):
    monkeypatch.setattr(get_settings(), "memory_summary_every", 2)
    use_structured(RuntimeError("vertex down"))
    conversation_id = conversations.create("cliente1988")
    state = initial_state("r", principal_for("cliente1988"), "q0", conversation_id, 1)
    state.update(answer="a0", current_agent="support")
    update = remember_node(state)
    stored = conversations.get(conversation_id, "cliente1988")
    assert stored.summary == "" and stored.summary_through_turn == 0
    assert update["trace"][-1].detail.endswith("+summary deferred")
