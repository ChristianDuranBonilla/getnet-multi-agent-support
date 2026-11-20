"""Shared fixtures: fake models, temporary SQLite state, no network, no credentials."""

import json
from datetime import date
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app import auth, llm, repositories
from app.auth import Principal, RateLimiter
from app.config import ROOT_DIR, get_settings
from app.graph import build_graph, checkpoint_serializer
from app.repositories.json_repository import JsonSupportRepository
from app.repositories.sqlite_repository import (
    SqliteConversationRepository,
    SqliteStore,
    SqliteTicketRepository,
)

DEMO_KEYS = json.loads((ROOT_DIR / "evals" / "demo_keys.json").read_text(encoding="utf-8"))
OPERATOR_KEY = "op_test_8f1c2d3e4b5a6978"


class FakeStructured:
    """Stands in for `llm.structured(schema)` and returns a fixed object."""

    def __init__(self, result):
        self.result = result
        self.calls = []
        self.last_usage = (11, 3)

    def invoke(self, messages):
        self.calls.append(messages)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class FakeChat:
    """Stands in for `llm.chat()`; returns scripted AIMessages in order."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if len(self.responses) > 1:
            return self.responses.pop(0)
        return self.responses[0]


class FakeRetriever:
    """Returns pre-defined hits regardless of the query."""

    def __init__(self, hits=None, size=0):
        self.hits = hits or []
        self.size = size
        self.queries = []

    def search(self, query, k=None):
        self.queries.append(query)
        return self.hits[: k or len(self.hits)]

    def count(self):
        return self.size


def principal_for(merchant_id: str) -> Principal:
    """Build a merchant principal for tests."""
    return Principal(
        merchant_id=merchant_id, auth_method="api_key", scopes=("chat",), key_id="test"
    )


def tool_turn(name: str, args: dict, call_id: str = "call-1") -> AIMessage:
    """Build an AIMessage that requests one tool call."""
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}]
    )


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch, tmp_path):
    """Point conversations and tickets at a temporary SQLite file and relax the rate limit."""
    store = SqliteStore(tmp_path / "app.sqlite")
    conversations = SqliteConversationRepository(store)
    tickets = SqliteTicketRepository(store)
    monkeypatch.setattr(repositories, "get_conversation_repository", lambda: conversations)
    monkeypatch.setattr(repositories, "get_ticket_repository", lambda: tickets)
    monkeypatch.setattr(auth, "get_rate_limiter", lambda: RateLimiter(10**9, 10**6))
    monkeypatch.setattr(get_settings(), "operator_api_key", OPERATOR_KEY)
    monkeypatch.setattr(get_settings(), "state_dir", tmp_path)
    return {"conversations": conversations, "tickets": tickets}


@pytest.fixture
def conversations(isolated_state):
    """Temporary conversation repository."""
    return isolated_state["conversations"]


@pytest.fixture
def tickets(isolated_state):
    """Temporary ticket repository."""
    return isolated_state["tickets"]


@pytest.fixture
def repository() -> JsonSupportRepository:
    """Repository over the real mock data with a fixed reference date."""
    return JsonSupportRepository(ROOT_DIR / "data", today=date(2026, 11, 6))


@pytest.fixture
def use_repository(monkeypatch, repository):
    """Make the tools read the mock data with the fixed reference date."""
    monkeypatch.setattr(repositories, "get_repository", lambda: repository)
    return repository


@pytest.fixture
def data_dir() -> Path:
    """Path to the mock data directory."""
    return ROOT_DIR / "data"


@pytest.fixture
def principal() -> Principal:
    """Principal of the first demo merchant."""
    return principal_for("cliente1988")


@pytest.fixture
def demo_key() -> str:
    """Plaintext demo key of the first demo merchant."""
    return DEMO_KEYS["cliente1988"]


@pytest.fixture
def use_structured(monkeypatch):
    """Patch `llm.structured` to return the given object."""

    def _apply(result):
        fake = FakeStructured(result)
        monkeypatch.setattr(llm, "structured", lambda schema: fake)
        return fake

    return _apply


@pytest.fixture
def use_structured_map(monkeypatch):
    """Patch `llm.structured` to answer by schema, independent of call order."""

    def _apply(results: dict):
        fakes = {schema: FakeStructured(result) for schema, result in results.items()}
        monkeypatch.setattr(llm, "structured", lambda schema: fakes[schema])
        return fakes

    return _apply


@pytest.fixture
def use_chat(monkeypatch):
    """Patch `llm.chat` to return scripted messages."""

    def _apply(*responses):
        fake = FakeChat(responses or [AIMessage(content="ok")])
        monkeypatch.setattr(llm, "chat", lambda tools=None, thinking="low": fake)
        return fake

    return _apply


@pytest.fixture
def checkpointed_graph():
    """Graph compiled with an in-memory checkpointer, so interrupts can resume."""
    return build_graph(InMemorySaver(serde=checkpoint_serializer()))
