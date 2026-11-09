"""Shared fixtures: fake models so that no test reaches Vertex AI."""

from datetime import date
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from app import llm
from app.config import ROOT_DIR
from app.repositories.json_repository import JsonSupportRepository


class FakeStructured:
    """Stands in for `llm.structured(schema)` and returns a fixed object."""

    def __init__(self, result):
        self.result = result
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
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

    def search(self, query, k=None):
        return self.hits[: k or len(self.hits)]

    def count(self):
        return self.size


@pytest.fixture
def repository() -> JsonSupportRepository:
    """Repository over the real mock data with a fixed reference date."""
    return JsonSupportRepository(ROOT_DIR / "data", today=date(2026, 11, 6))


@pytest.fixture
def data_dir() -> Path:
    """Path to the mock data directory."""
    return ROOT_DIR / "data"


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
