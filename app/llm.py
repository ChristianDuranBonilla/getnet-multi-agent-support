"""Factories for chat models and embeddings.

Every agent obtains its model through this module so that tests can replace the
factories with fakes and so that model ids live in one place. Nothing here talks to
Vertex AI at import time.
"""

import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from app.config import get_settings

GOOGLE_SEARCH_TOOL: dict = {"google_search": {}}


ThinkingLevel = Literal["low", "medium", "high"]


@lru_cache
def _base_model(thinking: ThinkingLevel) -> BaseChatModel:
    """Build the Gemini client once per thinking level and reuse it.

    Every `ChatGoogleGenerativeAI` instance opens its own HTTP client and fetches an
    access token on its first request, which cost close to one second per graph node
    when a fresh instance was built for every call.

    Args:
        thinking: Reasoning depth passed as `thinking_level`.

    Returns:
        A chat model that is safe to share between calls and threads.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
        vertexai=settings.google_genai_use_vertexai,
        thinking_level=thinking,
        timeout=settings.llm_timeout_seconds,
        max_retries=3,
    )


def chat(tools: Sequence | None = None, thinking: ThinkingLevel = "low") -> Runnable:
    """Return a Gemini chat model, optionally bound to tools.

    The client is shared across calls (see `_base_model`); binding tools returns a
    lightweight wrapper and never mutates the shared instance.

    Temperature is left at the model default on purpose: Google advises against
    lowering it on Gemini 3 models, where it can cause degraded reasoning or loops.

    Args:
        tools: LangChain tools or Gemini built-in tool specs to bind.
        thinking: Reasoning depth. `low` is the lightest level this model accepts.

    Returns:
        A runnable that accepts a list of messages.
    """
    model = _base_model(thinking)
    if tools:
        return model.bind_tools(list(tools))
    return model


def structured(schema: type[BaseModel]) -> Runnable:
    """Return a Gemini chat model constrained to a Pydantic schema.

    Structured output cannot be combined with Gemini built-in tools such as
    google_search, which is why this factory never binds tools.

    Args:
        schema: Pydantic model describing the expected JSON.

    Returns:
        A runnable that returns an instance of `schema`.
    """
    model = chat(thinking="low")
    try:
        return model.with_structured_output(schema, method="json_schema")
    except (TypeError, ValueError):
        return model.with_structured_output(schema)


class GeminiEmbeddings(Embeddings):
    """LangChain embeddings backed by the google-genai client on Vertex AI.

    Documents and queries use different task types, which improves retrieval quality
    with the Gemini embedding models.
    """

    def __init__(self, model: str, project: str, location: str, vertexai: bool) -> None:
        """Store the client configuration; the client is created lazily.

        Args:
            model: Embedding model id.
            project: GCP project id.
            location: Vertex AI location.
            vertexai: Whether to route through Vertex AI.
        """
        self._model = model
        self._project = project
        self._location = location
        self._vertexai = vertexai
        self._client = None
        self._lock = threading.Lock()

    def _get_client(self):
        with self._lock:
            if self._client is None:
                from google import genai

                self._client = genai.Client(
                    vertexai=self._vertexai, project=self._project, location=self._location
                )
            return self._client

    def _embed_one(self, text: str, task_type: str) -> list[float]:
        from google.genai import types

        result = self._get_client().models.embed_content(
            model=self._model,
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type),
        )
        return list(result.embeddings[0].values)

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        # Vertex AI serves one embedding per request for this model, so requests are
        # issued individually and in parallel instead of in batches.
        self._get_client()
        with ThreadPoolExecutor(max_workers=6) as pool:
            return list(pool.map(lambda text: self._embed_one(text, task_type), texts))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents for indexing."""
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        """Embed a query for retrieval."""
        return self._embed([text], "RETRIEVAL_QUERY")[0]


def embeddings() -> Embeddings:
    """Return the embeddings model configured in settings."""
    settings = get_settings()
    return GeminiEmbeddings(
        model=settings.embedding_model,
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
        vertexai=settings.google_genai_use_vertexai,
    )


def message_text(content) -> str:
    """Flatten a LangChain message content into plain text.

    Gemini returns a string for plain answers and a list of parts when built-in
    tools such as google_search are involved.

    Args:
        content: The `content` attribute of an AIMessage.

    Returns:
        The concatenated text parts.
    """
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif isinstance(part, dict) and part.get("type", "text") == "text":
            parts.append(str(part.get("text", "")))
    return "".join(parts).strip()
