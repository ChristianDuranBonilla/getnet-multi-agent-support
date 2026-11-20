"""Factories for chat models and embeddings.

Every agent obtains its model through this module so that tests can replace the
factories with fakes and so that model ids live in one place. Nothing here talks to
Vertex AI at import time.
"""

import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from app.config import get_settings
from app.observability import model_of, usage_of

GOOGLE_SEARCH_TOOL: dict = {"google_search": {}}


ThinkingLevel = Literal["low", "medium", "high"]


@lru_cache
def _base_model(thinking: ThinkingLevel, model_name: str) -> BaseChatModel:
    """Build a Gemini client once per (thinking level, model) and reuse it.

    Every `ChatGoogleGenerativeAI` instance opens its own HTTP client and fetches an
    access token on its first request, which cost close to one second per graph node
    when a fresh instance was built for every call.

    Args:
        thinking: Reasoning depth passed as `thinking_level`.
        model_name: Gemini model id.

    Returns:
        A chat model that is safe to share between calls and threads.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI

    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=model_name,
        project=settings.google_cloud_project,
        location=settings.google_cloud_location,
        vertexai=settings.google_genai_use_vertexai,
        thinking_level=thinking,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )


def _candidates(thinking: ThinkingLevel) -> list[BaseChatModel]:
    """Return the primary model followed by the fallback model, when one is configured."""
    settings = get_settings()
    models = [_base_model(thinking, settings.llm_model)]
    fallback = settings.llm_fallback_model
    if fallback and fallback != settings.llm_model:
        models.append(_base_model(thinking, fallback))
    return models


def _with_fallbacks(runnables: list[Runnable]) -> Runnable:
    """Chain runnables so that the next one runs when the previous raises."""
    primary, *others = runnables
    return primary.with_fallbacks(others) if others else primary


def chat(tools: Sequence | None = None, thinking: ThinkingLevel = "low") -> Runnable:
    """Return a Gemini chat model, optionally bound to tools, with the fallback model.

    The clients are shared across calls (see `_base_model`); binding tools returns a
    lightweight wrapper and never mutates the shared instances. When the primary model
    raises (timeout, 5xx from the endpoint), the same request is sent once to
    `LLM_FALLBACK_MODEL`; the model that answered is visible in the trace.

    Temperature is left at the model default on purpose: Google advises against
    lowering it on Gemini 3 models, where it can cause degraded reasoning or loops.

    Args:
        tools: LangChain tools or Gemini built-in tool specs to bind.
        thinking: Reasoning depth. `low` is the lightest level this model accepts.

    Returns:
        A runnable that accepts a list of messages.
    """
    models = _candidates(thinking)
    if tools:
        models = [model.bind_tools(list(tools)) for model in models]
    return _with_fallbacks(models)


class StructuredModel:
    """Runnable that returns parsed structured output and remembers the token usage.

    `with_structured_output(..., include_raw=True)` yields the raw message next to the
    parsed object; this wrapper hands back the parsed object, as callers expect, and
    keeps the usage of the last call in `last_usage` for the trace.
    """

    def __init__(self, runnable: Runnable) -> None:
        """Wrap a structured-output runnable."""
        self._runnable = runnable
        self.last_usage: tuple[int, int] = (0, 0)
        self.last_model: str = ""

    def invoke(self, messages):
        """Run the model and return the parsed object.

        Raises:
            Exception: The parsing error reported by LangChain when the output could not
                be parsed into the schema.
        """
        output = self._runnable.invoke(messages)
        if isinstance(output, dict) and "parsed" in output:
            raw = output.get("raw")
            self.last_usage = usage_of(raw) if raw is not None else (0, 0)
            self.last_model = model_of(raw) if raw is not None else ""
            if output.get("parsed") is None and output.get("parsing_error") is not None:
                raise output["parsing_error"]
            return output["parsed"]
        return output


def _structured_runnable(model: BaseChatModel, schema: type[BaseModel]) -> Runnable:
    try:
        return model.with_structured_output(schema, method="json_schema", include_raw=True)
    except (TypeError, ValueError):
        return model.with_structured_output(schema, include_raw=True)


def structured(schema: type[BaseModel]) -> StructuredModel:
    """Return a Gemini chat model constrained to a Pydantic schema, with the fallback model.

    Structured output cannot be combined with Gemini built-in tools such as
    google_search, which is why this factory never binds tools.

    Args:
        schema: Pydantic model describing the expected JSON.

    Returns:
        A runnable whose `invoke` returns an instance of `schema`.
    """
    runnables = [_structured_runnable(model, schema) for model in _candidates("low")]
    return StructuredModel(_with_fallbacks(runnables))


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

                if self._vertexai:
                    self._client = genai.Client(
                        vertexai=True, project=self._project, location=self._location
                    )
                else:
                    # Gemini Developer API: the key comes from GOOGLE_API_KEY / GEMINI_API_KEY
                    # and the client refuses project/location.
                    self._client = genai.Client(vertexai=False)
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


QUICK_FAILURE_SECONDS = 10.0


def invoke_resilient(runnable, messages, quick_failure_seconds: float = QUICK_FAILURE_SECONDS):
    """Invoke a model and retry once when it failed quickly.

    A fast failure (connection reset, 5xx returned at once) is almost always transient
    on the shared endpoint, so one retry after half a second recovers it cheaply. A slow
    failure (a timeout) is not retried here: the SDK retry and the fallback model have
    already spent their budget and the caller's degraded path is the right answer.

    Args:
        runnable: Anything with `invoke(messages)`.
        messages: The messages to send.
        quick_failure_seconds: Failures slower than this are not retried.

    Returns:
        The runnable's result.
    """
    started = time.perf_counter()
    try:
        return runnable.invoke(messages)
    except Exception:
        if time.perf_counter() - started > quick_failure_seconds:
            raise
        time.sleep(0.5)
        return runnable.invoke(messages)


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
