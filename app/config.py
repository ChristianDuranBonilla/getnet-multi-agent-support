"""Application settings loaded from the environment."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration read from environment variables or a `.env` file.

    Attributes:
        google_genai_use_vertexai: Route Gemini calls through Vertex AI.
        google_cloud_project: GCP project id that owns the Vertex AI quota.
        google_cloud_location: Vertex AI location; Gemini 3.x is served from `global`.
        llm_model: Chat model id used by every agent.
        llm_timeout_seconds: Per-request timeout, which caps the intermittent stalls
            observed on the global endpoint.
        llm_max_retries: SDK-level retries of one request before the fallback model runs.
        llm_fallback_model: Model tried once when the primary raises (timeout, 5xx);
            empty disables the fallback.
        llm_price_input_per_mtok: List price in USD per million input tokens, for the
            cost estimate returned with every response.
        llm_price_output_per_mtok: List price in USD per million output tokens.
        embedding_model: Embedding model id used by the RAG index.
        chroma_dir: Directory of the persistent Chroma index.
        chroma_collection: Name of the Chroma collection.
        rag_top_k: Number of chunks retrieved per query.
        rag_distance_threshold: Maximum cosine distance for a chunk to count as relevant.
        rag_mode: `dense` uses the vector index alone; `hybrid` fuses it with BM25.
        state_dir: Directory of the SQLite files (conversations, tickets, checkpoints).
        operator_api_key: Credential of the human operator; compared by hash, never logged.
            Empty or the development placeholder never authenticates.
        rate_limit_per_minute: Sustained requests per minute allowed per API key.
        rate_limit_burst: Requests served back to back before the limiter applies.
        log_salt: Salt of the pseudonymous merchant reference written to logs.
        guardrail_fail_mode: What happens when the guardrail classifier is unavailable:
            `closed` answers with a templated unavailable message, `open` lets the
            message through with the deterministic checks only.
        memory_window_turns: Turns of each role loaded into the prompt as history.
        memory_summary_every: Turns after which the rolling summary is refreshed.
        max_hops: Maximum specialist-to-specialist hops in one turn.
        otel_exporter_otlp_endpoint: When set, spans are exported over OTLP.
        log_level: Logging level name.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    google_genai_use_vertexai: bool = True
    google_cloud_project: str = ""
    google_cloud_location: str = "global"
    llm_model: str = "gemini-3.8-flash"
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 1
    llm_fallback_model: str = "gemini-3.5-flash"
    llm_price_input_per_mtok: float = 0.50
    llm_price_output_per_mtok: float = 3.00
    embedding_model: str = "gemini-embedding-2"
    chroma_dir: Path = ROOT_DIR / "data" / "chroma"
    chroma_collection: str = "getnet_kb"
    rag_top_k: int = 5
    rag_distance_threshold: float = 0.38
    rag_mode: Literal["dense", "hybrid"] = "dense"
    state_dir: Path = ROOT_DIR / "data" / "state"
    operator_api_key: str = "op_dev_change_me"
    rate_limit_per_minute: float = 30.0
    rate_limit_burst: int = 10
    log_salt: str = "dev-salt"
    guardrail_fail_mode: Literal["closed", "open"] = "closed"
    memory_window_turns: int = 6
    memory_summary_every: int = 10
    max_hops: int = 1
    otel_exporter_otlp_endpoint: str = ""
    log_level: str = "INFO"

    @property
    def prompts_dir(self) -> Path:
        """Directory that holds one Markdown prompt per agent."""
        return ROOT_DIR / "app" / "prompts"

    @property
    def data_dir(self) -> Path:
        """Directory that holds mock data and raw crawled pages."""
        return ROOT_DIR / "data"

    @property
    def uses_default_secrets(self) -> bool:
        """Whether a development default is still in place for a secret setting."""
        return self.operator_api_key == "op_dev_change_me" or self.log_salt == "dev-salt"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
