"""Application settings loaded from the environment."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration read from environment variables or a `.env` file.

    Attributes:
        google_genai_use_vertexai: Route Gemini calls through Vertex AI.
        google_cloud_project: GCP project id that owns the Vertex AI quota.
        google_cloud_location: Vertex AI location; Gemini 3.x is served from `global`.
        llm_model: Chat model id used by every agent.
        llm_timeout_seconds: Per-request timeout; the SDK retries on timeout, which caps
            the intermittent stalls observed on the global endpoint.
        embedding_model: Embedding model id used by the RAG index.
        chroma_dir: Directory of the persistent Chroma index.
        chroma_collection: Name of the Chroma collection.
        rag_top_k: Number of chunks retrieved per query.
        rag_distance_threshold: Maximum cosine distance for a chunk to count as relevant.
        log_level: Logging level name.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    google_genai_use_vertexai: bool = True
    google_cloud_project: str = ""
    google_cloud_location: str = "global"
    llm_model: str = "gemini-3.8-flash"
    llm_timeout_seconds: float = 30.0
    embedding_model: str = "gemini-embedding-2"
    chroma_dir: Path = ROOT_DIR / "data" / "chroma"
    chroma_collection: str = "getnet_kb"
    rag_top_k: int = 5
    rag_distance_threshold: float = 0.38
    log_level: str = "INFO"

    @property
    def prompts_dir(self) -> Path:
        """Directory that holds one Markdown prompt per agent."""
        return ROOT_DIR / "app" / "prompts"

    @property
    def data_dir(self) -> Path:
        """Directory that holds mock data and raw crawled pages."""
        return ROOT_DIR / "data"


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance."""
    return Settings()
