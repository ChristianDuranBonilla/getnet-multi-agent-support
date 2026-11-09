"""Similarity search over the Chroma index."""

from functools import lru_cache

from app import llm
from app.config import get_settings


class Retriever:
    """Thin wrapper around the Chroma collection used by the Knowledge agent."""

    def __init__(self) -> None:
        """Defer opening the collection until the first call."""
        self._store = None

    def _get_store(self):
        if self._store is None:
            from langchain_chroma import Chroma

            settings = get_settings()
            self._store = Chroma(
                collection_name=settings.chroma_collection,
                embedding_function=llm.embeddings(),
                persist_directory=str(settings.chroma_dir),
                collection_metadata={"hnsw:space": "cosine"},
            )
        return self._store

    def search(self, query: str, k: int | None = None) -> list[tuple[str, dict, float]]:
        """Return the closest chunks to the query.

        Args:
            query: User question.
            k: Number of results; defaults to `rag_top_k`.

        Returns:
            A list of (text, metadata, cosine_distance) tuples, closest first.
        """
        top_k = k or get_settings().rag_top_k
        results = self._get_store().similarity_search_with_score(query, k=top_k)
        return [(doc.page_content, dict(doc.metadata), float(score)) for doc, score in results]

    def count(self) -> int:
        """Return the number of chunks in the index, or 0 when it is unavailable."""
        try:
            return int(self._get_store()._collection.count())
        except Exception:
            return 0


@lru_cache
def get_retriever() -> Retriever:
    """Return the process-wide retriever."""
    return Retriever()
