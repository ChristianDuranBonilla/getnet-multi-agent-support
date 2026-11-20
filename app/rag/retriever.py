"""Similarity search over the Chroma index, optionally fused with BM25."""

from functools import lru_cache
from pathlib import Path

from app import llm
from app.config import get_settings
from app.rag.lexical import BM25Index, cosine_distance, reciprocal_rank_fusion

CANDIDATES = 20
LEXICAL_FILE = "lexical.json"


def lexical_path(chroma_dir: Path) -> Path:
    """Return where the BM25 index of a Chroma directory lives."""
    return Path(chroma_dir) / LEXICAL_FILE


class Retriever:
    """Thin wrapper around the Chroma collection used by the Knowledge agent.

    `RAG_MODE=dense` queries the vector index alone. `RAG_MODE=hybrid` also queries the
    BM25 index built at ingest time and fuses both rankings with reciprocal rank fusion;
    every returned chunk still carries its cosine distance so the relevance threshold
    keeps its meaning.
    """

    def __init__(self) -> None:
        """Defer opening the collection until the first call."""
        self._store = None
        self._lexical: BM25Index | None = None
        self._lexical_checked = False

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

    def _get_lexical(self) -> BM25Index | None:
        if not self._lexical_checked:
            self._lexical_checked = True
            path = lexical_path(get_settings().chroma_dir)
            if path.exists():
                self._lexical = BM25Index.load(path)
        return self._lexical

    def search(self, query: str, k: int | None = None) -> list[tuple[str, dict, float]]:
        """Return the best chunks for the query.

        Args:
            query: User question.
            k: Number of results; defaults to `rag_top_k`.

        Returns:
            A list of (text, metadata, cosine_distance) tuples, best first.
        """
        settings = get_settings()
        top_k = k or settings.rag_top_k
        if settings.rag_mode == "hybrid" and self._get_lexical() is not None:
            return self._search_hybrid(query, top_k, self._get_lexical())
        return self._search_dense(query, top_k)

    def _search_dense(self, query: str, top_k: int) -> list[tuple[str, dict, float]]:
        results = self._get_store().similarity_search_with_score(query, k=top_k)
        return [(doc.page_content, dict(doc.metadata), float(score)) for doc, score in results]

    def _search_hybrid(
        self, query: str, top_k: int, lexical: BM25Index
    ) -> list[tuple[str, dict, float]]:
        store = self._get_store()
        collection = store._collection
        embedding = store._embedding_function.embed_query(query)
        dense = collection.query(
            query_embeddings=[embedding],
            n_results=CANDIDATES,
            include=["documents", "metadatas", "distances"],
        )
        found: dict[str, tuple[str, dict, float]] = {}
        for doc_id, text, meta, distance in zip(
            dense["ids"][0],
            dense["documents"][0],
            dense["metadatas"][0],
            dense["distances"][0],
            strict=True,
        ):
            found[doc_id] = (text, dict(meta or {}), float(distance))
        lexical_ids = [doc_id for doc_id, _ in lexical.top(query, CANDIDATES)]
        fused = reciprocal_rank_fusion([list(found), lexical_ids])
        chosen = [doc_id for doc_id, _ in fused[:top_k]]
        missing = [doc_id for doc_id in chosen if doc_id not in found]
        if missing:
            extra = collection.get(ids=missing, include=["documents", "metadatas", "embeddings"])
            for doc_id, text, meta, vector in zip(
                extra["ids"],
                extra["documents"],
                extra["metadatas"],
                extra["embeddings"],
                strict=True,
            ):
                found[doc_id] = (text, dict(meta or {}), cosine_distance(embedding, list(vector)))
        return [found[doc_id] for doc_id in chosen if doc_id in found]

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
