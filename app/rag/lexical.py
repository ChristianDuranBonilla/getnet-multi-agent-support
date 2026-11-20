"""Lexical retrieval: a small BM25 index over the same chunks as the vector index.

Kept dependency-free on purpose: the corpus is a few hundred chunks, so a plain
implementation is faster to load than a library and fully testable offline. Portuguese,
English and Spanish share the tokenizer, which folds accents so that "antecipação" and
"antecipacao" match.
"""

import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
MIN_TOKEN_LENGTH = 2
RRF_K = 60


def tokenize(text: str) -> list[str]:
    """Lowercase, fold accents and split into alphanumeric tokens.

    Args:
        text: Any text.

    Returns:
        Tokens of at least two characters.
    """
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(char for char in folded if not unicodedata.combining(char))
    return [token for token in TOKEN_PATTERN.findall(folded) if len(token) >= MIN_TOKEN_LENGTH]


class BM25Index:
    """Okapi BM25 over tokenized documents identified by string ids."""

    def __init__(
        self, ids: list[str], documents: list[list[str]], k1: float = 1.5, b: float = 0.75
    ) -> None:
        """Build the index from pre-tokenized documents.

        Args:
            ids: Document identifiers, aligned with `documents`.
            documents: Token lists.
            k1: Term-frequency saturation parameter.
            b: Length normalisation parameter.
        """
        self.ids = list(ids)
        self._documents = [Counter(tokens) for tokens in documents]
        self._lengths = [len(tokens) for tokens in documents]
        self._average_length = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0
        self._k1 = k1
        self._b = b
        frequencies: Counter[str] = Counter()
        for counts in self._documents:
            frequencies.update(counts.keys())
        total = len(self._documents)
        self._idf = {
            term: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for term, count in frequencies.items()
        }

    def __len__(self) -> int:
        """Return the number of indexed documents."""
        return len(self.ids)

    def scores(self, query: str) -> list[float]:
        """Return the BM25 score of every document for the query, aligned with `ids`."""
        tokens = tokenize(query)
        result = []
        for counts, length in zip(self._documents, self._lengths, strict=True):
            score = 0.0
            for token in tokens:
                frequency = counts.get(token)
                if not frequency:
                    continue
                idf = self._idf.get(token, 0.0)
                normaliser = self._k1 * (1 - self._b + self._b * length / self._average_length)
                score += idf * frequency * (self._k1 + 1) / (frequency + normaliser)
            result.append(score)
        return result

    def top(self, query: str, n: int) -> list[tuple[str, float]]:
        """Return the `n` best (id, score) pairs with a positive score, best first."""
        ranked = sorted(
            zip(self.ids, self.scores(query), strict=True), key=lambda pair: pair[1], reverse=True
        )
        return [(doc_id, score) for doc_id, score in ranked[:n] if score > 0]

    def save(self, path: Path) -> None:
        """Write the token lists to a JSON file."""
        payload = {
            "ids": self.ids,
            "documents": [list(counts.elements()) for counts in self._documents],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        """Read an index written by `save`."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload["ids"], payload["documents"])

    @classmethod
    def from_texts(cls, ids: list[str], texts: list[str]) -> "BM25Index":
        """Tokenize and index raw texts."""
        return cls(ids, [tokenize(text) for text in texts])


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Fuse several ranked id lists into one.

    Args:
        rankings: Ranked lists, best first; an id may appear in several lists.
        k: Smoothing constant; 60 is the value from the original paper.

    Returns:
        (id, fused_score) pairs, best first.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda pair: pair[1], reverse=True)


def cosine_distance(left: list[float], right: list[float]) -> float:
    """Return `1 - cosine similarity` of two vectors."""
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norm = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    if norm == 0:
        return 1.0
    return 1.0 - dot / norm
