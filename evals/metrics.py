"""Ranking metrics for the retrieval evaluation."""

import math


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Return the share of relevant items found in the first `k` results."""
    if not relevant:
        return 0.0
    found = sum(1 for item in ranked[:k] if item in relevant)
    return found / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str]) -> float:
    """Return 1 / rank of the first relevant item, or 0 when none is ranked."""
    for position, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Return the normalised discounted cumulative gain with binary relevance."""
    gain = sum(
        1.0 / math.log2(position + 1)
        for position, item in enumerate(ranked[:k], 1)
        if item in relevant
    )
    ideal = sum(1.0 / math.log2(position + 1) for position in range(1, min(len(relevant), k) + 1))
    return gain / ideal if ideal else 0.0


def mean(values: list[float]) -> float:
    """Return the arithmetic mean, 0 for an empty list."""
    return sum(values) / len(values) if values else 0.0
