"""Retrieval quality on a held-out query set: recall@5, MRR, nDCG@5 and the threshold.

Runs the queries in `retrieval.jsonl` through the retriever in `dense` and `hybrid` mode
and writes `evals/results/<date>-retrieval.json`. Relevance is judged at page level: a
chunk counts when its page URL is one of the expected URLs. Queries marked `off_topic`
have no relevant page; they measure whether `RAG_DISTANCE_THRESHOLD` separates Getnet
questions (best relevant chunk distance) from unrelated ones (best chunk distance), and
how many unrelated chunks would still enter the prompt. Requires the index built by
`scripts/ingest.py` and Vertex AI credentials for the query embeddings.
"""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.rag import retriever as rag  # noqa: E402
from evals.metrics import mean, ndcg_at_k, recall_at_k, reciprocal_rank  # noqa: E402

QUERIES = Path(__file__).with_name("retrieval.jsonl")
RESULTS_DIR = Path(__file__).with_name("results")
K = 5


def load_queries() -> list[dict]:
    """Read the held-out queries."""
    with QUERIES.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def page_ranking(hits: list[tuple[str, dict, float]]) -> list[str]:
    """Collapse chunk hits to a ranking of distinct page URLs, best first."""
    pages: list[str] = []
    for _, meta, _ in hits:
        url = meta.get("url", "")
        if url and url not in pages:
            pages.append(url)
    return pages


def threshold_report(
    relevant_best: list[float], off_topic_best: list[float], off_topic_kept: list[int]
) -> dict:
    """Describe how the distance threshold separates on-topic from off-topic queries."""
    threshold = get_settings().rag_distance_threshold
    return {
        "threshold": threshold,
        "relevant_best_distance_max": round(max(relevant_best), 3) if relevant_best else None,
        "relevant_best_distance_median": (
            round(sorted(relevant_best)[len(relevant_best) // 2], 3) if relevant_best else None
        ),
        "relevant_lost_at_threshold": sum(1 for d in relevant_best if d > threshold),
        "off_topic_best_distance_min": round(min(off_topic_best), 3) if off_topic_best else None,
        "off_topic_chunks_kept_at_threshold": sum(off_topic_kept),
        "off_topic_queries": len(off_topic_best),
    }


def evaluate_mode(mode: str, queries: list[dict]) -> dict:
    """Run every query in one retrieval mode and aggregate the metrics."""
    settings = get_settings()
    settings.rag_mode = mode
    retriever = rag.Retriever()
    rows = []
    relevant_best: list[float] = []
    off_topic_best: list[float] = []
    off_topic_kept: list[int] = []
    for query in queries:
        hits = retriever.search(query["query"], K * 2)
        if query.get("off_topic"):
            distances = [distance for _, _, distance in hits]
            off_topic_best.append(min(distances) if distances else 1.0)
            off_topic_kept.append(
                sum(1 for distance in distances[:K] if distance <= settings.rag_distance_threshold)
            )
            continue
        relevant = set(query["expected_urls"])
        matching = [distance for _, meta, distance in hits if meta.get("url") in relevant]
        if matching:
            relevant_best.append(min(matching))
        ranking = page_ranking(hits)
        rows.append(
            {
                "id": query["id"],
                "lang": query["lang"],
                "recall_at_5": recall_at_k(ranking, relevant, K),
                "mrr": reciprocal_rank(ranking, relevant),
                "ndcg_at_5": ndcg_at_k(ranking, relevant, K),
                "top": ranking[:K],
            }
        )
    return {
        "mode": mode,
        "recall_at_5": round(mean([row["recall_at_5"] for row in rows]), 3),
        "mrr": round(mean([row["mrr"] for row in rows]), 3),
        "ndcg_at_5": round(mean([row["ndcg_at_5"] for row in rows]), 3),
        "by_language": {
            lang: round(mean([row["recall_at_5"] for row in rows if row["lang"] == lang]), 3)
            for lang in sorted({row["lang"] for row in rows})
        },
        "threshold": threshold_report(relevant_best, off_topic_best, off_topic_kept),
        "queries": rows,
    }


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Measure retrieval quality.")
    parser.add_argument("--modes", nargs="+", default=["dense", "hybrid"])
    args = parser.parse_args(argv)
    queries = load_queries()
    report = {"run_at": datetime.now(UTC).isoformat(timespec="seconds"), "k": K, "modes": {}}
    for mode in args.modes:
        result = evaluate_mode(mode, queries)
        report["modes"][mode] = result
        print(
            f"{mode:<7} recall@5={result['recall_at_5']:.3f} mrr={result['mrr']:.3f} "
            f"ndcg@5={result['ndcg_at_5']:.3f} by language {result['by_language']}"
        )
        gap = result["threshold"]
        print(
            f"        threshold {gap['threshold']}: relevant best distance max "
            f"{gap['relevant_best_distance_max']} (lost {gap['relevant_lost_at_threshold']}), "
            f"off-topic best distance min {gap['off_topic_best_distance_min']} "
            f"({gap['off_topic_chunks_kept_at_threshold']} off-topic chunks kept over "
            f"{gap['off_topic_queries']} queries)"
        )
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now(UTC).date().isoformat()}-retrieval.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"retrieval report written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
