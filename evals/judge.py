"""Semantic evaluation of the regression answers with an evidence-based judge.

Reads the newest `evals/results/<date>-regression.json` (or the file given with
`--results`), asks Gemini to score every answer against the evidence it was produced
from and the hand-written reference answer, and writes
`evals/results/<date>-judge.json`. `--calibrate` compares the judge with the human labels
in `evals/results/human_calibration.json`. Means are computed over the scenarios whose
answer must be grounded in evidence; guardrail and operator scenarios
(`must_be_grounded: false`) are reported apart so they do not dilute the measure, and
scenarios whose expected outcome is an HTTP status (401, 403) are verified by the
regression runner and listed as `status_checks` instead of being scored. A
scenario the judge could not score after retries is reported as unjudged and fails the
run instead of being scored as zero.
"""

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import llm  # noqa: E402

RESULTS_DIR = Path(__file__).with_name("results")
PROMPT = Path(__file__).with_name("prompts") / "judge.md"
CALIBRATION = RESULTS_DIR / "human_calibration.json"
MIN_MEAN_GROUNDEDNESS = 0.85
MIN_SCENARIO_GROUNDEDNESS = 0.6
MIN_MEAN_CORRECTNESS = 0.8
JUDGE_ATTEMPTS = 3


class JudgeVerdict(BaseModel):
    """Structured output of the judge."""

    groundedness: float = Field(ge=0, le=1)
    correctness: float = Field(ge=0, le=1)
    relevance: float = Field(ge=0, le=1)
    usefulness: float = Field(ge=0, le=1)
    unsupported_claims: list[str] = Field(default_factory=list, max_length=20)
    rationale: str = Field(default="", max_length=400)
    error: str = Field(default="", exclude=True)


def latest_results() -> Path:
    """Return the newest full regression result file."""
    files = sorted(
        path for path in RESULTS_DIR.glob("*-regression.json") if "partial" not in path.name
    )
    if not files:
        raise FileNotFoundError("no regression results; run evals/run_eval.py first")
    return files[-1]


def format_evidence(evidence: list[dict]) -> str:
    """Render the evidence list for the judge prompt."""
    if not evidence:
        return "(no evidence was retrieved or returned by tools)"
    lines = []
    for index, item in enumerate(evidence, start=1):
        distance = f" distance={item['distance']:.2f}" if item.get("distance") is not None else ""
        lines.append(f"[{index}] {item['type']} {item['ref']}{distance}\n{item.get('excerpt', '')}")
    return "\n\n".join(lines)


def judge_one(result: dict) -> JudgeVerdict:
    """Score one regression result.

    Args:
        result: One entry of the regression file.

    Returns:
        The judge's verdict, or a verdict with `error` set when the judge could not be
        reached after `JUDGE_ATTEMPTS` attempts.
    """
    user = (
        f"QUESTION:\n{result['message']}\n\n"
        f"EVIDENCE:\n{format_evidence(result.get('evidence', []))}\n\n"
        f"REFERENCE:\n{result.get('reference_answer', '')}\n\n"
        f"ANSWER:\n{result.get('answer', '')}"
    )
    messages = [
        SystemMessage(content=PROMPT.read_text(encoding="utf-8")),
        HumanMessage(content=user),
    ]
    error = ""
    for attempt in range(JUDGE_ATTEMPTS):
        try:
            verdict = llm.invoke_resilient(llm.structured(JudgeVerdict), messages)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"[:200]
            time.sleep(2 * (attempt + 1))
            continue
        if isinstance(verdict, dict):
            verdict = JudgeVerdict.model_validate(verdict)
        return verdict
    # An unreachable judge is an outage, not a verdict: the scenario is reported as
    # unjudged, excluded from the means, and the run fails.
    return JudgeVerdict(
        groundedness=0, correctness=0, relevance=0, usefulness=0, rationale="", error=error
    )


def mean(values: list[float]) -> float:
    """Return the arithmetic mean, 0 for an empty list."""
    return sum(values) / len(values) if values else 0.0


def evaluate(results_file: Path) -> tuple[dict, bool]:
    """Judge every answered scenario and apply the pass criteria.

    Args:
        results_file: Regression result file.

    Returns:
        The report and whether the pass criteria hold.
    """
    data = json.loads(results_file.read_text(encoding="utf-8"))
    rows = []
    status_checks = []
    for result in data["results"]:
        if not result.get("answer") or result["failures"] and "exception" in result["failures"][0]:
            continue
        if result.get("expected_status"):
            # An HTTP status (401, 403) is the whole answer: the regression runner checked
            # it and there is no text for a judge to score.
            status_checks.append(result["id"])
            print(f"{result['id']:>3}  HTTP {result['expected_status']} verified by the runner")
            continue
        verdict = judge_one(result)
        rows.append(
            {
                "id": result["id"],
                "must_be_grounded": result.get("must_be_grounded", True),
                **verdict.model_dump(),
                "judge_error": verdict.error,
                "answer": result["answer"],
            }
        )
        if verdict.error:
            print(f"{result['id']:>3}  UNJUDGED  {verdict.error[:90]}")
            continue
        print(
            f"{result['id']:>3}  g={verdict.groundedness:.2f} c={verdict.correctness:.2f} "
            f"r={verdict.relevance:.2f} u={verdict.usefulness:.2f}  "
            f"{'; '.join(verdict.unsupported_claims)[:80]}"
        )
    judged = [row for row in rows if not row["judge_error"]]
    unjudged = [row["id"] for row in rows if row["judge_error"]]
    grounded = [row for row in judged if row["must_be_grounded"]]
    policy = [row for row in judged if not row["must_be_grounded"]]
    report = {
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": results_file.name,
        "grounded_scenarios": len(grounded),
        "policy_scenarios": len(policy),
        "mean_groundedness": round(mean([row["groundedness"] for row in grounded]), 3),
        "mean_correctness": round(mean([row["correctness"] for row in grounded]), 3),
        "mean_relevance": round(mean([row["relevance"] for row in grounded]), 3),
        "mean_usefulness": round(mean([row["usefulness"] for row in grounded]), 3),
        "min_groundedness": round(min((row["groundedness"] for row in grounded), default=0), 3),
        "policy_mean_correctness": round(mean([row["correctness"] for row in policy]), 3),
        "unjudged": unjudged,
        "status_checks": status_checks,
        "scenarios": rows,
    }
    passed = (
        not unjudged
        and report["mean_groundedness"] >= MIN_MEAN_GROUNDEDNESS
        and report["min_groundedness"] >= MIN_SCENARIO_GROUNDEDNESS
        and report["mean_correctness"] >= MIN_MEAN_CORRECTNESS
    )
    report["passed"] = passed
    print(
        f"\n{len(grounded)} grounded scenarios: groundedness {report['mean_groundedness']} "
        f"(min {report['min_groundedness']}), correctness {report['mean_correctness']}, "
        f"relevance {report['mean_relevance']}, usefulness {report['mean_usefulness']}; "
        f"{len(policy)} policy scenarios: correctness {report['policy_mean_correctness']}; "
        f"unjudged {unjudged or 'none'} -> {'PASS' if passed else 'FAIL'}"
    )
    return report, passed


def calibrate(judge_file: Path) -> int:
    """Compare the judge with the human labels and print the agreement.

    Args:
        judge_file: Judge report to compare.

    Returns:
        0 when labels exist for at least one scenario, 1 otherwise.
    """
    if not CALIBRATION.exists():
        print(f"no human labels at {CALIBRATION}")
        return 1
    labels = {
        int(key): value
        for key, value in json.loads(CALIBRATION.read_text(encoding="utf-8")).items()
        if key.isdigit() and value.get("groundedness") is not None
    }
    report = json.loads(judge_file.read_text(encoding="utf-8"))
    pairs = [
        (row, labels[row["id"]])
        for row in report["scenarios"]
        if row["id"] in labels and not row.get("judge_error")
    ]
    if not pairs:
        print("no labelled scenarios yet; fill evals/results/human_calibration.json")
        return 1
    for dimension in ("groundedness", "correctness"):
        diffs = [row[dimension] - label[dimension] for row, label in pairs]
        print(
            f"{dimension}: mean absolute error {mean([abs(d) for d in diffs]):.3f}, "
            f"judge minus human {mean(diffs):+.3f} over {len(pairs)} scenarios"
        )
    agree = sum(
        1
        for row, label in pairs
        if (row["groundedness"] >= MIN_SCENARIO_GROUNDEDNESS)
        == (label["groundedness"] >= MIN_SCENARIO_GROUNDEDNESS)
    )
    print(f"pass/fail agreement on groundedness: {agree}/{len(pairs)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Judge the regression answers.")
    parser.add_argument("--results", help="Regression file; defaults to the newest full one.")
    parser.add_argument("--calibrate", action="store_true", help="Compare with human labels.")
    parser.add_argument("--judge-file", help="Judge report for --calibrate; defaults to newest.")
    parser.add_argument("--out", help="Report file; defaults to evals/results/<date>-judge.json")
    args = parser.parse_args(argv)
    if args.calibrate:
        if args.judge_file:
            return calibrate(Path(args.judge_file))
        files = sorted(RESULTS_DIR.glob("*-judge.json"))
        if not files:
            print("no judge report; run evals/judge.py first")
            return 1
        return calibrate(files[-1])
    results_file = Path(args.results) if args.results else latest_results()
    report, passed = evaluate(results_file)
    out = (
        Path(args.out)
        if args.out
        else RESULTS_DIR / f"{datetime.now(UTC).date().isoformat()}-judge.json"
    )
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"judge report written to {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
