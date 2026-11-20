"""Render the sheet a human labels to calibrate the judge.

Reads the newest regression and judge results and writes a Markdown file with, for every
scenario, the question, the reference answer, the evidence the answer was produced from
and the answer itself. The labeller writes groundedness and correctness (0 to 1) into
`evals/results/human_calibration.json`; `python evals/judge.py --calibrate` then reports
the agreement. The judge's own scores are printed last in each block so that labelling
is done before looking at them.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evals.judge import CALIBRATION, RESULTS_DIR, latest_results  # noqa: E402

DEFAULT_OUT = RESULTS_DIR / "CALIBRATION_SHEET.md"


def latest_judge() -> Path | None:
    """Return the newest judge report, if any."""
    files = sorted(RESULTS_DIR.glob("*-judge.json"))
    return files[-1] if files else None


def render(regression: dict, judge: dict | None) -> str:
    """Build the Markdown sheet."""
    scores = {row["id"]: row for row in (judge or {}).get("scenarios", [])}
    lines = [
        "# Calibration sheet (fill human_calibration.json from this)",
        "",
        "For each scenario, read the question, the evidence and the answer, then write your own",
        "groundedness and correctness (0.0 to 1.0) in `evals/results/human_calibration.json`.",
        "Label first; the judge's numbers close each block for comparison afterwards.",
        "",
        f"Regression run: {regression.get('run_at', '')}",
        "",
    ]
    for row in regression["results"]:
        if row.get("agent") is None and not row.get("answer"):
            continue
        question = " / ".join(row["turns"]) if row.get("turns") else row["message"]
        lines += [
            f"## {row['id']}. {question}",
            f"Merchant: {row.get('user_id', '')} | must_be_grounded: {row.get('must_be_grounded')}",
            "",
            f"Reference: {row.get('reference_answer', '')}",
            "",
            "Evidence:",
        ]
        for item in row.get("evidence") or []:
            excerpt = " ".join(item.get("excerpt", "").split())[:160]
            lines.append(f"- [{item['type']}] {item['ref']} :: {excerpt}")
        if not row.get("evidence"):
            lines.append("- (none)")
        lines += ["", "Answer:"]
        lines += [f"> {part}" for part in (row.get("answer") or "").splitlines()]
        judged = scores.get(row["id"])
        if judged:
            lines += [
                "",
                f"Judge: groundedness {judged['groundedness']}, correctness "
                f"{judged['correctness']}, unsupported claims {judged['unsupported_claims']}",
            ]
        lines.append("")
    return "\n".join(lines)


def reset_labels(regression: dict) -> None:
    """Write an empty label file when none exists, one entry per judged scenario."""
    if CALIBRATION.exists():
        return
    labels = {
        "_note": (
            "Human labels for the judge calibration: for each scenario id, groundedness and "
            "correctness on the same 0-1 scale, assigned by the author from the evidence in the "
            "regression results. null means not labelled yet. Run `python evals/judge.py "
            "--calibrate` to compare."
        )
    }
    for row in regression["results"]:
        if row.get("answer"):
            labels[str(row["id"])] = {"groundedness": None, "correctness": None}
    CALIBRATION.write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Render the human calibration sheet.")
    parser.add_argument("--results", help="Regression file; defaults to the newest full one.")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Where to write the sheet.")
    args = parser.parse_args(argv)
    regression_path = Path(args.results) if args.results else latest_results()
    regression = json.loads(regression_path.read_text(encoding="utf-8"))
    judge_path = latest_judge()
    judge = json.loads(judge_path.read_text(encoding="utf-8")) if judge_path else None
    Path(args.out).write_text(render(regression, judge), encoding="utf-8")
    reset_labels(regression)
    print(f"sheet written to {args.out} from {regression_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
