"""Regression evaluation over the challenge scenarios.

Runs every line of `scenarios.jsonl` either through the graph in-process (default,
Vertex AI required) or against a running API with `--api URL`, and checks routing,
handoff, guardrail category, sources and answer keywords.
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph import get_graph, resolved_agent  # noqa: E402
from app.state import Source, ToolCall, initial_state  # noqa: E402

SCENARIOS = Path(__file__).with_name("scenarios.jsonl")
LAST_RUN = Path(__file__).with_name("last_run.json")


def load_scenarios() -> list[dict]:
    """Read the JSONL scenarios file."""
    with SCENARIOS.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def check(scenario: dict, final: dict) -> list[str]:
    """Compare the final state with the scenario expectations.

    Args:
        scenario: One line of `scenarios.jsonl`.
        final: Final graph state.

    Returns:
        A list of failure descriptions; empty when the scenario passes.
    """
    failures = []
    agent = resolved_agent(final)
    route = final.get("route") if agent else None
    answer = (final.get("answer") or "").lower()
    if "expected_agent" in scenario and scenario["expected_agent"] != agent:
        failures.append(f"agent {agent!r} != {scenario['expected_agent']!r}")
    if "expected_intent" in scenario and route and route.intent != scenario["expected_intent"]:
        failures.append(f"intent {route.intent!r} != {scenario['expected_intent']!r}")
    handoff = bool(final.get("handoff")) if agent else False
    if "expected_handoff" in scenario and handoff != scenario["expected_handoff"]:
        failures.append(f"handoff {handoff} != {scenario['expected_handoff']}")
    verdict = final.get("guardrail")
    if "expected_guardrail" in scenario and (
        not verdict or verdict.category != scenario["expected_guardrail"]
    ):
        category = verdict.category if verdict else None
        failures.append(f"guardrail {category!r} != {scenario['expected_guardrail']!r}")
    keywords = scenario.get("must_contain_any", [])
    if keywords and not any(word.lower() in answer for word in keywords):
        failures.append(f"answer lacks any of {keywords}")
    for word in scenario.get("must_not_contain", []):
        if word.lower() in answer:
            failures.append(f"answer contains forbidden {word!r}")
    if "expected_tool" in scenario:
        names = [call.name for call in final.get("tool_calls", [])]
        if scenario["expected_tool"] not in names:
            failures.append(f"tool {scenario['expected_tool']!r} not called ({names})")
    if "expected_source_type" in scenario:
        types = {source.type for source in final.get("sources", [])}
        if scenario["expected_source_type"] not in types:
            failures.append(
                f"no source of type {scenario['expected_source_type']!r} ({sorted(types)})"
            )
    return failures


def state_from_response(body: dict) -> dict:
    """Rebuild the subset of the final state that `check` needs from an API response.

    Args:
        body: JSON returned by `POST /chat`.

    Returns:
        A state-like dict with `route`, `guardrail`, `handoff`, `answer`, `sources`
        and `tool_calls`.
    """
    agent = body.get("agent")
    route = SimpleNamespace(agent=agent, intent=body.get("intent")) if agent else None
    category = body.get("guardrail") or (None if agent else body.get("intent"))
    guardrail = SimpleNamespace(category=category) if category else None
    sources = [Source(**item) for item in body.get("sources", [])]
    tool_calls = [
        ToolCall(name=item.ref, args={}, ok=True) for item in sources if item.type == "tool"
    ]
    return {
        "route": route,
        "guardrail": guardrail,
        "handoff": body.get("handoff", False),
        "answer": body.get("response", ""),
        "sources": sources,
        "tool_calls": tool_calls,
    }


def run_scenario(scenario: dict, api: str | None, graph) -> dict:
    """Execute one scenario in-process or over HTTP.

    Args:
        scenario: One line of `scenarios.jsonl`.
        api: Base URL of a running service, or `None` to use the graph directly.
        graph: Compiled graph, used when `api` is `None`.

    Returns:
        The final state, or a state-like dict built from the API response.
    """
    if api is None:
        state = initial_state(uuid.uuid4().hex, scenario["user_id"], scenario["message"])
        return graph.invoke(state)
    payload = {"message": scenario["message"], "user_id": scenario["user_id"]}
    response = httpx.post(f"{api.rstrip('/')}/chat", json=payload, timeout=180.0)
    response.raise_for_status()
    return state_from_response(response.json())


def main(argv: list[str] | None = None) -> int:
    """Run every scenario and print a summary table.

    Args:
        argv: Command-line arguments; `--api URL` evaluates a running service.

    Returns:
        0 when all scenarios pass, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="Evaluate the challenge scenarios.")
    parser.add_argument("--api", help="Base URL of a running service, e.g. http://localhost:8080")
    args = parser.parse_args(argv)
    graph = None if args.api else get_graph()
    results = []
    for scenario in load_scenarios():
        started = time.perf_counter()
        try:
            final = run_scenario(scenario, args.api, graph)
            failures = check(scenario, final)
            agent = resolved_agent(final) or "-"
            handoff = bool(final.get("handoff")) if agent != "-" else False
            answer = final.get("answer") or ""
        except Exception as exc:
            failures, agent, handoff, answer = (
                [f"exception {type(exc).__name__}: {exc}"],
                "-",
                False,
                "",
            )
        latency = int((time.perf_counter() - started) * 1000)
        results.append(
            {
                "id": scenario["id"],
                "message": scenario["message"],
                "expected_agent": scenario.get("expected_agent"),
                "agent": agent,
                "handoff": handoff,
                "latency_ms": latency,
                "failures": failures,
                "answer": answer,
            }
        )
        status = "PASS" if not failures else "FAIL"
        print(
            f"{scenario['id']:>3}  {status}  {str(scenario.get('expected_agent')):<10} {agent:<10} "
            f"handoff={str(handoff):<5} {latency:>6} ms  {'; '.join(failures)}"
        )
    passed = sum(1 for item in results if not item["failures"])
    print(f"\n{passed}/{len(results)} scenarios passed")
    LAST_RUN.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
