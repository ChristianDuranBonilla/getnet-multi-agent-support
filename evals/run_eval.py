"""Regression evaluation over the challenge scenarios.

Runs every line of `scenarios.jsonl` either through the graph in-process (default,
Vertex AI required) or against a running API with `--api URL`, and checks routing
(`expected_agent` is the first specialist the Router chose; `expected_final_agent` the
one that wrote the answer after any hop), plan and hops, handoff, guardrail category,
tools, sources, security events and answer keywords. Multi-turn scenarios (`turns`)
share one conversation; the operator round trip and the HTTP status scenarios run only
against the API. Results, including the evidence each answer was produced from, are
written to `evals/results/<date>-regression.json` so that `judge.py` can score them and
the repository keeps a versioned record.
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import repositories  # noqa: E402
from app.auth import Principal  # noqa: E402
from app.graph import get_graph, new_thread_id, resolved_agent, thread_config  # noqa: E402
from app.state import Source, ToolCall, initial_state  # noqa: E402

SCENARIOS = Path(__file__).with_name("scenarios.jsonl")
DEMO_KEYS = Path(__file__).with_name("demo_keys.json")
RESULTS_DIR = Path(__file__).with_name("results")


def load_scenarios() -> list[dict]:
    """Read the JSONL scenarios file."""
    with SCENARIOS.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_keys(path: str | None) -> dict[str, str]:
    """Read the merchant API keys for black-box runs.

    Args:
        path: JSON file with `{merchant_id: key}`; when `None`, the `EVAL_API_KEYS`
            environment variable (same JSON) is used, then `evals/demo_keys.json`.

    Returns:
        Mapping from merchant id to plaintext key.
    """
    if path:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    elif os.environ.get("EVAL_API_KEYS"):
        data = json.loads(os.environ["EVAL_API_KEYS"])
    else:
        data = json.loads(DEMO_KEYS.read_text(encoding="utf-8"))
    data.pop("_note", None)
    return data


def check(scenario: dict, final: dict) -> list[str]:
    """Compare the final state with the scenario expectations.

    Args:
        scenario: One line of `scenarios.jsonl`.
        final: Final graph state, or the state-like dict built from an API response.

    Returns:
        A list of failure descriptions; empty when the scenario passes.
    """
    failures = []
    agent = resolved_agent(final)
    route = final.get("route") if agent else None
    answer = (final.get("answer") or "").lower()
    plan = list(final.get("plan") or [])
    first_agent = (plan[0] if plan else agent) if agent else None
    if "expected_agent" in scenario and scenario["expected_agent"] != first_agent:
        failures.append(f"first agent {first_agent!r} != {scenario['expected_agent']!r}")
    if "expected_final_agent" in scenario and scenario["expected_final_agent"] != agent:
        failures.append(f"final agent {agent!r} != {scenario['expected_final_agent']!r}")
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
    names = [call.name for call in final.get("tool_calls", [])]
    if "expected_tool" in scenario and scenario["expected_tool"] not in names:
        failures.append(f"tool {scenario['expected_tool']!r} not called ({names})")
    for tool in scenario.get("expected_tools", []):
        if tool not in names:
            failures.append(f"tool {tool!r} not called ({names})")
    types = {source.type for source in final.get("sources", [])}
    if "expected_source_type" in scenario and scenario["expected_source_type"] not in types:
        failures.append(f"no source of type {scenario['expected_source_type']!r} ({sorted(types)})")
    for source_type in scenario.get("expected_source_types", []):
        if source_type not in types:
            failures.append(f"no source of type {source_type!r} ({sorted(types)})")
    if "expected_plan" in scenario and list(final.get("plan") or []) != scenario["expected_plan"]:
        failures.append(f"plan {final.get('plan')} != {scenario['expected_plan']}")
    if "expected_hops" in scenario and int(final.get("hops", 0)) != scenario["expected_hops"]:
        failures.append(f"hops {final.get('hops', 0)} != {scenario['expected_hops']}")
    if "expected_security_event" in scenario:
        events = final.get("security_events", [])
        if not any(scenario["expected_security_event"] in event for event in events):
            failures.append(f"security event {scenario['expected_security_event']!r} missing")
    return failures


def state_from_response(body: dict) -> dict:
    """Rebuild the subset of the final state that `check` needs from an API response.

    Args:
        body: JSON returned by `POST /chat`.

    Returns:
        A state-like dict with the fields `check` reads.
    """
    agent = body.get("agent")
    route = SimpleNamespace(agent=agent, intent=body.get("intent")) if agent else None
    category = body.get("guardrail") or (None if agent else body.get("intent"))
    guardrail = SimpleNamespace(category=category) if category else None
    sources = [Source(**item) for item in body.get("sources", [])]
    tool_calls = [ToolCall(name=name, args={}, ok=True) for name in body.get("tool_calls", [])]
    return {
        "route": route,
        "current_agent": agent,
        "guardrail": guardrail,
        "handoff": body.get("handoff", False),
        "answer": body.get("response", ""),
        "sources": sources,
        "tool_calls": tool_calls,
        "plan": body.get("plan", []),
        "hops": body.get("hops", 0),
        "security_events": body.get("security_events", []),
        "evidence": body.get("evidence") or [],
        "trace": body.get("trace") or [],
        "ticket_id": body.get("ticket_id"),
        "conversation_id": body.get("conversation_id"),
    }


class ApiClient:
    """HTTP access to a running service with the demo credentials."""

    def __init__(self, base_url: str, keys: dict[str, str], operator_key: str) -> None:
        """Remember the base URL and the credentials."""
        self._base = base_url.rstrip("/")
        self._keys = keys
        self._operator_key = operator_key

    def chat(self, merchant: str, message: str, conversation_id: str | None) -> httpx.Response:
        """Send one message as the given merchant."""
        payload = {"message": message, "user_id": merchant, "include_evidence": True}
        if conversation_id:
            payload["conversation_id"] = conversation_id
        return httpx.post(
            f"{self._base}/chat",
            json=payload,
            headers={"X-API-Key": self._keys.get(merchant, "")},
            timeout=180.0,
        )

    def chat_as(self, key_owner: str, body_user_id: str, message: str) -> httpx.Response:
        """Send a message with one merchant's key and another merchant's `user_id`."""
        return httpx.post(
            f"{self._base}/chat",
            json={"message": message, "user_id": body_user_id},
            headers={"X-API-Key": self._keys.get(key_owner, "")},
            timeout=180.0,
        )

    def reply(self, ticket_id: str, message: str) -> httpx.Response:
        """Answer a ticket as the operator."""
        return httpx.post(
            f"{self._base}/tickets/{ticket_id}/reply",
            json={"message": message, "resolution": "resolved"},
            headers={"X-API-Key": self._operator_key},
            timeout=180.0,
        )

    def conversation(self, merchant: str, conversation_id: str) -> httpx.Response:
        """Read a conversation as its merchant."""
        return httpx.get(
            f"{self._base}/conversations/{conversation_id}",
            headers={"X-API-Key": self._keys.get(merchant, "")},
            timeout=30.0,
        )


class InProcessClient:
    """Runs turns through the compiled graph, sharing a conversation like the API does."""

    def __init__(self) -> None:
        """Compile the graph once."""
        self._graph = get_graph()
        self._conversations = repositories.get_conversation_repository()

    def chat(self, merchant: str, message: str, conversation_id: str | None) -> dict:
        """Run one turn and return the final state."""
        principal = Principal(
            merchant_id=merchant, auth_method="api_key", scopes=("chat",), key_id="eval"
        )
        if conversation_id is None:
            conversation_id, turn = self._conversations.create(merchant), 1
        else:
            conversation = self._conversations.get(conversation_id, merchant)
            turn = sum(1 for item in conversation.turns if item.role == "merchant") + 1
        thread_id = new_thread_id(conversation_id, turn)
        state = initial_state(
            uuid.uuid4().hex, principal, message, conversation_id, turn, thread_id
        )
        final = self._graph.invoke(state, config=thread_config(thread_id))
        final["conversation_id"] = conversation_id
        return final


def run_scenario(scenario: dict, client) -> tuple[dict, list[str]]:
    """Execute one scenario and return the final state and the failures.

    Args:
        scenario: One line of `scenarios.jsonl`.
        client: `ApiClient` or `InProcessClient`.

    Returns:
        The final state (possibly state-like) and the list of failures.
    """
    api = isinstance(client, ApiClient)
    if "expected_status" in scenario:
        response = client.chat_as(scenario["api_key_of"], scenario["user_id"], scenario["message"])
        failures = []
        if response.status_code != scenario["expected_status"]:
            failures.append(f"status {response.status_code} != {scenario['expected_status']}")
        return {"answer": response.text[:200]}, failures
    turns = scenario.get("turns") or [scenario["message"]]
    conversation_id = None
    final: dict = {}
    for message in turns:
        result = client.chat(scenario["user_id"], message, conversation_id)
        if api:
            result.raise_for_status()
            final = state_from_response(result.json())
        else:
            final = result
        conversation_id = final.get("conversation_id")
    failures = check(scenario, final)
    roundtrip = scenario.get("operator_roundtrip")
    if roundtrip and api:
        ticket_id = final.get("ticket_id")
        if not ticket_id:
            return final, [*failures, "no ticket to reply to"]
        reply = client.reply(ticket_id, roundtrip["reply"])
        if reply.status_code != 200:
            failures.append(f"operator reply status {reply.status_code}")
        conversation = client.conversation(scenario["user_id"], conversation_id).json()
        roles = [turn["role"] for turn in conversation.get("turns", [])]
        if "operator" not in roles:
            failures.append(f"no operator turn in conversation ({roles})")
        follow = client.chat(scenario["user_id"], roundtrip["follow_up"], conversation_id)
        follow.raise_for_status()
        final = state_from_response(follow.json())
        answer = (final.get("answer") or "").lower()
        if not any(word.lower() in answer for word in roundtrip["must_contain_any"]):
            failures.append(f"follow-up lacks any of {roundtrip['must_contain_any']}")
    return final, failures


def main(argv: list[str] | None = None) -> int:
    """Run every scenario and print a summary table.

    Args:
        argv: Command-line arguments; `--api URL` evaluates a running service.

    Returns:
        0 when all scenarios pass, 1 otherwise.
    """
    parser = argparse.ArgumentParser(description="Evaluate the challenge scenarios.")
    parser.add_argument("--api", help="Base URL of a running service, e.g. http://localhost:8080")
    parser.add_argument(
        "--operator-key",
        default=os.environ.get("OPERATOR_API_KEY", ""),
        help="Operator credential for the round-trip scenario (API mode); defaults to the"
        " demo operator key in the key file.",
    )
    parser.add_argument(
        "--keys", help="JSON file with merchant API keys (default: EVAL_API_KEYS or demo keys)."
    )
    parser.add_argument("--only", help="Comma-separated scenario ids to run, for quick iteration.")
    parser.add_argument(
        "--out", help="Result file; defaults to evals/results/<date>-regression.json"
    )
    args = parser.parse_args(argv)
    if args.api:
        keys = load_keys(args.keys)
        operator_key = args.operator_key or keys.get("operator_demo", "")
        client = ApiClient(args.api, keys, operator_key)
    else:
        client = InProcessClient()
    only = {int(item) for item in args.only.split(",")} if args.only else None
    results = []
    for scenario in load_scenarios():
        if only is not None and scenario["id"] not in only:
            continue
        mode = scenario.get("api_mode")
        if (mode == "only" and not args.api) or (mode == "skip" and args.api):
            print(f"{scenario['id']:>3}  SKIP  {scenario.get('note', mode + ' in this mode')}")
            continue
        started = time.perf_counter()
        try:
            final, failures = run_scenario(scenario, client)
            agent = resolved_agent(final) or "-"
            handoff = bool(final.get("handoff")) if agent != "-" else False
            answer = final.get("answer") or ""
            evidence = [
                item.model_dump() if hasattr(item, "model_dump") else item
                for item in final.get("evidence", [])
            ]
            security = list(final.get("security_events", []))
            trace = [
                {
                    "node": step["node"] if isinstance(step, dict) else step.node,
                    "ms": step["ms"] if isinstance(step, dict) else step.ms,
                    "model": (step.get("model") if isinstance(step, dict) else step.model) or "",
                }
                for step in final.get("trace", [])
            ]
        except Exception as exc:
            failures = [f"exception {type(exc).__name__}: {exc}"]
            agent, handoff, answer, evidence, security, trace = "-", False, "", [], [], []
        latency = int((time.perf_counter() - started) * 1000)
        results.append(
            {
                "id": scenario["id"],
                "message": scenario["message"],
                "turns": scenario.get("turns"),
                "user_id": scenario["user_id"],
                "expected_agent": scenario.get("expected_agent"),
                "agent": agent,
                "handoff": handoff,
                "latency_ms": latency,
                "failures": failures,
                "answer": answer,
                "evidence": evidence,
                "security_events": security,
                "trace": trace,
                "reference_answer": scenario.get("reference_answer", ""),
                "must_be_grounded": scenario.get("must_be_grounded", True),
                "expected_status": scenario.get("expected_status"),
            }
        )
        status = "PASS" if not failures else "FAIL"
        print(
            f"{scenario['id']:>3}  {status}  {str(scenario.get('expected_agent')):<10} {agent:<10} "
            f"handoff={str(handoff):<5} {latency:>6} ms  {'; '.join(failures)}"
        )
    passed = sum(1 for item in results if not item["failures"])
    latencies = sorted(item["latency_ms"] for item in results)
    median = latencies[len(latencies) // 2] if latencies else 0
    print(f"\n{passed}/{len(results)} scenarios passed, median latency {median} ms")
    RESULTS_DIR.mkdir(exist_ok=True)
    suffix = "-partial" if only else ""
    out = (
        Path(args.out)
        if args.out
        else RESULTS_DIR / f"{datetime.now(UTC).date().isoformat()}-regression{suffix}.json"
    )
    out.write_text(
        json.dumps(
            {
                "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "mode": "api" if args.api else "in-process",
                "passed": passed,
                "total": len(results),
                "median_latency_ms": median,
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"results written to {out}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
