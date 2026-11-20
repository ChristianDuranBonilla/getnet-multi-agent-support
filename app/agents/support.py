"""Support agent: a bounded tool-calling loop over the authenticated merchant's data.

The node reports what the tools returned (`tool_calls`, `evidence`); whether a failed
lookup or an account under review must go to a human is the replanner's policy.
"""

import json
import logging
from typing import NamedTuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app import llm
from app.agents.prompts import render_history, render_partial_answers, render_prompt
from app.auth import Principal
from app.observability import log_event, model_of, timed_node, usage_of
from app.state import AgentState, Evidence, Source, ToolCall
from app.tools.support_tools import SUPPORT_TOOLS, request_product_knowledge

logger = logging.getLogger(__name__)
MAX_ITERATIONS = 4
TOOLS_BY_NAME = {tool.name: tool for tool in SUPPORT_TOOLS}
IDENTITY_KEYS = frozenset({"user_id", "merchant_id", "customer_id", "account_id"})
IDENTITY_STEMS = frozenset(key.replace("_", "") for key in IDENTITY_KEYS)
HOP_TOOL = request_product_knowledge.name


def is_identity_key(name: str) -> bool:
    """Return whether an argument name denotes an identity, in any spelling.

    `user_id`, `userId`, `USER_ID`, `user-id` and `user_id ` all count.
    """
    return "".join(char for char in name.lower() if char.isalpha()) in IDENTITY_STEMS


class ToolOutcome(NamedTuple):
    """Result of executing one tool call through the identity boundary."""

    result: dict
    ok: bool
    args: dict
    override_attempted: bool


def allowed_arguments(tool) -> set[str]:
    """Return the argument names the model is allowed to supply for a tool.

    This is the schema the model is shown (`tool_call_schema`), so injected arguments
    such as the merchant id are never in it.
    """
    return set(tool.tool_call_schema.model_json_schema().get("properties", {}))


def run_tool(name: str, args: dict, principal: Principal) -> ToolOutcome:
    """Execute a tool on behalf of the authenticated merchant.

    This is the only code path that invokes a Support tool. Arguments are filtered by
    allowlist: only the names in the tool's model-facing schema pass. Anything else is
    dropped; when the dropped name denotes an identity in any spelling it is logged and
    reported as a security event. The merchant id of the principal is then injected, so
    the model decides which operation runs but never on whose data.

    Args:
        name: Tool name as requested by the model.
        args: Tool arguments as produced by the model.
        principal: Authenticated caller.

    Returns:
        The tool result, whether it succeeded, the arguments actually executed and
        whether the model tried to supply an identity.
    """
    tool = TOOLS_BY_NAME.get(name)
    allowed = allowed_arguments(tool) if tool is not None else set()
    clean = {key: value for key, value in args.items() if key in allowed}
    dropped = [str(key) for key in args if key not in allowed]
    stray = [key[:40] for key in dropped if not is_identity_key(key)][:10]
    override = len(stray) < len(dropped)
    if override:
        log_event(
            logger,
            "identity_override_attempt",
            level=logging.WARNING,
            tool=name,
            key_id=principal.key_id,
        )
    if stray:
        log_event(logger, "unexpected_tool_arguments", tool=name, arguments=stray)
    if tool is None:
        return ToolOutcome({"ok": False, "error": "unknown_tool"}, False, clean, override)
    try:
        result = tool.invoke({**clean, "merchant_id": principal.merchant_id})
    except Exception as exc:
        error = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return ToolOutcome(error, False, clean, override)
    return ToolOutcome(result, bool(result.get("ok", True)), clean, override)


def summarise(result: dict) -> str:
    """Return a short description of a tool result for the trace."""
    if not result.get("ok", True):
        return result.get("error", "error")
    if "sales" in result:
        return f"{len(result['sales'])} sales"
    if "connectivity" in result:
        return f"{result.get('model')} {result.get('connectivity')}"
    if "account_status" in result:
        return f"account {result.get('account_status')}"
    if "note" in result:
        return "knowledge requested"
    return "ok"


def _evidence(name: str, result: dict) -> Evidence:
    return Evidence(type="tool", ref=name, excerpt=json.dumps(result, default=str)[:600])


@timed_node("support")
def support_node(state: AgentState) -> dict:
    """Resolve account, settlement and terminal questions with tools.

    Args:
        state: Current graph state.

    Returns:
        State update with `answer`, `tool_calls`, `sources`, `evidence`, possibly
        `handoff`, `hop_request` and `security_events`.
    """
    principal: Principal = state["principal"]
    system = render_prompt(
        "support",
        language=state.get("language", "en"),
        history=render_history(state.get("summary", ""), state.get("history", [])),
        partial_answers=render_partial_answers(state.get("partial_answers", [])),
    )
    question = state.get("hop_question") or state["message"]
    messages = [SystemMessage(content=system), HumanMessage(content=question)]
    model = llm.chat(tools=SUPPORT_TOOLS)
    calls: list[ToolCall] = []
    evidence: list[Evidence] = []
    security_events: list[str] = []
    hop_request = None
    handoff = False
    reason = ""
    answer = ""
    tokens_in = tokens_out = 0
    model_name = ""
    for _ in range(MAX_ITERATIONS):
        try:
            response = llm.invoke_resilient(model, messages)
        except Exception as exc:
            return {
                "current_agent": "support",
                "answer": None,
                "tool_calls": calls,
                "evidence": evidence,
                "security_events": security_events,
                "handoff": True,
                "handoff_reason": f"support model error: {type(exc).__name__}",
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "trace_detail": "model error",
            }
        used_in, used_out = usage_of(response)
        tokens_in, tokens_out = tokens_in + used_in, tokens_out + used_out
        model_name = model_of(response) or model_name
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            answer = llm.message_text(response.content)
            break
        for call in tool_calls:
            outcome = run_tool(call["name"], call.get("args", {}), principal)
            calls.append(
                ToolCall(
                    name=call["name"],
                    args=outcome.args,
                    ok=outcome.ok,
                    summary=summarise(outcome.result),
                    account_status=outcome.result.get("account_status"),
                )
            )
            if outcome.override_attempted:
                security_events.append(f"identity_override_attempt:{call['name']}")
            if call["name"] == HOP_TOOL:
                hop_request = {"agent": "knowledge", "question": outcome.args.get("question", "")}
            else:
                evidence.append(_evidence(call["name"], outcome.result))
            messages.append(
                ToolMessage(
                    content=json.dumps(outcome.result, default=str), tool_call_id=call["id"]
                )
            )
    else:
        last = messages[-1]
        answer = llm.message_text(last.content) if isinstance(last, AIMessage) else ""
        handoff, reason = True, "tool loop exhausted"
    if not answer.strip():
        handoff, reason = True, reason or "empty answer"
    data_calls = [call for call in calls if call.name != HOP_TOOL]
    return {
        "current_agent": "support",
        "answer": answer or None,
        "tool_calls": calls,
        "sources": [Source(type="tool", ref=call.name) for call in data_calls],
        "evidence": evidence,
        "security_events": security_events,
        "hop_request": hop_request,
        "handoff": handoff,
        "handoff_reason": reason,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "model": model_name,
        "trace_detail": f"{len(data_calls)} tool calls" + (" +hop" if hop_request else ""),
    }
