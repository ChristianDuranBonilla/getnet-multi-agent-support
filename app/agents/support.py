"""Support agent: a bounded tool-calling loop over merchant data."""

import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app import llm
from app.agents.prompts import render_prompt
from app.observability import timed_node
from app.state import AgentState, Source, ToolCall
from app.tools.support_tools import SUPPORT_TOOLS

MAX_ITERATIONS = 4
TOOLS_BY_NAME = {tool.name: tool for tool in SUPPORT_TOOLS}


def run_tool(name: str, args: dict) -> tuple[dict, bool]:
    """Execute a tool by name and report whether it succeeded.

    Args:
        name: Tool name as requested by the model.
        args: Tool arguments.

    Returns:
        A tuple of (result, ok). Unknown tools and exceptions yield ok=False.
    """
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        return {"ok": False, "error": "unknown_tool"}, False
    try:
        result = tool.invoke(args)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}, False
    return result, bool(result.get("ok", True))


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
    return "ok"


@timed_node("support")
def support_node(state: AgentState) -> dict:
    """Resolve account, settlement and terminal questions with tools.

    Args:
        state: Current graph state.

    Returns:
        State update with `answer`, `tool_calls`, `sources` and possibly `handoff`.
    """
    system = render_prompt(
        "support", language=state.get("language", "en"), user_id=state["user_id"]
    )
    messages = [SystemMessage(content=system), HumanMessage(content=state["message"])]
    model = llm.chat(tools=SUPPORT_TOOLS)
    calls: list[ToolCall] = []
    handoff = False
    reason = ""
    answer = ""
    for _ in range(MAX_ITERATIONS):
        try:
            response = model.invoke(messages)
        except Exception as exc:
            return {
                "answer": None,
                "tool_calls": calls,
                "handoff": True,
                "handoff_reason": f"support model error: {type(exc).__name__}",
                "trace_detail": "model error",
            }
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            answer = llm.message_text(response.content)
            break
        for call in tool_calls:
            result, ok = run_tool(call["name"], call.get("args", {}))
            calls.append(
                ToolCall(
                    name=call["name"], args=call.get("args", {}), ok=ok, summary=summarise(result)
                )
            )
            if not ok:
                handoff, reason = True, result.get("error", "tool error")
            elif result.get("account_status") == "under_review":
                handoff, reason = True, "account under review"
            messages.append(
                ToolMessage(content=json.dumps(result, default=str), tool_call_id=call["id"])
            )
    else:
        last = messages[-1]
        answer = llm.message_text(last.content) if isinstance(last, AIMessage) else ""
        handoff, reason = True, "tool loop exhausted"
    if not answer.strip():
        handoff, reason = True, reason or "empty answer"
    return {
        "answer": answer or None,
        "tool_calls": calls,
        "sources": [Source(type="tool", ref=call.name) for call in calls],
        "handoff": handoff,
        "handoff_reason": reason,
        "trace_detail": f"{len(calls)} tool calls",
    }
