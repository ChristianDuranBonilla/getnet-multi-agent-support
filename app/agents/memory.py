"""Memory nodes: load the conversation at the start of a turn, store it at the end."""

import logging
from datetime import UTC, datetime

from langchain_core.messages import HumanMessage, SystemMessage

from app import llm, repositories
from app.agents.guardrail import BLOCKED_CATEGORIES
from app.config import get_settings
from app.observability import timed_node
from app.state import AgentState, ConversationSummary, Evidence, Turn

logger = logging.getLogger(__name__)

SUMMARY_PROMPT = (
    "Summarise the conversation between a Getnet merchant and the support assistant in"
    " at most 120 words, in the language of the conversation ({language}). Keep facts the"
    " assistant will need later: the merchant's device, the problem, what was answered,"
    " open tickets and what the operator said. Do not include card numbers or documents."
)


@timed_node("recall")
def recall_node(state: AgentState) -> dict:
    """Load the recent turns and the rolling summary of the conversation.

    Args:
        state: Current graph state with `conversation_id` and `principal`.

    Returns:
        State update with `history` and `summary`.
    """
    conversation = repositories.get_conversation_repository().get(
        state.get("conversation_id") or "", state["principal"].merchant_id
    )
    if conversation is None:
        return {"history": [], "summary": "", "trace_detail": "new conversation"}
    window = 2 * get_settings().memory_window_turns
    history = conversation.turns[-window:] if window else []
    operator_replies = [
        Evidence(type="operator", ref="operator_reply", excerpt=turn.content[:600])
        for turn in history
        if turn.role == "operator"
    ]
    return {
        "history": history,
        "summary": conversation.summary,
        "evidence": operator_replies,
        "trace_detail": f"{len(history)} turns" + (" +summary" if conversation.summary else ""),
    }


def summarise(
    turns: list[Turn], previous: str, language: str
) -> tuple[str | None, tuple[int, int], str]:
    """Ask the model for a rolling summary.

    Args:
        turns: Every turn of the conversation, oldest first.
        previous: The summary stored so far, possibly empty.
        language: ISO code of the conversation language.

    Returns:
        The new summary, the token usage and the model; `None` as the summary when the
        model could not be reached, so the caller keeps the cursor where it was and the
        turns are summarised on a later attempt.
    """
    transcript = "\n".join(f"{turn.role}: {turn.content}" for turn in turns)
    if previous:
        transcript = f"Previous summary: {previous}\n{transcript}"
    try:
        model = llm.structured(ConversationSummary)
        result = llm.invoke_resilient(
            model,
            [
                SystemMessage(content=SUMMARY_PROMPT.format(language=language)),
                HumanMessage(content=transcript),
            ],
        )
    except Exception as exc:
        logger.warning("summary model call failed: %s", type(exc).__name__)
        return None, (0, 0), ""
    if isinstance(result, dict):
        result = ConversationSummary.model_validate(result)
    return result.summary, getattr(model, "last_usage", (0, 0)), getattr(model, "last_model", "")


def stored_message(state: AgentState) -> str:
    """Return what is persisted as the merchant's turn.

    The sanitized text is stored, so masked PII never persists. A message the guardrail
    blocked is replaced by a marker: storing it would feed the injection or the unsafe
    request back into every later prompt through the history block.
    """
    verdict = state.get("guardrail")
    if verdict is not None and verdict.category in BLOCKED_CATEGORIES:
        return f"[message not stored: {verdict.category}]"
    return state["message"]


@timed_node("remember")
def remember_node(state: AgentState) -> dict:
    """Append this turn to the conversation and refresh the summary when due.

    The merchant turn is stored after sanitisation and blocked messages are replaced by
    a marker (see `stored_message`). When the turn is the resumption of an escalation,
    only the operator's reply is stored.

    Args:
        state: Final graph state of the turn.

    Returns:
        State update with the token usage of the summary call, if any.
    """
    repository = repositories.get_conversation_repository()
    conversation_id = state.get("conversation_id") or ""
    merchant_id = state["principal"].merchant_id
    if not conversation_id or repository.get(conversation_id, merchant_id) is None:
        return {"trace_detail": "no conversation"}
    now = datetime.now(UTC)
    answer = state.get("answer") or ""
    if state.get("operator_reply") is not None:
        repository.append(
            conversation_id, Turn(role="operator", content=answer, agent="operator", created_at=now)
        )
        detail = "operator turn"
    else:
        repository.append(
            conversation_id, Turn(role="merchant", content=stored_message(state), created_at=now)
        )
        agent = state.get("current_agent") or "guardrail"
        repository.append(
            conversation_id, Turn(role="assistant", content=answer, agent=agent, created_at=now)
        )
        detail = "2 turns"
    conversation = repository.get(conversation_id, merchant_id)
    tokens = (0, 0)
    model = ""
    if conversation is not None:
        pending = len(conversation.turns) - conversation.summary_through_turn
        if pending >= get_settings().memory_summary_every:
            summary, tokens, model = summarise(
                conversation.turns[conversation.summary_through_turn :],
                conversation.summary,
                state.get("language", "en"),
            )
            if summary is not None:
                repository.update_summary(conversation_id, summary, len(conversation.turns))
                detail += " +summary"
            else:
                detail += " +summary deferred"
    return {"tokens_in": tokens[0], "tokens_out": tokens[1], "model": model, "trace_detail": detail}
