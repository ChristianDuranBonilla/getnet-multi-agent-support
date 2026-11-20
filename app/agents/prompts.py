"""Load agent prompts from Markdown files and render their shared blocks."""

from functools import lru_cache

from app.config import get_settings
from app.state import PartialAnswer, Turn


@lru_cache
def load_prompt(name: str) -> str:
    """Return the raw prompt text for an agent.

    Args:
        name: File stem under `app/prompts/`, for example `router`.

    Returns:
        The prompt with `{placeholders}` left intact for `str.format`.
    """
    path = get_settings().prompts_dir / f"{name}.md"
    return path.read_text(encoding="utf-8").strip()


def render_prompt(name: str, **values: str) -> str:
    """Return a prompt with its placeholders filled.

    Args:
        name: File stem under `app/prompts/`.
        **values: Placeholder values.

    Returns:
        The rendered prompt.
    """
    return load_prompt(name).format(**values)


def render_history(summary: str, history: list[Turn]) -> str:
    """Format the conversation memory for a prompt.

    Args:
        summary: Rolling summary of the older turns, possibly empty.
        history: Most recent turns, oldest first.

    Returns:
        A block headed "Conversation so far:", or a note that this is the first message.
    """
    if not summary and not history:
        return "Conversation so far: this is the first message."
    lines = ["Conversation so far:"]
    if summary:
        lines.append(f"Summary of earlier turns: {summary}")
    lines.extend(f"{turn.role}: {turn.content}" for turn in history)
    return "\n".join(lines)


def render_partial_answers(partial_answers: list[PartialAnswer]) -> str:
    """Format the answers of specialists that already ran in this turn.

    Args:
        partial_answers: Answers produced before a hop, oldest first.

    Returns:
        An instruction block, or an empty string when no specialist ran before.
    """
    if not partial_answers:
        return ""
    lines = [
        "Another specialist already answered part of this request. Write ONE combined",
        "answer that keeps their facts and adds yours; do not repeat a handoff notice.",
    ]
    for partial in partial_answers:
        lines.append(f"{partial.agent} agent said: {partial.answer}")
        if partial.tool_summaries:
            lines.append("Data it used: " + "; ".join(partial.tool_summaries))
    return "\n".join(lines)
