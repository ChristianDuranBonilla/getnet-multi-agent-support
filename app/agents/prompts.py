"""Load agent prompts from Markdown files."""

from functools import lru_cache

from app.config import get_settings


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
