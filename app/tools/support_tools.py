"""LangChain tools that give the Support agent access to merchant data.

The docstrings double as the tool descriptions the model reads, so they describe what
each tool returns rather than how it is implemented. The merchant is an injected
argument: it is absent from the schema the model sees and is supplied by the runtime
from the authenticated principal (see `app.agents.support.run_tool`).
"""

from typing import Annotated

from langchain_core.tools import InjectedToolArg, tool

from app import repositories

USER_NOT_FOUND = {"ok": False, "error": "user_not_found"}


def _repository():
    return repositories.get_repository()


@tool
def get_user_profile(merchant_id: Annotated[str, InjectedToolArg]) -> dict:
    """Return the merchant's name, plan, terminal model, account status and bank account link."""
    user = _repository().get_user(merchant_id)
    if user is None:
        return dict(USER_NOT_FOUND)
    return {"ok": True, **user}


@tool
def get_settlement_status(merchant_id: Annotated[str, InjectedToolArg]) -> dict:
    """Return the merchant's recent sales: date, amount, method, status and expected deposit date.

    The result also includes today's date so that relative expressions such as
    "yesterday" can be resolved against real dates.
    """
    sales = _repository().get_settlements(merchant_id)
    if sales is None:
        return dict(USER_NOT_FOUND)
    return {"ok": True, "today": _repository().today().isoformat(), "sales": sales}


@tool
def get_terminal_status(merchant_id: Annotated[str, InjectedToolArg]) -> dict:
    """Return the card machine model, connectivity, network, last heartbeat and recent errors."""
    terminal = _repository().get_terminal(merchant_id)
    if terminal is None:
        return dict(USER_NOT_FOUND)
    return {"ok": True, **terminal}


@tool
def request_product_knowledge(question: str) -> dict:
    """Ask the Knowledge agent to add official product information or a procedure.

    Use it when the merchant's data shows that an official procedure, a product rule or
    a fee explanation is needed to finish the answer, for example the reset steps for an
    offline card machine or how receivables advance affects a deposit. Phrase the
    question as a product question without any merchant data. The Knowledge agent runs
    after you and merges its answer with yours.

    Args:
        question: The product question to pass on.
    """
    return {"ok": True, "note": "the Knowledge agent will add the official information"}


MERCHANT_DATA_TOOLS = [get_user_profile, get_settlement_status, get_terminal_status]
SUPPORT_TOOLS = [*MERCHANT_DATA_TOOLS, request_product_knowledge]
