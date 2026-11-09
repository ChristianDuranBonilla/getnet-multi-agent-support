"""LangChain tools that give the Support agent access to merchant data.

The docstrings double as the tool descriptions the model reads, so they describe
what each tool returns rather than how it is implemented.
"""

from langchain_core.tools import tool

from app import repositories

USER_NOT_FOUND = {"ok": False, "error": "user_not_found"}


def _repository():
    return repositories.get_repository()


@tool
def get_user_profile(user_id: str) -> dict:
    """Return the merchant's name, plan, terminal model, account status and bank account link.

    Args:
        user_id: Merchant identifier.
    """
    user = _repository().get_user(user_id)
    if user is None:
        return dict(USER_NOT_FOUND)
    return {"ok": True, **user}


@tool
def get_settlement_status(user_id: str) -> dict:
    """Return the merchant's recent sales: date, amount, method, status and expected deposit date.

    Args:
        user_id: Merchant identifier.
    """
    sales = _repository().get_settlements(user_id)
    if sales is None:
        return dict(USER_NOT_FOUND)
    return {"ok": True, "today": _repository().today().isoformat(), "sales": sales}


@tool
def get_terminal_status(user_id: str) -> dict:
    """Return the card machine model, connectivity, network, last heartbeat and recent errors.

    Args:
        user_id: Merchant identifier.
    """
    terminal = _repository().get_terminal(user_id)
    if terminal is None:
        return dict(USER_NOT_FOUND)
    return {"ok": True, **terminal}


SUPPORT_TOOLS = [get_user_profile, get_settlement_status, get_terminal_status]
