from app import repositories
from app.tools.support_tools import get_settlement_status, get_terminal_status, get_user_profile


def test_settlements_use_iso_dates_relative_to_today(repository):
    sales = repository.get_settlements("cliente1988")
    yesterday = next(sale for sale in sales if sale["sale_id"] == "S-77201")
    assert yesterday["sale_date"] == "2026-11-05"
    assert yesterday["expected_deposit_date"] == "2026-12-05"
    assert "days_ago" not in yesterday


def test_unknown_user_returns_none(repository):
    assert repository.get_user("cliente0000") is None
    assert repository.get_settlements("cliente0000") is None
    assert repository.get_terminal("cliente0000") is None


def test_ticket_roundtrip(repository):
    ticket_id = repository.create_ticket("cliente1988", "deposit question", "medium")
    assert ticket_id.startswith("TCK-")
    ticket = repository.get_ticket(ticket_id)
    assert ticket["user_id"] == "cliente1988"
    assert ticket["status"] == "open"


def test_tools_wrap_repository(monkeypatch, repository):
    monkeypatch.setattr(repositories, "get_repository", lambda: repository)
    profile = get_user_profile.invoke({"user_id": "cliente1988"})
    assert profile["ok"] is True
    assert profile["terminal_model"] == "Get Smart"
    settlements = get_settlement_status.invoke({"user_id": "cliente1988"})
    assert settlements["ok"] is True
    assert len(settlements["sales"]) == 3
    terminal = get_terminal_status.invoke({"user_id": "cliente2001"})
    assert terminal["connectivity"] == "offline"


def test_tools_report_unknown_user_without_raising(monkeypatch, repository):
    monkeypatch.setattr(repositories, "get_repository", lambda: repository)
    result = get_settlement_status.invoke({"user_id": "nobody"})
    assert result == {"ok": False, "error": "user_not_found"}
